"""
V2GAT Model - 主模型整合
Physics-Informed Variational Volatility-Decoupled Graph Attention Network

论文完整流程:
1. VMD分解: 将历史功率信号分解为K个变分模态分量
2. 节点编码: a_{i,k} = W_enc M^{vmd}_{i,k} + b_enc, h_{i,k} = a_{i,k} + SPE(i)
3. 动态图构建: 为每个模态构建物理引导的动态图
4. 图特征提取: 使用GAT从动态图中提取特征
5. 层级融合: 多头注意力融合不同频率的特征
6. 预测: 结合未来NWP数据进行多步预测
"""

import torch
import torch.nn as nn
from .vmd import VMDDecomposer
from .encoder import WindFarmEncoder
from .graph_constructor import DynamicGraphConstructor
from .gat import GraphFeatureExtractor
from .fusion import ModeAwareFusion
from .forecasting import MultiFarmForecaster


class V2GAT(nn.Module):
    """
    V2GAT: Physics-Informed Variational Volatility-Decoupled Graph Attention Network

    用于超短期区域风功率预测的频率感知动态图网络
    """

    def __init__(self, num_farms, seq_len, nwp_features_dim=3, d_model=64,
                 K=3, num_heads=4, hidden_dim=128, ell=100.0, beta=1.0,
                 forecast_horizon=8, coords=None, **kwargs):
        """
        参数:
            num_farms: 风电场数量N
            seq_len: 历史窗口长度L
            nwp_features_dim: NWP特征维度 (风速、风向sin/cos)
            d_model: 嵌入维度D
            K: VMD模态数量
            num_heads: 注意力头数
            hidden_dim: MLP隐藏层维度
            ell: 距离衰减尺度
            beta: 物理先验强度
            forecast_horizon: 预测步数H
            coords: 风电场坐标 [N, 2]
        """
        super(V2GAT, self).__init__()

        self.num_farms = num_farms
        self.seq_len = seq_len
        self.d_model = d_model
        self.K = K
        self.forecast_horizon = forecast_horizon

        # 1. VMD分解模块
        self.vmd_decomposer = VMDDecomposer(K=K, **kwargs)

        # 2. 节点编码模块
        self.node_encoder = WindFarmEncoder(
            num_farms=num_farms,
            seq_len=seq_len,
            d_model=d_model,
            coords=coords
        )

        # 3. 动态图构建模块
        self.graph_constructor = DynamicGraphConstructor(
            num_farms=num_farms,
            d_model=d_model,
            hidden_dim=hidden_dim,
            ell=ell,
            beta=beta
        )

        # 初始化空间参数 (需要在外部设置)
        self._spatial_params_initialized = False

        # 4. 图特征提取模块 (GAT)
        self.graph_extractor = GraphFeatureExtractor(
            d_model=d_model,
            d_out=d_model,
            num_heads=num_heads,
            num_layers=2,
            dropout=0.1
        )

        # 5. 层级融合模块
        self.mode_fusion = ModeAwareFusion(
            d_model=d_model,
            num_heads=num_heads
        )

        # 6. 预测头
        self.forecaster = MultiFarmForecaster(
            d_model=d_model,
            nwp_features_dim=nwp_features_dim,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2
        )

    def set_spatial_params(self, bearing_angle, distance_matrix, boundary_mask):
        """
        设置空间参数

        参数:
            bearing_angle: 航向角矩阵 [N, N]
            distance_matrix: 距离矩阵 [N, N]
            boundary_mask: 边界掩码 [N, N]
        """
        self.graph_constructor.set_spatial_params(bearing_angle, distance_matrix, boundary_mask)
        self._spatial_params_initialized = True

    def forward(self, power_signals, nwp_hist, nwp_future):
        """
        V2GAT完整前向传播

        参数:
            power_signals: 历史功率信号 [batch, N, L]
                          论文中的 X^{vmd} 或 M
            nwp_hist: 历史NWP数据 [batch, L, N, nwp_features_dim]
                     包含: [wind_speed, sin(wind_dir), cos(wind_dir)]
            nwp_future: 未来NWP数据 [batch, H, N, nwp_features_dim]

        返回:
            predictions: 预测结果 [batch, N, H]
        """
        batch_size = power_signals.shape[0]

        # ========== 步骤1: VMD分解 ==========
        # 将功率信号分解为K个模态分量
        vmd_modes = self.vmd_decomposer(power_signals)  # [batch, N, K, L]

        # ========== 步骤2: 节点编码 ==========
        # 将VMD模态序列编码为固定维度表示，并融合地理感知SPE
        h = self.node_encoder(vmd_modes)  # [batch, N, K, d_model]

        # ========== 步骤3: 动态图构建 ==========
        # 确保空间参数已设置
        if not self._spatial_params_initialized:
            raise RuntimeError("Spatial parameters not initialized. Call set_spatial_params() first.")

        # 构建物理引导的动态邻接矩阵
        adj_matrices = self.graph_constructor(h, nwp_hist)  # [batch, K, N, N]

        # ========== 步骤4: 图特征提取 ==========
        # 使用GAT从每个频率特定的图中提取特征
        z_graph = self.graph_extractor(h, adj_matrices)  # [batch, N, K, d_model]

        # ========== 步骤5: 层级融合 ==========
        # 使用多头注意力自适应融合不同频率的图特征
        z_fused = self.mode_fusion(z_graph)  # [batch, N, d_model]

        # ========== 步骤6: 预测 ==========
        # 结合未来NWP数据进行多步预测
        predictions = self.forecaster(z_fused, nwp_hist, nwp_future)  # [batch, N, H]

        return predictions

    def get_graph_weights(self):
        """
        获取当前学习的图权重，用于可解释性分析

        返回:
            adj_matrices: 归一化的邻接矩阵 [batch, K, N, N]
        """
        return self._last_adj_matrices

    def compute_loss(self, predictions, targets):
        """
        计算MSE损失

        参数:
            predictions: 模型预测 [batch, N, H]
            targets: 真实值 [batch, N, H]

        返回:
            loss: MSE损失
        """
        loss = nn.MSELoss()(predictions, targets)
        return loss


def create_v2gat_model(config):
    """
    根据配置创建V2GAT模型

    参数:
        config: 配置字典

    返回:
        model: V2GAT模型实例
    """
    model = V2GAT(
        num_farms=config['num_farms'],
        seq_len=config['seq_len'],
        nwp_features_dim=config.get('nwp_features_dim', 3),
        d_model=config.get('d_model', 64),
        K=config.get('K', 3),
        num_heads=config.get('num_heads', 4),
        hidden_dim=config.get('hidden_dim', 128),
        ell=config.get('ell', 100.0),
        beta=config.get('beta', 1.0),
        forecast_horizon=config.get('forecast_horizon', 8),
        coords=config.get('coords', None)
    )

    return model