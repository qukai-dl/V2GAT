"""
Physics-Guided Dynamic Graph Construction Module
基于物理先验（风方向对齐、距离衰减、可行传播约束）的动态图学习
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PhysicsGuidedEdgeLearner(nn.Module):
    """
    物理引导的动态边权重学习器

    对于每个VMD模态k，构建独立的动态图 G_k(t) = (V, E_k(t))

    论文公式:
        g_{i,j}(t) = [d_{i,j}, sin(φ_{i,j}), cos(φ_{i,j}), v_{t,j},
                      sin(θ_{t,j}), cos(θ_{t,j}), c_{i,j}(t), b_{i,j}]^T

        e_{i,j,k}(t) = MLP(h_{i,k} || h_{j,k} || g_{i,j}(t))
                       + β log(P^{phys}_{i,j}(t) + ε)

        其中 P^{phys}_{i,j}(t) = b_{i,j} · exp(-d_{i,j}/ℓ) · (ε + c_{i,j}(t))
    """

    def __init__(self, d_model, hidden_dim=128):
        """
        参数:
            d_model: 节点嵌入维度
            hidden_dim: MLP隐藏层维度
        """
        super(PhysicsGuidedEdgeLearner, self).__init__()

        # 物理特征: d_{i,j}, sin(φ), cos(φ), v, sin(θ), cos(θ), c, b
        # 共8维
        self.edge_feature_dim = 8

        # MLP边权重学习器
        # 输入: h_i || h_j || g_{i,j}
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2 + self.edge_feature_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def compute_physical_prior(self, wind_speed, wind_direction,
                              bearing_angle, distance, boundary_mask, ell=100.0, beta=1.0, eps=1e-6):
        """
        计算物理引导的边先验

        参数:
            wind_speed: 风速 [batch, N] 或标量
            wind_direction: 风方向 [batch, N]
            bearing_angle: 航向角 [N, N] - 从源到目标的航向
            distance: 距离矩阵 [N, N]
            boundary_mask: 边界掩码 [N, N] - b_{i,j}
            ell: 距离衰减尺度
            beta: 物理先验强度
            eps: 数值稳定性常数

        返回:
            prior: 物理先验 [batch, N, N]
        """
        batch_size = wind_speed.shape[0]
        num_farms = wind_speed.shape[1]

        # 风方向对齐 c_{i,j}(t) = max(0, cos(θ_t,j - φ_{i,j}))
        # wind_direction [batch, N], bearing_angle [N, N]
        wind_dir_expanded = wind_direction.unsqueeze(2)  # [batch, N, 1]
        bearing_expanded = bearing_angle.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]

        cos_angle = torch.cos(wind_dir_expanded - bearing_expanded)  # [batch, N, N]
        alignment = torch.clamp(cos_angle, min=0)  # c_{i,j}(t)

        # 距离衰减 exp(-d_{i,j}/ℓ)
        distance_expanded = distance.unsqueeze(0).unsqueeze(0)  # [1, 1, N, N]
        decay = torch.exp(-distance_expanded / ell)

        # 边界掩码
        boundary = boundary_mask.unsqueeze(0)  # [1, N, N]

        # 物理先验
        prior = boundary * decay * (eps + alignment)  # [batch, N, N]

        # 取对数并乘以β
        prior_log = beta * torch.log(prior + eps)

        return prior_log

    def forward(self, h, wind_speed, wind_direction, bearing_angle,
                distance, boundary_mask, ell=100.0, beta=1.0):
        """
        计算动态边权重

        参数:
            h: 节点嵌入 [batch, N, K, d_model]
            wind_speed: 历史风速 [batch, N]
            wind_direction: 历史风方向 [batch, N]
            bearing_angle: 航向角 [N, N]
            distance: 距离矩阵 [N, N]
            boundary_mask: 边界掩码 [N, N]
            ell: 距离衰减尺度
            beta: 物理先验强度

        返回:
            adjacency: 归一化邻接矩阵 [batch, K, N, N]
        """
        batch_size, num_farms, num_modes, d_model = h.shape

        # 物理先验
        prior_log = self.compute_physical_prior(
            wind_speed, wind_direction, bearing_angle, distance, boundary_mask, ell, beta
        )  # [batch, N, N]

        # 为每个模态分别计算边权重
        adjacency_matrices = []

        for k in range(num_modes):
            h_k = h[:, :, k, :]  # [batch, N, d_model]

            # 构建所有节点对的特征
            # h_i: [batch, N, 1, d_model]
            # h_j: [batch, 1, N, d_model]
            h_i = h_k.unsqueeze(2)  # [batch, N, 1, d_model]
            h_j = h_k.unsqueeze(1)  # [batch, 1, N, d_model]

            # 拼接 h_i || h_j -> [batch, N, N, 2*d_model]
            h_concat = torch.cat([h_i.expand(-1, -1, num_farms, -1),
                                  h_j.expand(-1, num_farms, -1, -1)], dim=-1)

            # 构建物理特征 g_{i,j}
            # 距离 [1, 1, N, N]
            d_expanded = distance.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

            # sin/cos 航向角
            sin_bearing = torch.sin(bearing_angle).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
            cos_bearing = torch.cos(bearing_angle).unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)

            # 风速 [batch, N, 1, 1]
            v_expanded = wind_speed.unsqueeze(2).unsqueeze(3)

            # sin/cos 风方向
            sin_wd = torch.sin(wind_direction).unsqueeze(2).unsqueeze(3)
            cos_wd = torch.cos(wind_direction).unsqueeze(2).unsqueeze(3)

            # 对齐 c_{i,j}
            cos_angle = torch.cos(wind_direction.unsqueeze(2) - bearing_angle.unsqueeze(0).unsqueeze(0))
            alignment = torch.clamp(cos_angle, min=0)  # [batch, N, N]

            # 边界掩码
            b_expanded = boundary_mask.unsqueeze(0).expand(batch_size, -1, -1)

            # 拼接所有物理特征 [batch, N, N, 8]
            g = torch.stack([
                d_expanded, sin_bearing, cos_bearing,
                v_expanded.expand(-1, -1, num_farms, -1),
                sin_wd.expand(-1, -1, num_farms, -1),
                cos_wd.expand(-1, -1, num_farms, -1),
                alignment.unsqueeze(3),
                b_expanded.unsqueeze(3)
            ], dim=-1).squeeze(-2)  # [batch, N, N, 8]

            # MLP计算边权重
            # 重塑以便批量处理
            h_concat_flat = h_concat.reshape(batch_size * num_farms * num_farms, -1)
            g_flat = g.reshape(batch_size * num_farms * num_farms, -1)

            # 拼接 h_i || h_j || g
            mlp_input = torch.cat([h_concat_flat, g_flat], dim=-1)
            e = self.mlp(mlp_input)  # [batch*N*N, 1]
            e = e.reshape(batch_size, num_farms, num_farms)

            # 加上物理先验
            e = e + prior_log.unsqueeze(1)  # 广播到K维度

            # 边界感知的masked softmax归一化
            # α_{i,j,k}(t) = I(b_{i,j}>0) exp(e_{i,j,k}(t)) / Σ_m I(b_{i,m}>0) exp(e_{i,m,k}(t))
            masked_exp = torch.where(
                boundary_mask.unsqueeze(0).expand(batch_size, -1, -1) > 0,
                torch.exp(e),
                torch.zeros_like(e)
            )
            denom = masked_exp.sum(dim=-1, keepdim=True) + 1e-10
            alpha = masked_exp / denom

            adjacency_matrices.append(alpha)

        # 堆叠所有模态的邻接矩阵 [batch, K, N, N]
        adjacency = torch.stack(adjacency_matrices, dim=1)

        return adjacency


class DynamicGraphConstructor(nn.Module):
    """
    动态图结构学习器 - 为每个VMD模态构建独立的动态图

    该模块整合了:
    1. 物理引导的边权重学习
    2. 边界感知的masked softmax归一化
    """

    def __init__(self, num_farms, d_model, hidden_dim=128, ell=100.0, beta=1.0):
        """
        参数:
            num_farms: 风电场数量N
            d_model: 嵌入维度D
            hidden_dim: MLP隐藏层维度
            ell: 距离衰减尺度
            beta: 物理先验强度
        """
        super(DynamicGraphConstructor, self).__init__()
        self.num_farms = num_farms
        self.d_model = d_model

        self.edge_learner = PhysicsGuidedEdgeLearner(d_model, hidden_dim)
        self.ell = ell
        self.beta = beta

    def forward(self, h, nwp_features):
        """
        构建动态图

        参数:
            h: 节点嵌入 [batch, N, K, d_model]
            nwp_features: 气象特征 [batch, N, F]
                      包含: [power, wind_speed, sin(wind_dir), cos(wind_dir)]

        返回:
            adj_matrices: 动态邻接矩阵 [batch, K, N, N]
        """
        # 提取风速和风方向
        wind_speed = nwp_features[:, :, 1]  # [batch, N]
        wind_direction = torch.atan2(nwp_features[:, :, 2], nwp_features[:, :, 3])  # sin/cos -> angle

        # 这些应该是预先计算好的（地理坐标、距离矩阵、边界掩码）
        # 在实际使用中应该作为模块的buffer或输入参数

        return self.edge_learner(h, wind_speed, wind_direction, self.bearing_angle,
                                 self.distance_matrix, self.boundary_mask,
                                 self.ell, self.beta)

    def set_spatial_params(self, bearing_angle, distance_matrix, boundary_mask):
        """
        设置空间参数（通常在模型初始化时设置一次）

        参数:
            bearing_angle: 航向角矩阵 [N, N]
            distance_matrix: 距离矩阵 [N, N]
            boundary_mask: 边界掩码 [N, N]
        """
        self.register_buffer('bearing_angle', bearing_angle)
        self.register_buffer('distance_matrix', distance_matrix)
        self.register_buffer('boundary_mask', boundary_mask)