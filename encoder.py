"""
Node Encoding Module - 风电场节点表示学习
包含: VMD模态编码 + 地理感知空间位置编码 (SPE)
"""

import torch
import torch.nn as nn
import math


class GeographyAwareSPE(nn.Module):
    """
    地理感知空间位置编码 (Spatial Positional Encoding)

    使用经纬度坐标的sinusoidal编码，注入真实空间拓扑信息

    论文公式:
        s^{lon}_{i,2d} = sin(lo_i / B^{2d/D})
        s^{lon}_{i,2d+1} = cos(lo_i / B^{2d/D})
        同理对纬度编码
        SPE(i) = W_spe [s^{lon}_i || s^{lat}_i]
    """

    def __init__(self, num_farms, d_model, B=10000):
        """
        参数:
            num_farms: 风电场数量
            d_model: 嵌入维度
            B: 空间波长上界 (默认10000)
        """
        super(GeographyAwareSPE, self).__init__()
        self.num_farms = num_farms
        self.d_model = d_model
        self.B = B

        # 经纬度归一化参数 (会在forward中计算)
        self.register_buffer('lon_min', torch.tensor(0.0))
        self.register_buffer('lon_max', torch.tensor(1.0))
        self.register_buffer('lat_min', torch.tensor(0.0))
        self.register_buffer('lat_max', torch.tensor(1.0))

        # 投影层
        self.proj = nn.Linear(d_model, d_model)

    def set_coordinates(self, coords):
        """
        设置风电场坐标并计算归一化参数

        参数:
            coords: 坐标 tensor [N, 2] - [lon, lat]
        """
        self.lon_min = coords[:, 0].min()
        self.lon_max = coords[:, 0].max()
        self.lat_min = coords[:, 1].min()
        self.lat_max = coords[:, 1].max()

    def forward(self, farm_ids=None, coords=None):
        """
        生成地理感知空间位置编码

        参数:
            farm_ids: 风电场索引 [N]
            coords: 归一化坐标 [N, 2] (可选)

        返回:
            spe: 空间位置编码 [N, d_model]
        """
        if coords is None:
            # 使用默认均匀分布
            positions = torch.arange(self.num_farms, dtype=torch.float32, device=self.proj.weight.device)
            positions = positions.unsqueeze(1) / self.num_farms
        else:
            # 归一化坐标到[0,1]
            lon_norm = (coords[:, 0] - self.lon_min) / (self.lon_max - self.lon_min + 1e-10)
            lat_norm = (coords[:, 1] - self.lat_min) / (self.lat_max - self.lat_min + 1e-10)
            positions = torch.stack([lon_norm, lat_norm], dim=1)

        d = self.d_model // 2
        # 计算sinusoidal编码
        embeddings = torch.zeros(self.num_farms, self.d_model, device=positions.device)

        for dim in range(d):
            freq = 1.0 / (self.B ** (2.0 * dim / self.d_model))
            embeddings[:, 2*dim] = torch.sin(positions[:, 0] * freq)
            embeddings[:, 2*dim + 1] = torch.cos(positions[:, 0] * freq)

        # 对纬度也进行编码 (复用后面的维度)
        for dim in range(d):
            freq = 1.0 / (self.B ** (2.0 * dim / self.d_model))
            offset = d  # 纬度编码从中间开始
            embeddings[:, offset + 2*dim] = torch.sin(positions[:, 1] * freq)
            embeddings[:, offset + 2*dim + 1] = torch.cos(positions[:, 1] * freq)

        # 投影
        spe = self.proj(embeddings)
        return spe


class NodeEncoder(nn.Module):
    """
    节点编码器 - 将VMD模态序列压缩为固定维度表示

    论文公式:
        a_{i,k} = W_enc M^{vmd}_{i,k} + b_enc

    其中:
        M^{vmd}_{i,k} ∈ R^L 是风电场i的第k个VMD模态
        a_{i,k} ∈ R^D 是编码后的表示
    """

    def __init__(self, seq_len, d_model):
        """
        参数:
            seq_len: 输入序列长度L
            d_model: 输出嵌入维度D
        """
        super(NodeEncoder, self).__init__()
        self.seq_len = seq_len
        self.d_model = d_model

        # 线性投影
        self.encoder = nn.Linear(seq_len, d_model)
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, vmd_modes):
        """
        编码VMD模态

        参数:
            vmd_modes: VMD模态 [batch, N, K, L]

        返回:
            embeddings: 节点嵌入 [batch, N, K, d_model]
        """
        batch_size, num_farms, num_modes, seq_len = vmd_modes.shape

        # 重塑为 [batch*N*K, L]
        vmd_flat = vmd_modes.reshape(batch_size, num_farms, num_modes, seq_len)
        vmd_flat = vmd_flat.reshape(batch_size * num_farms * num_modes, seq_len)

        # 线性编码
        embeddings = self.encoder(vmd_flat) + self.bias

        # 重塑回 [batch, N, K, d_model]
        embeddings = embeddings.reshape(batch_size, num_farms, num_modes, self.d_model)

        return embeddings


class WindFarmEncoder(nn.Module):
    """
    综合节点编码器 - 整合VMD模态编码和地理感知空间位置编码

    论文公式:
        h_{i,k} = a_{i,k} + SPE(i)

    输出:
        h_{i,k} ∈ R^D - 风电场i在VMD模式k下的最终嵌入
    """

    def __init__(self, num_farms, seq_len, d_model, coords=None, B=10000):
        """
        参数:
            num_farms: 风电场数量N
            seq_len: 历史窗口长度L
            d_model: 嵌入维度D
            coords: 风电场坐标 [N, 2] (可选)
            B: SPE波长上界
        """
        super(WindFarmEncoder, self).__init__()
        self.num_farms = num_farms
        self.d_model = d_model

        # VMD模态编码器
        self.mode_encoder = NodeEncoder(seq_len, d_model)

        # 地理感知空间位置编码
        self.spe = GeographyAwareSPE(num_farms, d_model, B)
        if coords is not None:
            self.spe.set_coordinates(coords)

    def forward(self, vmd_modes, farm_ids=None):
        """
        生成风电场节点嵌入

        参数:
            vmd_modes: VMD模态 [batch, N, K, L]
            farm_ids: 风电场索引 (可选)

        返回:
            h: 节点嵌入 [batch, N, K, d_model]
        """
        # VMD模态编码
        a = self.mode_encoder(vmd_modes)  # [batch, N, K, d_model]

        # 地理感知SPE
        spe = self.spe(farm_ids)  # [N, d_model]

        # 融合: h = a + SPE
        h = a + spe.unsqueeze(0).unsqueeze(2)  # [batch, N, K, d_model]

        return h