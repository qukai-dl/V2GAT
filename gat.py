"""
Graph Attention Network Module - 频率特定的图特征提取
使用多头注意力机制从动态学习的图结构中进行消息传递
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadGraphAttention(nn.Module):
    """
    多头图注意力层

    论文公式:
        h^{multi}_{i,k} = ||_{h=1}^{N_h} σ(Σ_{j∈N^{phys}(i)} α_{i,j,k}(t) W_g^{(h)} h_{j,k})

    输出:
        z_{i,k} = W_proj h^{multi}_{i,k} + b_proj
    """

    def __init__(self, d_model, d_out=None, num_heads=4, dropout=0.1):
        """
        参数:
            d_model: 输入维度
            d_out: 输出维度 (默认等于输入)
            num_heads: 注意力头数
            dropout: Dropout比率
        """
        super(MultiHeadGraphAttention, self).__init__()

        if d_out is None:
            d_out = d_model

        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_out = d_out

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        # 多头投影
        self.W = nn.ModuleList([
            nn.Linear(d_model, self.d_head, bias=False)
            for _ in range(num_heads)
        ])

        # 输出投影
        self.out_proj = nn.Linear(d_model, d_out)
        self.bias = nn.Parameter(torch.zeros(d_out))

        self.dropout = nn.Dropout(dropout)
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, h, adj):
        """
        图注意力前向传播

        参数:
            h: 节点嵌入 [batch, N, K, d_model] 或 [batch, N, d_model]
            adj: 邻接矩阵 [batch, N, N] 或 [batch, K, N, N]

        返回:
            out: 更新后的节点表示
        """
        # 处理不同形状的输入
        if h.dim() == 4:
            # [batch, N, K, d_model] -> 需要为每个模态分别处理
            batch_size, num_farms, num_modes, d_model = h.shape

            outputs = []
            for k in range(num_modes):
                h_k = h[:, :, k, :]  # [batch, N, d_model]
                if adj.dim() == 4:
                    adj_k = adj[:, k, :, :]  # [batch, N, N]
                else:
                    adj_k = adj

                out_k = self._single_mode_attention(h_k, adj_k)
                outputs.append(out_k)

            out = torch.stack(outputs, dim=2)  # [batch, N, K, d_out]
        else:
            out = self._single_mode_attention(h, adj)

        return out

    def _single_mode_attention(self, h, adj):
        """
        单模态注意力

        参数:
            h: [batch, N, d_model]
            adj: [batch, N, N]

        返回:
            out: [batch, N, d_out]
        """
        batch_size, num_farms, d_model = h.shape

        # 多头注意力的拼接
        heads = []
        for head_idx in range(self.num_heads):
            # 线性变换
            h_transformed = self.W[head_idx](h)  # [batch, N, d_head]

            # 注意力分数: Q * K^T / sqrt(d_head)
            # 使用h作为query，邻接节点作为key
            # 简化: 直接用内积计算注意力
            attention_scores = torch.matmul(h_transformed, h_transformed.transpose(1, 2))  # [batch, N, N]
            attention_scores = attention_scores / math.sqrt(self.d_head)

            # 应用LeakyReLU
            attention_scores = self.leaky_relu(attention_scores)

            # 使用邻接矩阵进行mask
            # adj中0表示无边，但我们想要保持自环
            # 首先确保对角线有值（自环）
            diag_mask = torch.eye(num_farms, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
            adj_masked = adj.clone()
            # 如果adj没有自环，手动添加
            adj_masked = adj_masked + diag_mask

            # mask掉没有连接的节点
            attention_scores = attention_scores.masked_fill(adj_masked < 1e-10, float('-inf'))

            # softmax归一化
            attention_weights = F.softmax(attention_scores, dim=-1)  # [batch, N, N]
            attention_weights = self.dropout(attention_weights)

            # 消息传递: 聚合邻居节点
            aggregated = torch.matmul(attention_weights, h_transformed)  # [batch, N, d_head]

            heads.append(aggregated)

        # 拼接多头
        concatenated = torch.cat(heads, dim=-1)  # [batch, N, d_model]

        # 输出投影
        out = self.out_proj(concatenated) + self.bias

        return out


class FrequencySpecificGAT(nn.Module):
    """
    频率特定的图注意力网络

    为每个VMD模态维护独立的GAT分支

    结构:
        - 多头图注意力层
        - 残差连接
        - 层归一化
    """

    def __init__(self, d_model, d_out=None, num_heads=4, num_layers=2, dropout=0.1):
        """
        参数:
            d_model: 嵌入维度
            d_out: 输出维度
            num_heads: 注意力头数
            num_layers: GAT层数
            dropout: Dropout比率
        """
        super(FrequencySpecificGAT, self).__init__()

        if d_out is None:
            d_out = d_model

        self.d_model = d_model
        self.d_out = d_out
        self.num_heads = num_heads
        self.num_layers = num_layers

        # 多层GAT
        self.gat_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        for layer_idx in range(num_layers):
            in_dim = d_model if layer_idx == 0 else d_out
            self.gat_layers.append(
                MultiHeadGraphAttention(in_dim, d_out, num_heads, dropout)
            )
            self.layer_norms.append(nn.LayerNorm(d_out))

    def forward(self, h, adj_matrices):
        """
        频率特定的GAT前向传播

        参数:
            h: 节点嵌入 [batch, N, K, d_model]
            adj_matrices: 动态邻接矩阵 [batch, K, N, N]

        返回:
            z: 图增强的节点表示 [batch, N, K, d_out]
        """
        batch_size, num_farms, num_modes, d_model = h.shape

        x = h
        for layer_idx in range(self.num_layers):
            # 并行处理所有模态
            x_new = self.gat_layers[layer_idx](x, adj_matrices)

            # 残差连接和层归一化
            if x.shape[-1] == x_new.shape[-1]:
                x = self.layer_norms[layer_idx](x_new + x)
            else:
                x = self.layer_norms[layer_idx](x_new)

        return x


class GraphFeatureExtractor(nn.Module):
    """
    图特征提取器 - 整合频率特定的GAT

    输出:
        z_{i,k} - 风电场i在VMD模式k下的图增强节点表示
    """

    def __init__(self, d_model, d_out=None, num_heads=4, num_layers=2, dropout=0.1):
        """
        参数:
            d_model: 输入维度
            d_out: 输出维度
            num_heads: 注意力头数
            num_layers: GAT层数
            dropout: Dropout比率
        """
        super(GraphFeatureExtractor, self).__init__()

        self.gat = FrequencySpecificGAT(d_model, d_out, num_heads, num_layers, dropout)

    def forward(self, h, adj_matrices):
        """
        参数:
            h: 节点嵌入 [batch, N, K, d_model]
            adj_matrices: 动态邻接矩阵 [batch, K, N, N]

        返回:
            z: 图特征 [batch, N, K, d_out]
        """
        z = self.gat(h, adj_matrices)
        return z