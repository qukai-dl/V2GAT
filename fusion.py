"""
Hierarchical VMD-Mode Fusion Module - 层级VMD模态融合
使用多头注意力机制自适应融合不同频率的图特征

论文公式:
    Z_j = [z_{j,1}, z_{j,2}, ..., z_{j,K}] ∈ R^{K × D_out}
    Q_j = z_{j,1} (最低频率模态作为Query)
    K_j = V_j = Z_j

    Q_j^{(h)} = Q_j W_q^{(h)}, K_j^{(h)} = K_j W_k^{(h)}, V_j^{(h)} = V_j W_v^{(h)}
    head_j^{(h)} = Attention(Q_j^{(h)}, K_j^{(h)}, V_j^{(h)})
                = softmax(Q_j^{(h)} K_j^{(h)T} / sqrt(d_head)) V_j^{(h)}

    z_{j,attn} = Concat(head_j^{(1)}, ..., head_j^{(N_h)}) W_o
    z_{j,fused} = LayerNorm(z_{j,attn} + Q_j)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttentionFusion(nn.Module):
    """
    多头注意力融合模块

    使用最低频率模态作为Query，自适应融合所有频率的图特征
    """

    def __init__(self, d_model, num_heads=4):
        """
        参数:
            d_model: 嵌入维度
            num_heads: 注意力头数
        """
        super(MultiHeadAttentionFusion, self).__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V 投影
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, Z, query_idx=0):
        """
        多头注意力融合

        参数:
            Z: 模态特征序列 [batch, N, K, d_model]
                对于目标风电场j，Z_j = [z_{j,1}, ..., z_{j,K}]
            query_idx: 作为Query的模态索引 (默认0，即最低频率模态)

        返回:
            fused: 融合后的特征 [batch, N, d_model]
        """
        batch_size, num_farms, num_modes, d_model = Z.shape

        # 选择Query: 最低频率模态
        Q = Z[:, :, query_idx, :]  # [batch, N, d_model]

        # K和V使用所有模态
        K = Z.reshape(batch_size, num_farms, num_modes, d_model)  # [batch, N, K, d_model]
        V = Z.reshape(batch_size, num_farms, num_modes, d_model)

        # 批量计算注意力
        # 重塑为多头形式
        # Q: [batch, N, 1, d_model] -> 扩展以便与所有Key交互
        # 实际上要对每个风电场独立计算注意力

        # 对每个风电场分别计算
        outputs = []
        for b in range(batch_size):
            for n in range(num_farms):
                q = Q[b, n, :]  # [d_model]
                v_seq = V[b, n, :, :]  # [K, d_model]

                # 计算Q, K, V
                q = self.W_q(q)  # [d_model]
                k_seq = self.W_k(v_seq)  # [K, d_model]
                v_seq = self.W_v(v_seq)  # [K, d_model]

                # 重塑为多头
                q = q.reshape(self.num_heads, self.d_head)  # [N_h, d_head]
                k = k_seq.reshape(num_modes, self.num_heads, self.d_head)  # [K, N_h, d_head]
                v = v_seq.reshape(num_modes, self.num_heads, self.d_head)  # [K, N_h, d_head]

                # 跨模态计算注意力
                head_outputs = []
                for h in range(self.num_heads):
                    q_h = q[h]  # [d_head]
                    k_h = k[:, h, :]  # [K, d_head]
                    v_h = v[:, h, :]  # [K, d_head]

                    # 注意力分数
                    scores = torch.matmul(q_h.unsqueeze(0), k_h.transpose(0, 1)) / math.sqrt(self.d_head)  # [1, K]
                    attn_weights = F.softmax(scores, dim=-1)  # [1, K]

                    # 加权求和
                    context = torch.matmul(attn_weights, v_h)  # [1, d_head]
                    head_outputs.append(context)

                # 拼接所有头
                head_concat = torch.cat(head_outputs, dim=0)  # [d_model]
                outputs.append(head_concat)

        # 重塑输出
        outputs = torch.stack(outputs, dim=0).reshape(batch_size, num_farms, d_model)

        # 输出投影
        out = self.W_o(outputs)

        # 残差连接和层归一化
        fused = self.layer_norm(out + Q)

        return fused


class HierarchicalModeFusion(nn.Module):
    """
    层级VMD模态融合模块

    对每个风电场，使用多头注意力机制自适应融合不同频率的图特征
    """

    def __init__(self, d_model, num_heads=4):
        """
        参数:
            d_model: 嵌入维度
            num_heads: 注意力头数
        """
        super(HierarchicalModeFusion, self).__init__()

        self.attention_fusion = MultiHeadAttentionFusion(d_model, num_heads)

    def forward(self, Z):
        """
        融合不同频率的图特征

        参数:
            Z: 模态特征 [batch, N, K, d_model]

        返回:
            fused: 融合后的特征 [batch, N, d_model]
                对应论文中的 z_{j,fused}
        """
        fused = self.attention_fusion(Z, query_idx=0)
        return fused


class ModeAwareFusion(nn.Module):
    """
    模态感知融合器 - 整合层级融合模块

    输出:
        对于目标风电场j: z_{j,fused} ∈ R^{1 × D_out}
        这是一个丰富的、目标特定的表示，融合了层级频率信息
    """

    def __init__(self, d_model, num_heads=4):
        """
        参数:
            d_model: 嵌入维度
            num_heads: 注意力头数
        """
        super(ModeAwareFusion, self).__init__()

        self.hierarchical_fusion = HierarchicalModeFusion(d_model, num_heads)

    def forward(self, z_graph):
        """
        参数:
            z_graph: 图特征 [batch, N, K, d_out]

        返回:
            fused_features: 融合后的特征 [batch, N, d_out]
        """
        fused = self.hierarchical_fusion(z_graph)
        return fused