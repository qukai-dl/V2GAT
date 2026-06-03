"""
Forecasting Head Module - 预测头
将融合的图特征和未来气象数据转换为风功率预测

论文公式:
    z^{nwp}_{hist,j} = Enc_{nwp}(X^{nwp}_{hist,:,j,:})

    u_{h,j} = [z_{j,fused} || z^{nwp}_{hist,j} || x^{nwp}_{L+h,j}]

    H_{lstm,j}, c_{lstm,j} = LSTM(U_j)

    o_{1,j} = ReLU(W_1 H_{lstm,j} + b_1)
    o_{2,j} = ReLU(W_2 o_{1,j} + b_2) + o_{1,j}  (残差连接)
    y_hat_j = W_3 o_{2,j} + b_3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NWPEncoder(nn.Module):
    """
    气象数据编码器

    将历史NWP序列编码为固定维度表示
    """

    def __init__(self, nwp_features_dim, d_model):
        """
        参数:
            nwp_features_dim: NWP特征维度 (风速、风向的sin/cos等)
            d_model: 输出嵌入维度
        """
        super(NWPEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(nwp_features_dim, d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.ReLU()
        )

    def forward(self, nwp_history):
        """
        编码历史NWP数据

        参数:
            nwp_history: 历史NWP数据 [batch, seq_len, N, nwp_features_dim]

        返回:
            encoded: 编码后的表示 [batch, N, d_model]
        """
        batch_size, seq_len, num_farms, nwp_dim = nwp_history.shape

        # 对每个风电场独立处理
        # 重塑为 [batch*N, seq_len, nwp_dim]
        nwp_reshaped = nwp_history.reshape(batch_size * num_farms, seq_len, nwp_dim)

        # 时间维度上取平均 (或使用其他聚合方式)
        nwp_encoded = self.encoder(nwp_reshaped)  # [batch*N, seq_len, d_model]
        nwp_encoded = nwp_encoded.mean(dim=1)  # [batch*N, d_model]

        # 重塑回 [batch, N, d_model]
        nwp_encoded = nwp_encoded.reshape(batch_size, num_farms, -1)

        return nwp_encoded


class ForecasterHead(nn.Module):
    """
    预测头

    将目标风电场的融合图特征和NWP数据转换为多步预测
    """

    def __init__(self, d_model, nwp_features_dim, hidden_dim=256, num_layers=2, dropout=0.2):
        """
        参数:
            d_model: 嵌入维度
            nwp_features_dim: NWP特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(ForecasterHead, self).__init__()

        # NWP编码器
        self.nwp_encoder = NWPEncoder(nwp_features_dim, d_model)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=d_model * 2 + nwp_features_dim,  # z_fused + z_nwp_hist + x_nwp_future
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层 with 残差连接
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)  # 输出单步预测

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, z_fused, nwp_hist, nwp_future):
        """
        前向传播

        参数:
            z_fused: 融合后的图特征 [batch, N, d_model]
                    对应论文中的 z_{j,fused}
            nwp_hist: 历史NWP数据 [batch, L, N, nwp_features_dim]
            nwp_future: 未来NWP数据 [batch, H, N, nwp_features_dim]

        返回:
            predictions: 预测序列 [batch, N, H]
        """
        batch_size, num_farms, d_model = z_fused.shape
        _, horizon, _, nwp_dim = nwp_future.shape

        # 编码历史NWP
        z_nwp_hist = self.nwp_encoder(nwp_hist)  # [batch, N, d_model]

        # 对每个风电场分别预测
        all_predictions = []

        for n in range(num_farms):
            z_fused_n = z_fused[:, n, :]  # [batch, d_model]
            z_nwp_n = z_nwp_hist[:, n, :]  # [batch, d_model]
            nwp_future_n = nwp_future[:, :, n, :]  # [batch, H, nwp_dim]

            # 构建每个预测步的输入
            # u_{h,j} = [z_{j,fused} || z^{nwp}_{hist,j} || x^{nwp}_{L+h,j}]
            seq_inputs = []
            for h in range(horizon):
                future_nwp = nwp_future_n[:, h, :]  # [batch, nwp_dim]

                # 拼接
                u_h = torch.cat([z_fused_n, z_nwp_n, future_nwp], dim=-1)  # [batch, 2*d_model + nwp_dim]
                seq_inputs.append(u_h)

            # 堆叠为序列
            U = torch.stack(seq_inputs, dim=1)  # [batch, H, 2*d_model + nwp_dim]

            # LSTM处理
            lstm_out, _ = self.lstm(U)  # [batch, H, hidden_dim]

            # 全连接层
            o1 = self.relu(self.fc1(lstm_out))  # [batch, H, hidden_dim]
            o1 = self.dropout(o1)

            o2 = self.relu(self.fc2(o1)) + o1  # 残差连接
            o2 = self.dropout(o2)

            # 输出预测
            y_hat = self.out(o2)  # [batch, H, 1]
            y_hat = y_hat.squeeze(-1)  # [batch, H]

            all_predictions.append(y_hat)

        # 堆叠所有风电场的预测
        predictions = torch.stack(all_predictions, dim=-1)  # [batch, H, N]

        # 转置为 [batch, N, H]
        predictions = predictions.transpose(1, 2)

        return predictions


class MultiFarmForecaster(nn.Module):
    """
    多风电场预测器

    对所有风电场并行预测
    """

    def __init__(self, d_model, nwp_features_dim, hidden_dim=256, num_layers=2, dropout=0.2):
        """
        参数:
            d_model: 嵌入维度
            nwp_features_dim: NWP特征维度
            hidden_dim: LSTM隐藏层维度
            num_layers: LSTM层数
            dropout: Dropout比率
        """
        super(MultiFarmForecaster, self).__init__()

        self.forecasting_head = ForecasterHead(
            d_model, nwp_features_dim, hidden_dim, num_layers, dropout
        )

    def forward(self, z_fused, nwp_hist, nwp_future):
        """
        前向传播

        参数:
            z_fused: 融合后的图特征 [batch, N, d_model]
            nwp_hist: 历史NWP数据 [batch, L, N, nwp_features_dim]
            nwp_future: 未来NWP数据 [batch, H, N, nwp_features_dim]

        返回:
            predictions: 预测结果 [batch, N, H]
        """
        predictions = self.forecasting_head(z_fused, nwp_hist, nwp_future)
        return predictions