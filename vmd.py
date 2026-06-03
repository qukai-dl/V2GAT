"""
VMD (Variational Mode Decomposition) Module
用于将风功率信号分解为多个频率特定的变分模态分量
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import hilbert
import numpy as np


class VMD(nn.Module):
    """
    Variational Mode Decomposition
    将信号分解为K个具有自适应中心频率的模态分量

    论文中: 对每个风电场的功率时间序列应用VMD分解
    M ∈ R^{L × N × K}, M_{t,i,k} 表示风电场i在时刻t的第k个排序VMD模态
    """

    def __init__(self, K=3, tau=0.0, tol=1e-7, max_iter=500):
        """
        参数:
            K: 模态数量 (默认3)
            tau: 时间常数 (噪声容忍度)
            tol: 收敛容忍度
            max_iter: 最大迭代次数
        """
        super(VMD, self).__init__()
        self.K = K
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, signal):
        """
        对输入信号进行VMD分解

        参数:
            signal: 输入信号，形状为 [batch, seq_len] 或 [seq_len]

        返回:
            modes: 分解后的K个模态分量，形状为 [batch, K, seq_len]
        """
        if signal.dim() == 1:
            signal = signal.unsqueeze(0)
            squeeze_flag = True
        else:
            squeeze_flag = False

        batch_size, seq_len = signal.shape
        signal = signal.double()

        # 频域表示
        f = torch.fft.fft(signal, dim=-1)

        # 初始化模态中心频率
        f_sum = torch.sum(f, dim=1, keepdim=True)
        f_abs = torch.abs(f_sum)
        f_angle = torch.angle(f_sum)

        # 均匀初始化中心频率
        omega = torch.linspace(0, 0.5, seq_len // 2 + 1, device=signal.device, dtype=signal.dtype)
        omega = omega.repeat(batch_size, self.K, 1)  # [batch, K, seq_len//2+1]

        # 模态初始化
        u_hat = torch.zeros(batch_size, self.K, seq_len, dtype=signal.dtype, device=signal.device)

        lambda_hat = torch.zeros_like(f)  # 拉格朗日乘子

        # 迭代优化
        for n_iter in range(self.max_iter):
            # 更新模态
            for k in range(self.K):
                # 计算分子和分母
                numerator = f - torch.sum(u_hat, dim=1, keepdim=True).squeeze(1) + lambda_hat / 2
                denominator = 1 + tau * (omega[:, k:k+1, :seq_len//2+1] ** 2).squeeze(1)

                u_hat[:, k, :seq_len//2+1] = numerator[:, :seq_len//2+1] / denominator
                u_hat[:, k, seq_len//2+1:] = torch.conj(u_hat[:, k, 1:seq_len//2].flip(1))

                # 更新中心频率
                spec = u_hat[:, k, :seq_len//2+1]
                power_spec = torch.abs(spec) ** 2

                freq = omega[:, k, :seq_len//2+1]
                numerator_freq = torch.sum(freq * power_spec, dim=-1)
                denominator_freq = torch.sum(power_spec, dim=-1) + 1e-10
                omega[:, k, :seq_len//2+1] = numerator_freq / denominator_freq

            # 更新拉格朗日乘子
            lambda_hat = lambda_hat + tau * (f - torch.sum(u_hat, dim=1, keepdim=True).squeeze(1))

            # 检查收敛
            diff = torch.norm(f - torch.sum(u_hat, dim=1, keepdim=True).squeeze(1))
            if diff < self.tol:
                break

        # 重组信号
        modes = torch.zeros(batch_size, self.K, seq_len, dtype=signal.dtype, device=signal.device)
        for k in range(self.K):
            modes[:, k, :] = torch.fft.ifft(u_hat[:, k, :], dim=-1).real

        # 按中心频率排序 (低频在前)
        center_freqs = torch.sum(torch.abs(torch.fft.fft(modes, dim=-1))[:, :, :seq_len//2+1], dim=-1)
        _, indices = torch.sort(center_freqs, dim=1)
        modes = modes.gather(1, indices.unsqueeze(-1).expand_as(modes))

        if squeeze_flag:
            modes = modes.squeeze(0)

        return modes


class VMDDecomposer(nn.Module):
    """
    封装VMD模块，用于批量处理多个风电场的数据

    输入: [batch, N, L] - batch个样本，N个风电场，L个时间步
    输出: [batch, N, K, L] - K个VMD模态分量
    """

    def __init__(self, K=3, **kwargs):
        super(VMDDecomposer, self).__init__()
        self.K = K
        self.vmd = VMD(K=K, **kwargs)

    def forward(self, power_signals):
        """
        参数:
            power_signals: 风电功率信号 [batch, N, L]
        返回:
            modes: VMD模态分量 [batch, N, K, L]
        """
        batch_size, num_farms, seq_len = power_signals.shape

        # 对每个风电场分别进行VMD分解
        modes = []
        for i in range(num_farms):
            farm_signal = power_signals[:, i, :]  # [batch, L]
            farm_modes = self.vmd(farm_signal)    # [batch, K, L]
            modes.append(farm_modes)

        modes = torch.stack(modes, dim=1)  # [batch, N, K, L]
        return modes