"""
V2GAT - Physics-Informed Variational Volatility-Decoupled Graph Attention Network
超短期区域风功率预测模型

模块结构:
- vmd: 变分模态分解 (Variational Mode Decomposition)
- encoder: 节点编码器 (VMD模态编码 + 地理感知空间位置编码)
- graph_constructor: 物理引导动态图构建
- gat: 图注意力网络 (频率特定的图特征提取)
- fusion: 层级VMD模态融合
- forecasting: 预测头 (LSTM + 全连接层)
- v2gat: 主模型整合
"""

from .vmd import VMD, VMDDecomposer
from .encoder import NodeEncoder, GeographyAwareSPE, WindFarmEncoder
from .graph_constructor import PhysicsGuidedEdgeLearner, DynamicGraphConstructor
from .gat import MultiHeadGraphAttention, FrequencySpecificGAT, GraphFeatureExtractor
from .fusion import MultiHeadAttentionFusion, HierarchicalModeFusion, ModeAwareFusion
from .forecasting import NWPEncoder, ForecasterHead, MultiFarmForecaster
from .v2gat import V2GAT, create_v2gat_model

__all__ = [
    'VMD',
    'VMDDecomposer',
    'NodeEncoder',
    'GeographyAwareSPE',
    'WindFarmEncoder',
    'PhysicsGuidedEdgeLearner',
    'DynamicGraphConstructor',
    'MultiHeadGraphAttention',
    'FrequencySpecificGAT',
    'GraphFeatureExtractor',
    'MultiHeadAttentionFusion',
    'HierarchicalModeFusion',
    'ModeAwareFusion',
    'NWPEncoder',
    'ForecasterHead',
    'MultiFarmForecaster',
    'V2GAT',
    'create_v2gat_model'
]

__version__ = '1.0.0'