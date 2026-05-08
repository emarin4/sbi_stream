"""Neural network models for sbi_stream."""

from .layers import GNN, GNNBlock, MLP, TransformerA, TransformerB, MultiHeadAttentionBlock
from .layers import CNN, ConvBlock, ResidualBlock, BottleneckBlock, ResNet, build_resnet
from .flows import build_flows
from .lightning import GNNEmbedding, TransformerEmbedding, CNNEmbedding, NPE, NRE, SequentialNPE
from .utils import get_activation, configure_optimizers, build_embedding_loss

__all__ = [
    'GNN',
    'GNNBlock',
    'MLP',
    'TransformerA',
    'TransformerB',
    'MultiHeadAttentionBlock',
    'CNN',
    'ConvBlock',
    'ResidualBlock',
    'BottleneckBlock',
    'ResNet',
    'build_resnet',
    'GNNEmbedding',
    'TransformerEmbedding',
    'CNNEmbedding',
    'NPE',
    'NRE',
    'SequentialNPE',
    'build_flows',
    'get_activation',
    'configure_optimizers',
    'build_embedding_loss',
]
