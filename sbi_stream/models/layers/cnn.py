"""2D CNN feature extractor layers: plain CNN and ResNet-style."""

from typing import Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Single convolutional block: Conv2d -> [BN] -> act -> [Dropout2d] -> [MaxPool2d].

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int
        Default: 3.
    stride : int
        Default: 1.
    batch_norm : bool
        Default: True.
    dropout : float
        Dropout2d probability. Default: 0.0.
    act : nn.Module
        Activation function instance. Default: nn.ReLU().
    downsample : bool
        Apply 2×2 MaxPool2d after activation. Default: False.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        batch_norm: bool = True,
        dropout: float = 0.0,
        act: nn.Module = None,
        downsample: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
        )
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else None
        self.act = act if act is not None else nn.ReLU()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else None
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if downsample else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if self.pool is not None:
            x = self.pool(x)
        return x


class CNN(nn.Module):
    """2D CNN feature extractor built from :class:`ConvBlock` layers.

    A final AdaptiveAvgPool2d produces a fixed-size feature vector regardless
    of input spatial dimensions.

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    channels : list of int
        Output channels for each convolutional block.
    kernel_size : int or list of int
        Kernel size per block. Default: 3.
    stride : int or list of int
        Stride per block. Default: 1.
    downsample : bool or list of bool
        Apply 2×2 MaxPool after each block. Default: False.
    batch_norm : bool
        Use BatchNorm2d in every block. Default: True.
    dropout : float
        Dropout2d probability in every block. Default: 0.0.
    act : nn.Module
        Activation function instance shared across all blocks. Default: nn.ReLU().
    pooling_output_size : int
        Spatial size for the final AdaptiveAvgPool2d. Default: 1 (global avg pool).
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        kernel_size: Union[int, List[int]] = 3,
        stride: Union[int, List[int]] = 1,
        downsample: Union[bool, List[bool]] = False,
        batch_norm: bool = True,
        dropout: float = 0.0,
        act: nn.Module = None,
        pooling_output_size: int = 1,
    ):
        super().__init__()

        if act is None:
            act = nn.ReLU()

        n = len(channels)
        kernel_sizes = [kernel_size] * n if isinstance(kernel_size, int) else list(kernel_size)
        strides = [stride] * n if isinstance(stride, int) else list(stride)
        downsamples = [downsample] * n if isinstance(downsample, bool)  else list(downsample)

        self.blocks = nn.ModuleList()
        in_ch = in_channels
        for out_ch, ks, st, ds in zip(channels, kernel_sizes, strides, downsamples):
            self.blocks.append(
                ConvBlock(in_ch, out_ch, ks, st, batch_norm, dropout, act, ds)
            )
            in_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d(pooling_output_size)
        self.output_size = channels[-1] * pooling_output_size ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)

        Returns
        -------
        torch.Tensor, shape (B, output_size)
        """
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# ResNet-style blocks
# ---------------------------------------------------------------------------

class BottleneckBlock(nn.Module):
    """Pre-activation bottleneck block used in ResNet-50/101/152.

    Architecture: BN -> act -> Conv(1x1) -> BN -> act -> Conv(3×3) -> BN -> act -> Conv(1x1) + skip.

    Parameters
    ----------
    in_channels : int
    planes : int
        Inner (bottleneck) channel width. Output channels = ``planes * 4``.
    stride : int
        Applied to the 3×3 convolution. Use 2 to halve spatial dims. Default: 1.
    act : nn.Module
        Activation function instance. Default: nn.ReLU().
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        planes: int,
        stride: int = 1,
        act: nn.Module = None,
    ):
        super().__init__()
        self.act = act if act is not None else nn.ReLU()
        out_channels = planes * self.expansion

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, planes, kernel_size=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_channels, kernel_size=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.conv1(self.act(self.bn1(x)))
        x = self.conv2(self.act(self.bn2(x)))
        x = self.conv3(self.act(self.bn3(x)))
        return x + residual


class ResidualBlock(nn.Module):
    """Pre-activation residual block (He et al., 2016).

    Architecture: BN -> act -> Conv2d -> BN -> act -> Conv2d + skip.
    A 1x1 projection conv is added automatically when ``in_channels !=
    out_channels`` or ``stride > 1``.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    stride : int
        Stride for the first convolution. Use 2 to halve spatial dims. Default: 1.
    act : nn.Module
        Activation function instance. Default: nn.ReLU().
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        act: nn.Module = None,
    ):
        super().__init__()
        self.act = act if act is not None else nn.ReLU()

        self.bn1   = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels,
                                      kernel_size=1, stride=stride, bias=False)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.shortcut is None else self.shortcut(x)
        x = self.act(self.bn1(x))
        x = self.conv1(x)
        x = self.act(self.bn2(x))
        x = self.conv2(x)
        return x + residual


class ResNet(nn.Module):
    """Flexible ResNet feature extractor.

    Supports both basic (:class:`ResidualBlock`) and bottleneck
    (:class:`BottleneckBlock`) block types. Use :func:`build_resnet` to
    instantiate standard variants (ResNet-18/34/50/101/152).

    Parameters
    ----------
    in_channels : int
        Number of input image channels.
    channels : list of int
        Output channels for each stage. For ``block_type='bottleneck'``, these
        are the *final* output channels (i.e. ``planes * 4``).
    blocks_per_stage : int or list of int
        Residual blocks per stage. Default: 2.
    stem_channels : int
        Output channels of the initial 7×7 stem conv. Default: 64.
    stem_stride : int
        Stride of the stem conv. Default: 1.
    downsample_stages : bool or list of bool
        Apply stride=2 at the first block of each stage. Defaults to False for
        the first stage and True for all subsequent stages.
    block_type : str
        ``'basic'`` (:class:`ResidualBlock`, used in ResNet-18/34) or
        ``'bottleneck'`` (:class:`BottleneckBlock`, used in ResNet-50+).
        Default: ``'basic'``.
    act : nn.Module
        Activation function instance. Default: nn.ReLU().
    pooling_output_size : int
        Spatial size for the final AdaptiveAvgPool2d. Default: 1.
    """

    def __init__(
        self,
        in_channels: int,
        channels: List[int],
        blocks_per_stage: Union[int, List[int]] = 2,
        stem_channels: int = 64,
        stem_stride: int = 1,
        downsample_stages: Union[bool, List[bool]] = None,
        block_type: str = 'basic',
        act: nn.Module = None,
        pooling_output_size: int = 1,
    ):
        super().__init__()

        if block_type not in ('basic', 'bottleneck'):
            raise ValueError(f"block_type must be 'basic' or 'bottleneck', got '{block_type}'")

        self.act = act if act is not None else nn.ReLU()
        self.block_type = block_type

        n_stages = len(channels)
        if isinstance(blocks_per_stage, int):
            blocks_per_stage = [blocks_per_stage] * n_stages
        if downsample_stages is None:
            downsample_stages = [False] + [True] * (n_stages - 1)
        elif isinstance(downsample_stages, bool):
            downsample_stages = [downsample_stages] * n_stages

        # Stem
        self.stem_conv = nn.Conv2d(in_channels, stem_channels, kernel_size=7,
                                   stride=stem_stride, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(stem_channels)

        # Flat list of all blocks across all stages
        self.blocks = nn.ModuleList()
        prev_ch = stem_channels
        for out_ch, n_blocks, do_ds in zip(channels, blocks_per_stage, downsample_stages):
            for b in range(n_blocks):
                stride = 2 if (b == 0 and do_ds) else 1
                if block_type == 'basic':
                    self.blocks.append(ResidualBlock(prev_ch, out_ch, stride, self.act))
                else:
                    planes = out_ch // BottleneckBlock.expansion
                    self.blocks.append(BottleneckBlock(prev_ch, planes, stride, self.act))
                prev_ch = out_ch

        self.global_pool = nn.AdaptiveAvgPool2d(pooling_output_size)
        self.output_size = channels[-1] * pooling_output_size ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor, shape (B, C, H, W)

        Returns
        -------
        torch.Tensor, shape (B, output_size)
        """
        x = self.act(self.stem_bn(self.stem_conv(x)))
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        return x.flatten(1)


# ---------------------------------------------------------------------------
# Standard ResNet variant configs and factory
# ---------------------------------------------------------------------------

_RESNET_CONFIGS: Dict[str, dict] = {
    'resnet18':  dict(channels=[64, 128, 256, 512], blocks_per_stage=[2, 2, 2, 2], block_type='basic'),
    'resnet34':  dict(channels=[64, 128, 256, 512], blocks_per_stage=[3, 4, 6, 3], block_type='basic'),
    'resnet50':  dict(channels=[256, 512, 1024, 2048], blocks_per_stage=[3, 4, 6, 3], block_type='bottleneck'),
    'resnet101': dict(channels=[256, 512, 1024, 2048], blocks_per_stage=[3, 4, 23, 3], block_type='bottleneck'),
    'resnet152': dict(channels=[256, 512, 1024, 2048], blocks_per_stage=[3, 8, 36, 3], block_type='bottleneck'),
}


def build_resnet(
    variant: str,
    in_channels: int = 3,
    stem_channels: int = 64,
    stem_stride: int = 2,
    downsample_stages: Union[bool, List[bool]] = None,
    act: 'nn.Module' = None,
    pooling_output_size: int = 1,
    **kwargs,
) -> ResNet:
    """Build a standard ResNet variant.

    Parameters
    ----------
    variant : str
        One of ``'resnet18'``, ``'resnet34'``, ``'resnet50'``, ``'resnet101'``,
        ``'resnet152'``.
    in_channels : int
        Number of input image channels. Default 3.
    stem_channels : int
        Output channels of the 7×7 stem conv. Default 64.
    stem_stride : int
        Stride of the stem conv (use 2 for ImageNet-style, 1 for small images).
        Default 2.
    downsample_stages : bool or list of bool, optional
        Override default stage downsampling. Defaults to ``[False, True, True, True]``.
    act : nn.Module, optional
        Activation function instance. Default ``nn.ReLU()``.
    pooling_output_size : int
        Spatial size for final AdaptiveAvgPool2d. Default 1.
    **kwargs
        Additional keyword arguments override the variant config (e.g.
        ``channels``, ``blocks_per_stage``).

    Returns
    -------
    ResNet
    """
    if variant not in _RESNET_CONFIGS:
        raise ValueError(
            f"Unknown ResNet variant '{variant}'. "
            f"Available: {list(_RESNET_CONFIGS)}"
        )
    cfg = {**_RESNET_CONFIGS[variant], **kwargs}
    return ResNet(
        in_channels=in_channels,
        stem_channels=stem_channels,
        stem_stride=stem_stride,
        downsample_stages=downsample_stages,
        act=act,
        pooling_output_size=pooling_output_size,
        **cfg,
    )


# ---------------------------------------------------------------------------
# Pretrained backbones via timm
# ---------------------------------------------------------------------------

def build_pretrained(
    model_name: str,
    in_channels: int = 3,
    pretrained: bool = True,
    pooling_output_size: int = 1,
    **timm_kwargs,
) -> nn.Module:
    """Load a pretrained backbone from timm as a feature extractor.

    The classifier head is removed; the model returns a flat feature vector.
    Non-3-channel inputs are handled automatically by timm (it adapts the stem
    conv and initialises extra channels from the pretrained mean).

    Parameters
    ----------
    model_name : str
        Any model name supported by timm, e.g. ``'resnet50'``,
        ``'efficientnet_b0'``, ``'vit_base_patch16_224'``.
        Run ``timm.list_models(pretrained=True)`` to see all options.
    in_channels : int
        Number of input image channels. Default 3.
    pretrained : bool
        Load ImageNet pretrained weights. Default True.
    pooling_output_size : int
        For CNN backbones: spatial size of the global pooling output.
        ``1`` uses global average pooling (recommended).
        ``>1`` wraps the backbone with an :class:`nn.AdaptiveAvgPool2d`
        and disables timm's built-in pooling.
        Has no effect on transformer backbones (ViT etc.). Default 1.
    **timm_kwargs
        Extra keyword arguments forwarded verbatim to
        ``timm.create_model`` (e.g. ``drop_rate``, ``drop_path_rate``).

    Returns
    -------
    nn.Module
        Backbone with the classification head removed. Exposes an
        ``output_size`` attribute giving the flat feature dimension.

    Examples
    --------
    >>> backbone = build_pretrained('resnet50', in_channels=1)
    >>> backbone = build_pretrained('efficientnet_b4', in_channels=4, pretrained=False)
    >>> backbone = build_pretrained('vit_base_patch16_224', in_channels=3)
    """
    try:
        import timm
    except ImportError as exc:
        raise ImportError(
            "timm is required for pretrained backbones: pip install timm"
        ) from exc

    if pooling_output_size == 1:
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,      # removes classifier head
            global_pool='avg',
            **timm_kwargs,
        )
        model.output_size = model.num_features
    else:
        # Disable timm's pooling; add a custom AdaptiveAvgPool2d wrapper
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool='',
            **timm_kwargs,
        )
        num_features = model.num_features
        pool = nn.AdaptiveAvgPool2d(pooling_output_size)

        class _WrappedBackbone(nn.Module):
            def __init__(self, backbone, adaptive_pool, output_size):
                super().__init__()
                self.backbone = backbone
                self.pool = adaptive_pool
                self.output_size = output_size

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = self.backbone(x)
                x = self.pool(x)
                return x.flatten(1)

        model = _WrappedBackbone(
            model, pool,
            output_size=num_features * pooling_output_size ** 2,
        )

    return model
