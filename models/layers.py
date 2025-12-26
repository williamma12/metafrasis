"""
Shared neural network layers for OCR models

Contains reusable building blocks:
- ConvBNLayer: Convolution + BatchNorm + Activation
- SEModule: Squeeze-and-Excitation attention
- BasicBlock: ResNet basic residual block
- ResidualUnit: MobileNetV3 inverted residual block
- CTCDecoder: CTC greedy decoding utility
- CHARSETS: Predefined character sets
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List


def make_divisible(v: int, divisor: int = 8) -> int:
    """Make channel count divisible by divisor (for efficient hardware execution)"""
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBNLayer(nn.Module):
    """
    Convolution + BatchNorm + Activation

    Standard building block for CNN backbones.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Convolution padding
        groups: Convolution groups (1=standard, in_channels=depthwise)
        act: Activation type ('relu', 'hardswish', 'none')
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        act: str = "relu"
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

        if act == "hardswish":
            self.act = nn.Hardswish(inplace=True)
        elif act == "relu":
            self.act = nn.ReLU(inplace=True)
        elif act == "none" or act is None:
            self.act = nn.Identity()
        else:
            self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation attention module

    Channel attention mechanism from SENet.

    Args:
        channels: Number of channels
        reduction: Reduction ratio for bottleneck
    """

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1, bias=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.avg_pool(x)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.hardsigmoid(self.fc2(x), inplace=True)
        return identity * x


class BasicBlock(nn.Module):
    """
    Basic ResNet residual block

    Used in ResNet-18/34 backbones.

    Args:
        in_channels: Input channels
        out_channels: Output channels
        stride: Convolution stride (1 or 2)
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1,
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)

        return out


class ResidualUnit(nn.Module):
    """
    MobileNetV3-style inverted residual block with optional SE

    Structure: Expand (1x1) -> Depthwise (kxk) -> SE -> Project (1x1)

    Args:
        in_channels: Input channels
        mid_channels: Expansion channels
        out_channels: Output channels
        kernel_size: Depthwise conv kernel size
        stride: Depthwise conv stride
        use_se: Whether to use SE attention
        act: Activation type ('relu', 'hardswish')
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        use_se: bool = False,
        act: str = "hardswish"
    ):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels

        # Expansion (1x1 conv)
        self.expand = ConvBNLayer(in_channels, mid_channels, 1, act=act)

        # Depthwise convolution
        self.depthwise = ConvBNLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=mid_channels,
            act=act
        )

        # SE module (optional)
        self.se = SEModule(mid_channels) if use_se else nn.Identity()

        # Projection (1x1 conv, no activation)
        self.project = ConvBNLayer(mid_channels, out_channels, 1, act="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.expand(x)
        x = self.depthwise(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_shortcut:
            x = x + identity
        return x


class CTCDecoder:
    """
    CTC greedy decoder with confidence estimation

    Utility class for decoding CTC outputs to text.
    Handles blank removal and repeated character merging.
    """

    def __init__(self, charset: str, blank_idx: int = None):
        """
        Args:
            charset: String of characters (without blank)
            blank_idx: Index of blank token (default: len(charset))
        """
        self.charset = charset
        self.blank_idx = blank_idx if blank_idx is not None else len(charset)
        self.idx_to_char = {idx: char for idx, char in enumerate(charset)}
        self.char_to_idx = {char: idx for idx, char in enumerate(charset)}

    def decode_batch(
        self,
        logits: torch.Tensor
    ) -> Tuple[List[str], List[float]]:
        """
        Decode batch of CTC logits

        Args:
            logits: [B, T, num_classes] raw logits

        Returns:
            Tuple of (texts, confidences)
        """
        probs = F.softmax(logits, dim=-1)
        max_probs, max_indices = probs.max(dim=-1)

        # Move to CPU for numpy conversion (handles MPS, CUDA, etc.)
        max_indices = max_indices.cpu()
        max_probs = max_probs.cpu()

        texts = []
        confidences = []

        for i in range(max_indices.size(0)):
            text, conf = self.decode_single(
                max_indices[i].numpy(),
                max_probs[i].numpy()
            )
            texts.append(text)
            confidences.append(conf)

        return texts, confidences

    def decode_single(
        self,
        indices: np.ndarray,
        probs: np.ndarray
    ) -> Tuple[str, float]:
        """
        Decode single sequence with CTC rules

        Args:
            indices: [T] predicted class indices
            probs: [T] predicted probabilities

        Returns:
            Tuple of (decoded_text, mean_confidence)
        """
        chars = []
        char_probs = []
        prev_idx = None

        for idx, prob in zip(indices, probs):
            # Skip blank token
            if idx == self.blank_idx:
                prev_idx = None
                continue

            # Skip repeated characters
            if idx == prev_idx:
                continue

            # Add character if in charset
            if idx in self.idx_to_char:
                chars.append(self.idx_to_char[idx])
                char_probs.append(prob)

            prev_idx = idx

        text = ''.join(chars)
        confidence = float(np.mean(char_probs)) if char_probs else 0.0

        return text, confidence


# Character sets for different languages
CHARSETS = {
    # Basic Latin + digits
    "latin": "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",

    # Latin lowercase + digits (CRNN default)
    "latin_lower": "0123456789abcdefghijklmnopqrstuvwxyz",

    # Ancient/Polytonic Greek
    "greek": (
        # Lowercase
        "αβγδεζηθικλμνξοπρσςτυφχψω"
        # Uppercase
        "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
        # Basic diacritics
        "άέήίόύώϊϋΐΰ"
        # Polytonic - smooth/rough breathing with accents
        "ἀἁἂἃἄἅἆἇἈἉἊἋἌἍἎἏ"
        "ἐἑἒἓἔἕἘἙἚἛἜἝ"
        "ἠἡἢἣἤἥἦἧἨἩἪἫἬἭἮἯ"
        "ἰἱἲἳἴἵἶἷἸἹἺἻἼἽἾἿ"
        "ὀὁὂὃὄὅὈὉὊὋὌὍ"
        "ὐὑὒὓὔὕὖὗὙὛὝὟ"
        "ὠὡὢὣὤὥὦὧὨὩὪὫὬὭὮὯ"
        # Graves
        "ὰάὲέὴήὶίὸόὺύὼώ"
        # Iota subscript combinations
        "ᾀᾁᾂᾃᾄᾅᾆᾇᾈᾉᾊᾋᾌᾍᾎᾏ"
        "ᾐᾑᾒᾓᾔᾕᾖᾗᾘᾙᾚᾛᾜᾝᾞᾟ"
        "ᾠᾡᾢᾣᾤᾥᾦᾧᾨᾩᾪᾫᾬᾭᾮᾯ"
        "ᾰᾱᾲᾳᾴᾶᾷᾸᾹᾺΆᾼ"
        "ῂῃῄῆῇῈΈῊΉῌ"
        "ῐῑῒΐῖῗῘῙῚΊ"
        "ῠῡῢΰῤῥῦῧῨῩῪΎῬ"
        "ῲῳῴῶῷῸΌῺΏῼ"
        # Punctuation and space
        " .,;:·!?'\"()-"
        # Digits
        "0123456789"
    ),
}


def get_charset(name: str) -> str:
    """Get predefined charset by name"""
    if name in CHARSETS:
        return CHARSETS[name]
    raise ValueError(f"Unknown charset: {name}. Available: {list(CHARSETS.keys())}")
