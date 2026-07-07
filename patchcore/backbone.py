"""
ResNet-18 backbone with multi-layer feature extraction for PatchCore.

Extracts features from intermediate layers [layer1, layer2, layer3],
pools each to a fixed spatial grid, and concatenates channels.

Supports both PyTorch (training) and ONNX export.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import List, Tuple

from .config import INPUT_SIZE, POOL_SIZE, LAYERS, FEATURE_DIM, NUM_PATCHES


class PatchCoreFeatureExtractor(nn.Module):
    """
    ResNet-18 wrapper that outputs multi-layer feature maps pooled and concatenated.

    Output shape: (batch, NUM_PATCHES, FEATURE_DIM) = (batch, 256, 448)
    """

    def __init__(self, pretrained: bool = True):
        super().__init__()
        # Load pretrained ResNet-18
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet18(weights=weights)

        # Extract sequential parts up to layer3
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1   # outputs 64 channels, 1/4 scale
        self.layer2 = resnet.layer2   # outputs 128 channels, 1/8 scale
        self.layer3 = resnet.layer3   # outputs 256 channels, 1/16 scale

        # Adaptive pooling to fixed spatial grid
        self.adaptive_pool = nn.AdaptiveAvgPool2d((POOL_SIZE, POOL_SIZE))

        self._feature_dim = FEATURE_DIM
        self._num_patches = NUM_PATCHES

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 3, 256, 256)

        Returns:
            features: (batch, 256, 448) — 256 patches, each 448-dim
        """
        # Stem
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Layer 1: 1/4 scale
        f1 = self.layer1(x)
        f1_pooled = self.adaptive_pool(f1)  # (B, 64, 16, 16)

        # Layer 2: 1/8 scale
        f2 = self.layer2(f1)
        f2_pooled = self.adaptive_pool(f2)  # (B, 128, 16, 16)

        # Layer 3: 1/16 scale
        f3 = self.layer3(f2)
        f3_pooled = self.adaptive_pool(f3)  # (B, 256, 16, 16)

        # Concatenate channels: (B, 64+128+256, 16, 16) = (B, 448, 16, 16)
        features = torch.cat([f1_pooled, f2_pooled, f3_pooled], dim=1)

        # Reshape to (B, 256, 448)
        B = features.shape[0]
        features = features.reshape(B, self._feature_dim, self._num_patches)
        features = features.transpose(1, 2)  # (B, 256, 448)

        return features

    @property
    def feature_dim(self) -> int:
        return self._feature_dim

    @property
    def num_patches(self) -> int:
        return self._num_patches


def create_feature_extractor(pretrained: bool = True) -> PatchCoreFeatureExtractor:
    """Factory function to create the feature extractor."""
    return PatchCoreFeatureExtractor(pretrained=pretrained)
