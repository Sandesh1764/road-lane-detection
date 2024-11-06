"""Model zoo for binary road-lane segmentation (segmentation_models_pytorch)."""

from __future__ import annotations

from typing import Dict

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_model(
    architecture: str = "deeplabv3",
    encoder: str = "resnet18",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
) -> nn.Module:
    arch = architecture.lower().replace("-", "").replace("_", "")
    if arch in {"deeplabv3", "deeplab"}:
        return smp.DeepLabV3(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch in {"unetplusplus", "unetpp", "unet++"}:
        return smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    if arch == "unet":
        return smp.Unet(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=classes,
        )
    raise ValueError(f"Unknown architecture: {architecture}")


def get_models(pretrained: bool = True) -> Dict[str, nn.Module]:
    """Default comparison set used in the late-2024 experiments."""
    weights = "imagenet" if pretrained else None
    return {
        "DeepLabV3_ResNet18": build_model("deeplabv3", "resnet18", weights),
        "DeepLabV3_ResNet50": build_model("deeplabv3", "resnet50", weights),
        "UNetPlusPlus_ResNet18": build_model("unetplusplus", "resnet18", weights),
    }


def forward_logits(model: nn.Module, images):
    """Normalize SMP / torchvision-style outputs to a tensor of logits."""
    out = model(images)
    if isinstance(out, dict):
        return out.get("out", next(iter(out.values())))
    return out
