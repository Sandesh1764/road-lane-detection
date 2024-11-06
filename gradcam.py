"""Grad-CAM for binary lane segmentation (DeepLabV3 / U-Net++)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import IMAGENET_MEAN, IMAGENET_STD, overlay_mask
from evaluate import load_checkpoint
from models import forward_logits
from utils import get_device


class SegGradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model.eval()
        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._h = [
            target_layer.register_forward_hook(self._fwd),
            target_layer.register_full_backward_hook(self._bwd),
        ]

    def _fwd(self, _m, _i, o):
        self.activations = o.detach()

    def _bwd(self, _m, _gi, go):
        self.gradients = go[0].detach()

    def close(self):
        for h in self._h:
            h.remove()

    def __call__(self, x: torch.Tensor) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = forward_logits(self.model, x)
        score = torch.sigmoid(logits).mean()
        score.backward()
        acts = self.activations[0]
        grads = self.gradients[0]
        weights = grads.mean(dim=(1, 2))
        cam = F.relu((weights[:, None, None] * acts).sum(0))
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        cam = F.interpolate(cam[None, None], size=x.shape[-2:], mode="bilinear", align_corners=False)
        return cam[0, 0].detach().cpu().numpy()


def find_target_layer(model: torch.nn.Module) -> torch.nn.Module:
    # Prefer DeepLab decoder / ASPP-adjacent feature; fallback to last conv
    for name in ("decoder", "segmentation_head", "aspp"):
        if hasattr(model, name):
            mod = getattr(model, name)
            # deepest Conv2d inside
            last = None
            for m in mod.modules():
                if isinstance(m, torch.nn.Conv2d):
                    last = m
            if last is not None:
                return last
    last = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    if last is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM")
    return last


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("assets/gradcam"))
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--device", default=None)
    return p.parse_args()


def preprocess(path: Path, size: int):
    bgr = cv2.imread(str(path))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb_r = cv2.resize(rgb, (size, size))
    x = torch.from_numpy(rgb_r).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0), rgb_r


def heat_overlay(rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()
    model = load_checkpoint(args.checkpoint, device)
    target = find_target_layer(model)
    camper = SegGradCAM(model, target)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths: List[Path]
    if args.input.is_dir():
        paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"})[:6]
    else:
        paths = [args.input]

    for path in paths:
        x, rgb = preprocess(path, args.image_size)
        x = x.to(device)
        x.requires_grad_(True)
        with torch.enable_grad():
            cam = camper(x)
        with torch.no_grad():
            logits = forward_logits(model, x)
            pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy().astype(np.uint8)

        fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), dpi=150)
        fig.patch.set_facecolor("#0E1116")
        ax[0].imshow(rgb); ax[0].set_title("RGB", color="#E8EEF7")
        ax[1].imshow(overlay_mask(rgb, pred)); ax[1].set_title("Lane prediction", color="#E8EEF7")
        ax[2].imshow(heat_overlay(rgb, cam)); ax[2].set_title("Grad-CAM", color="#E8EEF7")
        for a in ax:
            a.axis("off")
        fig.suptitle(f"Where does the lane network look? — {path.name}", color="#F5C542", fontsize=11)
        fig.tight_layout()
        out = args.output_dir / f"gradcam_{path.stem}.jpg"
        fig.savefig(out, dpi=150, facecolor="#0E1116", bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out}")

    camper.close()


if __name__ == "__main__":
    main()
