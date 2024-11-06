"""Infer lane masks on an image or folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from dataset import IMAGENET_MEAN, IMAGENET_STD, overlay_mask
from evaluate import load_checkpoint
from models import forward_logits
from utils import get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/infer"))
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--device", default=None)
    return p.parse_args()


def preprocess(bgr: np.ndarray, size: int) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return ((x - mean) / std).unsqueeze(0), rgb


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else get_device()
    model = load_checkpoint(args.checkpoint, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = (
        sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if args.input.is_dir()
        else [args.input]
    )
    for path in paths:
        bgr = cv2.imread(str(path))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]
        x, rgb_small = preprocess(bgr, args.image_size)
        with torch.no_grad():
            logits = forward_logits(model, x.to(device))
            pred = (torch.sigmoid(logits)[0, 0] > args.threshold).cpu().numpy().astype(np.uint8)
        pred_full = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        overlay = overlay_mask(rgb, pred_full)
        panel = np.hstack([rgb, overlay])
        out = args.output_dir / f"{path.stem}_lane.jpg"
        cv2.imwrite(str(out), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
