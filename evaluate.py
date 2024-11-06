"""Evaluate a checkpoint and export prediction panels."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from dataset import LaneDataset, colorize_mask, overlay_mask
from models import build_model, forward_logits
from train import batch_iou
from utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("dataset/train"))
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-viz", type=int, default=8)
    p.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    p.add_argument("--device", default=None)
    return p.parse_args()


def load_checkpoint(path: Path, device: torch.device):
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(path, map_location=device)
    arch = ckpt.get("architecture", "deeplabv3") if isinstance(ckpt, dict) else "deeplabv3"
    enc = ckpt.get("encoder", "resnet18") if isinstance(ckpt, dict) else "resnet18"
    model = build_model(arch, enc, encoder_weights=None)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    # strip DDP prefix if present
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    args = parse_args()
    set_seed(42)
    device = torch.device(args.device) if args.device else get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ds = LaneDataset(args.data_root, image_size=args.image_size, binary=True)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    model = load_checkpoint(args.checkpoint, device)
    ious = []
    saved = 0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="eval"):
            images = images.to(device)
            masks = masks.to(device)
            logits = forward_logits(model, images)
            ious.append(batch_iou(logits, masks))
            if saved < args.num_viz:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                rgb = (images * std + mean).clamp(0, 1).cpu()
                for i in range(images.size(0)):
                    if saved >= args.num_viz:
                        break
                    r = (rgb[i].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    gt = masks[i, 0].cpu().numpy().astype(np.uint8)
                    pr = preds[i, 0].cpu().numpy().astype(np.uint8)
                    fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), dpi=140)
                    fig.patch.set_facecolor("#0E1116")
                    ax[0].imshow(r); ax[0].set_title("RGB", color="#E8EEF7")
                    ax[1].imshow(overlay_mask(r, gt)); ax[1].set_title("Ground truth", color="#E8EEF7")
                    ax[2].imshow(overlay_mask(r, pr)); ax[2].set_title("Prediction", color="#E8EEF7")
                    for a in ax:
                        a.axis("off")
                    fig.tight_layout()
                    out = args.output_dir / f"panel_{saved:02d}.jpg"
                    fig.savefig(out, dpi=140, facecolor="#0E1116", bbox_inches="tight")
                    plt.close(fig)
                    saved += 1

    print(f"Mean IoU: {float(np.mean(ious)):.4f}")
    print(f"Panels → {args.output_dir}")


if __name__ == "__main__":
    main()
