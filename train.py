"""Train binary road-lane segmentation models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from segmentation_models_pytorch import losses
from tqdm import tqdm

from dataset import build_loaders
from models import build_model, forward_logits
from utils import get_device, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train road-lane segmentation")
    p.add_argument("--data-root", type=Path, default=Path("dataset"))
    p.add_argument("--architecture", default="deeplabv3")
    p.add_argument("--encoder", default="resnet18")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    p.add_argument("--device", default=None)
    return p.parse_args()


@torch.no_grad()
def batch_iou(logits: torch.Tensor, masks: torch.Tensor, thr: float = 0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > thr).float()
    # masks expected (B,1,H,W)
    inter = (preds * masks).sum(dim=(1, 2, 3))
    union = ((preds + masks) > 0).float().sum(dim=(1, 2, 3)).clamp_min(1.0)
    return (inter / union).mean().item()


def plot_curves(history: dict, out_path: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=140)
    fig.patch.set_facecolor("#0E1116")
    epochs = range(1, len(history["train_loss"]) + 1)
    for ax, key, ylabel in [
        (axes[0], "loss", "Dice Loss"),
        (axes[1], "iou", "IoU"),
    ]:
        ax.set_facecolor("#0E1116")
        ax.plot(epochs, history[f"train_{key}"], color="#F5C542", label=f"train {key}")
        if history.get(f"val_{key}"):
            ax.plot(epochs, history[f"val_{key}"], color="#4CC9F0", label=f"val {key}")
        ax.set_xlabel("Epoch", color="#A8B3C5")
        ax.set_ylabel(ylabel, color="#A8B3C5")
        ax.tick_params(colors="#A8B3C5")
        ax.legend(facecolor="#1A1F27", edgecolor="#333", labelcolor="#E8EEF7")
        ax.grid(True, alpha=0.2)
        for spine in ax.spines.values():
            spine.set_color("#333")
    fig.suptitle(title, color="#F5C542", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, facecolor="#0E1116", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else get_device()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    train_loader, val_loader = build_loaders(
        args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print(f"train={len(train_loader.dataset)}  val={0 if val_loader is None else len(val_loader.dataset)}  device={device}")

    model = build_model(args.architecture, args.encoder).to(device)
    loss_fn = losses.DiceLoss(mode="binary")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "train_iou": [], "val_loss": [], "val_iou": []}
    best_val = float("inf")
    run_name = f"{args.architecture}_{args.encoder}"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_iou = 0.0
        n = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", unit="batch")
        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = forward_logits(model, images)
            loss = loss_fn(logits, masks)
            loss.backward()
            optimizer.step()
            bs = images.size(0)
            running_loss += loss.item() * bs
            running_iou += batch_iou(logits.detach(), masks) * bs
            n += bs
            pbar.set_postfix(loss=running_loss / n, iou=running_iou / n)

        train_loss = running_loss / max(n, 1)
        train_iou = running_iou / max(n, 1)
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)

        val_loss = val_iou = None
        if val_loader is not None:
            model.eval()
            v_loss = v_iou = 0.0
            vn = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    images = images.to(device, non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    logits = forward_logits(model, images)
                    loss = loss_fn(logits, masks)
                    bs = images.size(0)
                    v_loss += loss.item() * bs
                    v_iou += batch_iou(logits, masks) * bs
                    vn += bs
            val_loss = v_loss / max(vn, 1)
            val_iou = v_iou / max(vn, 1)
            history["val_loss"].append(val_loss)
            history["val_iou"].append(val_iou)

        msg = f"Epoch {epoch}: train loss={train_loss:.4f} iou={train_iou:.4f}"
        if val_loss is not None:
            msg += f" | val loss={val_loss:.4f} iou={val_iou:.4f}"
        print(msg)

        payload = {
            "epoch": epoch,
            "architecture": args.architecture,
            "encoder": args.encoder,
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "history": history,
        }
        torch.save(payload, ckpt_dir / "last.pth")
        score = val_loss if val_loss is not None else train_loss
        if score <= best_val:
            best_val = score
            torch.save(payload, ckpt_dir / "best.pth")
            print(f"  ↑ saved best → {ckpt_dir / 'best.pth'}")

    plot_curves(history, args.output_dir / "training_curves.jpg", run_name)
    with open(args.output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Done. Curves → {args.output_dir / 'training_curves.jpg'}")


if __name__ == "__main__":
    main()
