"""Lane segmentation dataset (custom bag-extracted campus / urban frames).

Labels follow a CULane-like scheme: 0 = background, >0 = lane instance ids.
Models in this repo train **binary** lane vs background (any id > 0 → 1).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Distinct colors for multi-lane GT visualization
LANE_COLORS = {
    1: (180, 255, 120),
    2: (255, 210, 40),
    3: (80, 200, 255),
    4: (255, 90, 60),
    5: (200, 120, 255),
}


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for k, c in LANE_COLORS.items():
        out[mask == k] = c
    # any other positive id → amber
    unknown = (mask > 0) & ~np.isin(mask, list(LANE_COLORS.keys()))
    out[unknown] = (255, 210, 40)
    return out


def overlay_mask(rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.55) -> np.ndarray:
    color = colorize_mask(mask).astype(np.float32)
    base = rgb.astype(np.float32)
    m = (mask > 0)[..., None]
    blended = np.where(m, base * (1.0 - alpha) + color * alpha, base)
    return blended.astype(np.uint8)


class LaneDataset(Dataset):
    """Pairs `imgs/**/*.jpg|png` with `label/**/<stem>.png` under a split root."""

    def __init__(
        self,
        root: Path | str,
        image_size: int = 512,
        binary: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.img_dir = self.root / "imgs"
        self.label_dir = self.root / "label"
        self.image_size = image_size
        self.binary = binary
        self.transform = transform

        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Missing images dir: {self.img_dir}")

        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        self.samples: list[Tuple[Path, Optional[Path]]] = []
        for img_path in sorted(self.img_dir.rglob("*")):
            if img_path.suffix.lower() not in exts or not img_path.is_file():
                continue
            # Prefer matching label by relative stem path
            rel = img_path.relative_to(self.img_dir)
            mask_path = self.label_dir / rel.with_suffix(".png")
            if not mask_path.is_file():
                # flat test folders may have no labels
                mask_path = None
            self.samples.append((img_path, mask_path))

        if not self.samples:
            raise ValueError(f"No images under {self.img_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
        image_np = np.asarray(image, dtype=np.uint8)

        if mask_path is not None and mask_path.is_file():
            mask = Image.open(mask_path)
            mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
            mask_np = np.asarray(mask)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            if self.binary:
                mask_np = (mask_np > 0).astype(np.float32)
            else:
                mask_np = mask_np.astype(np.int64)
        else:
            mask_np = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        if self.transform is not None:
            return self.transform(image_np, mask_np)

        # Default ImageNet normalize → CHW float
        x = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
        if self.binary:
            y = torch.from_numpy(mask_np).unsqueeze(0).float()
        else:
            y = torch.from_numpy(mask_np.astype(np.int64))
        return x, y


def build_loaders(
    data_root: Path | str,
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    binary: bool = True,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Build train (+ optional test) loaders from `data_root/{train,test}`."""
    root = Path(data_root)
    train_ds = LaneDataset(root / "train", image_size=image_size, binary=binary)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    test_loader = None
    if (root / "test").is_dir() and (root / "test" / "imgs").is_dir():
        test_ds = LaneDataset(root / "test", image_size=image_size, binary=binary)
        # only evaluate when labels exist
        labeled = sum(1 for _, m in test_ds.samples if m is not None)
        if labeled > 0:
            test_loader = DataLoader(
                test_ds,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )
    return train_loader, test_loader
