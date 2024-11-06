"""Render DeepLabV3 lane-segmentation architecture (asphalt / amber theme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#E8EEF7"
MUTED = "#8B95A8"
BG = "#0E1116"
CARD = "#1A1F27"
ENC = "#243447"
ASPP = "#3D2E12"
DEC = "#1E3A3A"
AMBER = "#F5C542"
CYAN = "#4CC9F0"


def box(ax, x, y, w, h, text, fc, fs=8.5, bold=False):
    ax.add_patch(
        FancyBboxPatch(
            (x - w / 2, y - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.1,
            edgecolor=AMBER if bold else "#3A4250",
            facecolor=fc,
        )
    )
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fs,
        fontweight="bold" if bold else "regular",
        linespacing=1.25,
    )


def arrow(ax, x1, y1, x2, y2, color=AMBER):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            lw=1.3,
            color=color,
        )
    )


def main():
    out = Path(__file__).resolve().parents[1] / "assets" / "deeplabv3_architecture.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(14.2, 7.6), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 14.2)
    ax.set_ylim(0, 7.6)
    ax.axis("off")

    ax.text(7.1, 7.2, "DeepLabV3 · Road Lane Segmentation", ha="center", color=AMBER, fontsize=16, fontweight="bold")
    ax.text(
        7.1,
        6.8,
        "ResNet encoder  →  Atrous Spatial Pyramid Pooling  →  decoder  →  binary lane mask",
        ha="center",
        color=MUTED,
        fontsize=9,
    )

    # Pipeline blocks
    stages = [
        (1.2, 4.0, 1.6, 1.5, "Input\nRGB 512²", CARD, True),
        (3.3, 4.0, 1.8, 1.5, "ResNet\nEncoder\n(stride 8/16)", ENC, False),
        (5.6, 4.0, 2.2, 2.2, "ASPP\nrates 12·24·36\n+ image pool", ASPP, True),
        (8.1, 4.0, 1.8, 1.5, "1×1 fuse\n+ upsample", DEC, False),
        (10.2, 4.0, 1.7, 1.5, "Seg head\n1 channel", DEC, False),
        (12.3, 4.0, 1.7, 1.5, "Lane mask\nsigmoid → IoU", CARD, True),
    ]
    for x, y, w, h, t, fc, bold in stages:
        box(ax, x, y, w, h, t, fc, fs=8.2, bold=bold)

    for i in range(len(stages) - 1):
        x1 = stages[i][0] + stages[i][2] / 2
        x2 = stages[i + 1][0] - stages[i + 1][2] / 2
        arrow(ax, x1 + 0.05, 4.0, x2 - 0.05, 4.0)

    # ASPP detail row
    ax.text(5.6, 2.35, "ASPP parallel branches", ha="center", color=CYAN, fontsize=9, fontweight="bold")
    branches = ["1×1", "r=12", "r=24", "r=36", "GAP"]
    bx0 = 3.6
    for i, b in enumerate(branches):
        box(ax, bx0 + i * 1.05, 1.6, 0.9, 0.7, b, "#2A2214", fs=8)
        arrow(ax, bx0 + i * 1.05, 2.0, 5.6 - 0.4 + i * 0.15, 2.85, color=CYAN)

    # Side notes
    box(
        ax,
        1.8,
        1.5,
        2.6,
        1.3,
        "Loss: Dice (binary)\nOpt: Adam 1e-4\nBackbone: ImageNet",
        "#161B22",
        fs=8,
    )
    box(
        ax,
        12.0,
        1.5,
        2.6,
        1.3,
        "Also compared:\nU-Net++ ResNet18\nDeepLabV3 ResNet50",
        "#161B22",
        fs=8,
    )

    ax.text(7.1, 0.45, "Theme: asphalt night · lane amber · ASPP multi-scale context for thin markings", ha="center", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180, facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
