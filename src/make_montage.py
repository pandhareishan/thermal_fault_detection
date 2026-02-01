#!/usr/bin/env python3
"""Create a montage of sample images from the training split."""

from __future__ import annotations

import argparse
import os
import random
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


CLASSES = ["normal", "overheating", "fault"]


def _gather_images(root_dir: str, cls: str) -> List[str]:
    cls_dir = os.path.join(root_dir, "train", cls)
    if not os.path.isdir(cls_dir):
        return []
    return [
        os.path.join(cls_dir, f)
        for f in os.listdir(cls_dir)
        if f.lower().endswith(".png")
    ]


def _select_samples(paths: List[str], n: int, rng: random.Random) -> List[str]:
    if len(paths) < n:
        raise ValueError(f"Not enough images for montage: need {n}, found {len(paths)}")
    return rng.sample(paths, n)


def _load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def create_montage(
    in_dir: str,
    out_path: str,
    seed: int,
    tile_size: Tuple[int, int] | None = None,
) -> None:
    rng = random.Random(seed)

    samples = []
    labels = []
    for cls in CLASSES:
        paths = _gather_images(in_dir, cls)
        selected = _select_samples(paths, 4, rng)
        samples.extend(selected)
        labels.extend([cls] * len(selected))

    images = [Image.open(p).convert("L") for p in samples]
    if tile_size is None:
        tile_size = images[0].size

    cols = 4
    rows = 3
    montage = Image.new("L", (tile_size[0] * cols, tile_size[1] * rows))
    font = _load_font(size=max(12, tile_size[0] // 18))

    for idx, (img, label) in enumerate(zip(images, labels)):
        r = idx // cols
        c = idx % cols
        img_resized = img.resize(tile_size)
        montage.paste(img_resized, (c * tile_size[0], r * tile_size[1]))

        draw = ImageDraw.Draw(montage)
        text = label
        text_pos = (c * tile_size[0] + 6, r * tile_size[1] + 6)
        draw.rectangle(
            [
                text_pos,
                (text_pos[0] + len(text) * font.size, text_pos[1] + font.size + 4),
            ],
            fill=0,
        )
        draw.text(text_pos, text, fill=255, font=font)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    montage.save(out_path, format="PNG")
    print(f"Saved montage to {out_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create sample montage from training data")
    parser.add_argument("--in_dir", type=str, default="data", help="Dataset root directory")
    parser.add_argument(
        "--out_path",
        type=str,
        default=os.path.join("figures", "samples.png"),
        help="Output montage path",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    create_montage(args.in_dir, args.out_path, args.seed)


if __name__ == "__main__":
    main()
