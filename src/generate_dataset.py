"""Synthetic thermal dataset generator.

Generates grayscale thermal images with background gradients, sensor-like noise,
and component hotspots modeled as 2D Gaussians.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageFilter


CLASSES = ["normal", "overheating", "fault"]


@dataclass
class Hotspot:
	x: float
	y: float
	amp: float
	sigma_x: float
	sigma_y: float
	theta: float
	kind: str


def _make_dirs(out_dir: str) -> Dict[str, str]:
	split_dirs = {}
	for split in ["train", "val", "test"]:
		for cls in CLASSES:
			path = os.path.join(out_dir, split, cls)
			os.makedirs(path, exist_ok=True)
			split_dirs[f"{split}/{cls}"] = path
	return split_dirs


def _background_gradient(rng: np.random.Generator, size: int) -> Tuple[np.ndarray, Dict[str, float]]:
	x = np.linspace(0, 1, size, dtype=np.float32)
	y = np.linspace(0, 1, size, dtype=np.float32)
	xv, yv = np.meshgrid(x, y)

	base = rng.uniform(30.0, 80.0)
	gx = rng.uniform(-20.0, 20.0)
	gy = rng.uniform(-20.0, 20.0)
	grad = base + gx * xv + gy * yv
	params = {"base": base, "gx": gx, "gy": gy}
	return grad, params


def _gaussian_2d(
	xv: np.ndarray,
	yv: np.ndarray,
	x0: float,
	y0: float,
	amp: float,
	sigma_x: float,
	sigma_y: float,
	theta: float,
) -> np.ndarray:
	cos_t = np.cos(theta)
	sin_t = np.sin(theta)
	x = xv - x0
	y = yv - y0
	xr = cos_t * x + sin_t * y
	yr = -sin_t * x + cos_t * y
	exp_term = (xr ** 2) / (2 * sigma_x ** 2) + (yr ** 2) / (2 * sigma_y ** 2)
	return amp * np.exp(-exp_term)


def _sample_hotspots(
	rng: np.random.Generator,
	size: int,
	cls: str,
) -> Tuple[List[Hotspot], Dict[str, str]]:
	n_hotspots = rng.integers(1, 4)
	hotspots: List[Hotspot] = []
	behavior: Dict[str, str] = {}

	if cls == "normal":
		amp_range = (40.0, 90.0)
		sigma_range = (10.0, 25.0)
	elif cls == "overheating":
		amp_range = (80.0, 140.0)
		sigma_range = (18.0, 40.0)
	else:
		amp_range = (120.0, 200.0)
		sigma_range = (4.0, 14.0)

	for idx in range(n_hotspots):
		x0 = rng.uniform(0.2, 0.8) * size
		y0 = rng.uniform(0.2, 0.8) * size
		amp = rng.uniform(*amp_range)
		sigma_x = rng.uniform(*sigma_range)
		sigma_y = rng.uniform(*sigma_range)
		theta = rng.uniform(0, np.pi)
		hotspots.append(
			Hotspot(
				x=x0,
				y=y0,
				amp=amp,
				sigma_x=sigma_x,
				sigma_y=sigma_y,
				theta=theta,
				kind="primary",
			)
		)

	if cls == "fault":
		fault_type = rng.choice(["tight_intense", "secondary_hotspot", "asymmetric"])
		behavior["fault_type"] = str(fault_type)

		if fault_type == "tight_intense":
			x0 = rng.uniform(0.3, 0.7) * size
			y0 = rng.uniform(0.3, 0.7) * size
			hotspots.append(
				Hotspot(
					x=x0,
					y=y0,
					amp=rng.uniform(180.0, 240.0),
					sigma_x=rng.uniform(3.0, 8.0),
					sigma_y=rng.uniform(3.0, 8.0),
					theta=rng.uniform(0, np.pi),
					kind="tight_intense",
				)
			)
		elif fault_type == "secondary_hotspot":
			base = hotspots[0]
			offset = rng.uniform(10.0, 30.0)
			angle = rng.uniform(0, 2 * np.pi)
			hotspots.append(
				Hotspot(
					x=np.clip(base.x + offset * np.cos(angle), 0, size - 1),
					y=np.clip(base.y + offset * np.sin(angle), 0, size - 1),
					amp=rng.uniform(140.0, 200.0),
					sigma_x=rng.uniform(5.0, 10.0),
					sigma_y=rng.uniform(5.0, 10.0),
					theta=rng.uniform(0, np.pi),
					kind="secondary",
				)
			)
		else:
			base = hotspots[0]
			hotspots.append(
				Hotspot(
					x=base.x,
					y=base.y,
					amp=rng.uniform(150.0, 210.0),
					sigma_x=rng.uniform(3.0, 6.0),
					sigma_y=rng.uniform(12.0, 20.0),
					theta=rng.uniform(0, np.pi),
					kind="asymmetric",
				)
			)

	return hotspots, behavior


def _render_image(
	rng: np.random.Generator,
	size: int,
	cls: str,
) -> Tuple[np.ndarray, Dict[str, object]]:
	background, bg_params = _background_gradient(rng, size)
	noise_sigma = rng.uniform(3.0, 9.0)
	noise = rng.normal(0.0, noise_sigma, (size, size)).astype(np.float32)

	xv, yv = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))
	hotspots, behavior = _sample_hotspots(rng, size, cls)
	hotspot_map = np.zeros((size, size), dtype=np.float32)
	for hs in hotspots:
		hotspot_map += _gaussian_2d(xv, yv, hs.x, hs.y, hs.amp, hs.sigma_x, hs.sigma_y, hs.theta)

	img = background + hotspot_map + noise

	blur_radius = rng.uniform(0.6, 1.4)
	img = np.clip(img, 0, 255)
	pil_img = Image.fromarray(img.astype(np.uint8), mode="L")
	pil_img = pil_img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
	img = np.array(pil_img, dtype=np.uint8)

	params = {
		"background": bg_params,
		"noise_sigma": float(noise_sigma),
		"blur_radius": float(blur_radius),
		"behavior": behavior,
		"hotspots": [
			{
				"x": float(hs.x),
				"y": float(hs.y),
				"amp": float(hs.amp),
				"sigma_x": float(hs.sigma_x),
				"sigma_y": float(hs.sigma_y),
				"theta": float(hs.theta),
				"kind": hs.kind,
			}
			for hs in hotspots
		],
	}
	return img, params


def _split_counts(n_total: int) -> Dict[str, int]:
	n_per_class = n_total // len(CLASSES)
	train = int(n_per_class * 0.8)
	val = int(n_per_class * 0.1)
	test = n_per_class - train - val
	return {"train": train, "val": val, "test": test}


def generate_dataset(out_dir: str, n_total: int, img_size: int, seed: int) -> None:
	os.makedirs(out_dir, exist_ok=True)
	split_dirs = _make_dirs(out_dir)

	rng = np.random.default_rng(seed)
	split_counts = _split_counts(n_total)
	n_per_class = n_total // len(CLASSES)

	labels_path = os.path.join(out_dir, "labels.csv")
	with open(labels_path, "w", newline="", encoding="utf-8") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerow(["filepath", "split", "class", "seed", "params_json"])

		for cls in CLASSES:
			for split, n_per_split in split_counts.items():
				for idx in range(n_per_split):
					image_seed = int(rng.integers(0, 2**32 - 1))
					img_rng = np.random.default_rng(image_seed)
					img, params = _render_image(img_rng, img_size, cls)

					filename = f"{cls}_{split}_{idx:04d}.png"
					out_path = os.path.join(split_dirs[f"{split}/{cls}"], filename)
					Image.fromarray(img, mode="L").save(out_path, format="PNG")

					rel_path = os.path.relpath(out_path, out_dir)
					writer.writerow(
						[
							rel_path.replace(os.sep, "/"),
							split,
							cls,
							image_seed,
							json.dumps(params),
						]
					)

	print(f"Generated {n_total} images into {out_dir}")
	print(f"Labels CSV written to {labels_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Generate synthetic thermal dataset")
	parser.add_argument("--out_dir", type=str, default="data", help="Output directory")
	parser.add_argument("--n_total", type=int, default=3000, help="Total number of images")
	parser.add_argument("--img_size", type=int, default=256, help="Image size (square)")
	parser.add_argument("--seed", type=int, default=42, help="Global random seed")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	if args.n_total % len(CLASSES) != 0:
		raise ValueError("n_total must be divisible by number of classes (3)")
	generate_dataset(args.out_dir, args.n_total, args.img_size, args.seed)


if __name__ == "__main__":
	main()
