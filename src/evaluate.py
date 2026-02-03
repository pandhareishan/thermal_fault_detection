"""Evaluate the baseline model on the test split."""

from __future__ import annotations

import argparse
import csv
import os
from typing import List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from .utils import confusion_matrix, plot_confusion_matrix, precision_recall_f1, set_seed


def build_test_loader(data_dir: str, batch_size: int) -> tuple[DataLoader, List[str]]:
	transform = transforms.Compose(
		[
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)
	test_dir = os.path.join(data_dir, "test")
	test_dataset = datasets.ImageFolder(test_dir, transform=transform)
	loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
	class_names = list(test_dataset.class_to_idx.keys())
	return loader, class_names


def build_model(arch: str, num_classes: int) -> torch.nn.Module:
	if arch == "resnet18":
		model = models.resnet18(weights=None)
		model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
		return model
	model = models.mobilenet_v2(weights=None)
	model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
	return model


def evaluate(data_dir: str, model_path: str, batch_size: int) -> None:
	set_seed(42)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	loader, class_names = build_test_loader(data_dir, batch_size)

	checkpoint = torch.load(model_path, map_location=device)
	arch = checkpoint.get("arch", "resnet18")
	class_names = checkpoint.get("class_names", class_names)
	model = build_model(arch, num_classes=len(class_names))
	model.load_state_dict(checkpoint["model_state_dict"])
	model = model.to(device)
	model.eval()

	y_true: List[int] = []
	y_pred: List[int] = []
	correct = 0
	total = 0

	with torch.no_grad():
		for inputs, targets in loader:
			inputs = inputs.to(device)
			targets = targets.to(device)
			outputs = model(inputs)
			preds = outputs.argmax(dim=1)
			y_true.extend(targets.cpu().tolist())
			y_pred.extend(preds.cpu().tolist())
			correct += (preds == targets).sum().item()
			total += targets.size(0)

	accuracy = correct / max(total, 1)
	print(f"Test accuracy: {accuracy:.4f}")

	cm = confusion_matrix(y_true, y_pred, num_classes=len(class_names))
	metrics, overall = precision_recall_f1(cm, class_names)

	figures_dir = os.path.join("figures")
	os.makedirs(figures_dir, exist_ok=True)
	cm_path = os.path.join(figures_dir, "confusion_matrix.png")
	plot_confusion_matrix(cm, class_names, cm_path)

	csv_path = os.path.join(figures_dir, "test_metrics.csv")
	with open(csv_path, "w", newline="", encoding="utf-8") as f:
		writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
		writer.writeheader()
		for row in metrics:
			writer.writerow(row)

	print(f"Saved confusion matrix to {cm_path}")
	print(f"Saved metrics CSV to {csv_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate baseline model")
	parser.add_argument("--data_dir", type=str, default="data", help="Dataset root directory")
	parser.add_argument(
		"--model_path",
		type=str,
		default=os.path.join("models", "baseline_resnet18.pt"),
		help="Path to model checkpoint",
	)
	parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	evaluate(args.data_dir, args.model_path, args.batch_size)


if __name__ == "__main__":
	main()
