"""Train a baseline transfer-learning model on the synthetic thermal dataset."""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from .utils import plot_training_curves, set_seed


def _build_dataloaders(
	data_dir: str,
	batch_size: int,
) -> Tuple[DataLoader, DataLoader, List[str]]:
	transform = transforms.Compose(
		[
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
			transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
		]
	)

	train_dir = os.path.join(data_dir, "train")
	val_dir = os.path.join(data_dir, "val")
	train_dataset = datasets.ImageFolder(train_dir, transform=transform)
	val_dataset = datasets.ImageFolder(val_dir, transform=transform)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=2,
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=2,
		pin_memory=True,
	)

	class_names = list(train_dataset.class_to_idx.keys())
	return train_loader, val_loader, class_names


def _get_model(num_classes: int) -> Tuple[nn.Module, str]:
	try:
		weights = models.ResNet18_Weights.DEFAULT
		model = models.resnet18(weights=weights)
		model.fc = nn.Linear(model.fc.in_features, num_classes)
		return model, "resnet18"
	except Exception:
		weights = models.MobileNet_V2_Weights.DEFAULT
		model = models.mobilenet_v2(weights=weights)
		model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
		return model, "mobilenet_v2"


def _set_trainable(model: nn.Module, arch: str, mode: str) -> None:
	for param in model.parameters():
		param.requires_grad = False

	if arch == "resnet18":
		if mode == "head":
			for param in model.fc.parameters():
				param.requires_grad = True
		else:
			for param in model.layer4.parameters():
				param.requires_grad = True
			for param in model.fc.parameters():
				param.requires_grad = True
	else:
		if mode == "head":
			for param in model.classifier.parameters():
				param.requires_grad = True
		else:
			for param in model.features[-1].parameters():
				param.requires_grad = True
			for param in model.classifier.parameters():
				param.requires_grad = True


def _epoch_pass(
	model: nn.Module,
	loader: DataLoader,
	criterion: nn.Module,
	optimizer: torch.optim.Optimizer | None,
	device: torch.device,
) -> Tuple[float, float]:
	is_train = optimizer is not None
	model.train(is_train)

	running_loss = 0.0
	correct = 0
	total = 0

	for inputs, targets in loader:
		inputs = inputs.to(device)
		targets = targets.to(device)

		if is_train:
			optimizer.zero_grad()

		outputs = model(inputs)
		loss = criterion(outputs, targets)

		if is_train:
			loss.backward()
			optimizer.step()

		running_loss += loss.item() * inputs.size(0)
		preds = outputs.argmax(dim=1)
		correct += (preds == targets).sum().item()
		total += targets.size(0)

	avg_loss = running_loss / max(total, 1)
	accuracy = correct / max(total, 1)
	return avg_loss, accuracy


def train(
	data_dir: str,
	epochs_head: int,
	epochs_ft: int,
	batch_size: int,
	lr: float,
	lr_ft: float,
	seed: int,
) -> None:
	set_seed(seed)
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	train_loader, val_loader, class_names = _build_dataloaders(data_dir, batch_size)
	model, arch = _get_model(num_classes=len(class_names))
	model = model.to(device)

	history: Dict[str, List[float]] = {
		"train_loss": [],
		"val_loss": [],
		"train_acc": [],
		"val_acc": [],
	}

	best_acc = 0.0
	model_dir = os.path.join("models")
	os.makedirs(model_dir, exist_ok=True)
	figures_dir = os.path.join("figures")
	os.makedirs(figures_dir, exist_ok=True)
	model_path = os.path.join(model_dir, "baseline_resnet18.pt")

	criterion = nn.CrossEntropyLoss()

	_set_trainable(model, arch, mode="head")
	optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

	total_epochs = epochs_head + epochs_ft
	for epoch in range(1, total_epochs + 1):
		if epoch == epochs_head + 1:
			_set_trainable(model, arch, mode="finetune")
			optimizer = torch.optim.Adam(
				filter(lambda p: p.requires_grad, model.parameters()), lr=lr_ft
			)

		train_loss, train_acc = _epoch_pass(model, train_loader, criterion, optimizer, device)
		val_loss, val_acc = _epoch_pass(model, val_loader, criterion, None, device)

		history["train_loss"].append(train_loss)
		history["val_loss"].append(val_loss)
		history["train_acc"].append(train_acc)
		history["val_acc"].append(val_acc)

		print(
			f"Epoch {epoch:02d}/{total_epochs} | "
			f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
			f"val loss {val_loss:.4f} acc {val_acc:.4f}"
		)

		if val_acc > best_acc:
			best_acc = val_acc
			torch.save(
				{
					"model_state_dict": model.state_dict(),
					"class_names": class_names,
					"arch": arch,
				},
				model_path,
			)

	plot_training_curves(history, os.path.join(figures_dir, "training_curves.png"))
	print(f"Best val accuracy: {best_acc:.4f}")
	print(f"Saved best model to {model_path}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train baseline model")
	parser.add_argument("--data_dir", type=str, default="data", help="Dataset root directory")
	parser.add_argument("--epochs_head", type=int, default=10, help="Head-only epochs")
	parser.add_argument("--epochs_ft", type=int, default=5, help="Fine-tuning epochs")
	parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
	parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
	parser.add_argument("--lr_ft", type=float, default=1e-4, help="Fine-tuning LR")
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	train(
		data_dir=args.data_dir,
		epochs_head=args.epochs_head,
		epochs_ft=args.epochs_ft,
		batch_size=args.batch_size,
		lr=args.lr,
		lr_ft=args.lr_ft,
		seed=args.seed,
	)


if __name__ == "__main__":
	main()
