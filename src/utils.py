"""Utility helpers for training and evaluation."""

from __future__ import annotations

import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False


def plot_training_curves(history: Dict[str, List[float]], out_path: str) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	epochs = range(1, len(history["train_loss"]) + 1)

	fig, ax = plt.subplots(1, 2, figsize=(10, 4))
	ax[0].plot(epochs, history["train_loss"], label="train")
	ax[0].plot(epochs, history["val_loss"], label="val")
	ax[0].set_title("Loss")
	ax[0].set_xlabel("Epoch")
	ax[0].set_ylabel("Loss")
	ax[0].legend()

	ax[1].plot(epochs, history["train_acc"], label="train")
	ax[1].plot(epochs, history["val_acc"], label="val")
	ax[1].set_title("Accuracy")
	ax[1].set_xlabel("Epoch")
	ax[1].set_ylabel("Accuracy")
	ax[1].legend()

	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)


def confusion_matrix(y_true: List[int], y_pred: List[int], num_classes: int) -> np.ndarray:
	cm = np.zeros((num_classes, num_classes), dtype=np.int64)
	for t, p in zip(y_true, y_pred):
		cm[t, p] += 1
	return cm


def precision_recall_f1(
	cm: np.ndarray,
	class_names: List[str],
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
	metrics = []
	for idx, name in enumerate(class_names):
		tp = cm[idx, idx]
		fp = cm[:, idx].sum() - tp
		fn = cm[idx, :].sum() - tp
		support = cm[idx, :].sum()

		precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
		recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
		f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

		metrics.append(
			{
				"class": name,
				"precision": float(precision),
				"recall": float(recall),
				"f1": float(f1),
				"support": int(support),
			}
		)

	overall = {
		"accuracy": float(np.trace(cm) / np.sum(cm)) if np.sum(cm) > 0 else 0.0,
	}
	return metrics, overall


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], out_path: str) -> None:
	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	fig, ax = plt.subplots(figsize=(5, 4))
	im = ax.imshow(cm, cmap="Blues")
	ax.set_xticks(range(len(class_names)))
	ax.set_yticks(range(len(class_names)))
	ax.set_xticklabels(class_names, rotation=45, ha="right")
	ax.set_yticklabels(class_names)
	ax.set_xlabel("Predicted")
	ax.set_ylabel("True")
	ax.set_title("Confusion Matrix")

	for i in range(cm.shape[0]):
		for j in range(cm.shape[1]):
			ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

	fig.colorbar(im, ax=ax)
	fig.tight_layout()
	fig.savefig(out_path, dpi=150)
	plt.close(fig)
