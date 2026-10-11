"""
PHASE 6 - Image-level evaluation.

Computes accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC,
confusion matrix, and per-class metrics on a given split (default: test). Also plots
ROC curve, PR curve, confusion matrix, and (if a *_history.json exists) training/val
loss & accuracy curves.

WHY NOT ACCURACY ALONE: in a screening task the classes are imbalanced and the cost
of a false negative (missed leukemic cell) is not the same as a false positive. A
model that always predicts "all" (the majority class) can score high accuracy while
having poor specificity -- useless in practice. Sensitivity and specificity, reported
together, expose this failure mode; accuracy alone hides it.

Run:
    python src/evaluate.py --config configs/config.yaml --checkpoint models/baseline_best.pt --split test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)

from utils import load_config, set_seed, get_device, ensure_dirs
from dataset import build_dataloaders
from models import build_model


def get_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs, all_subjects = [], [], [], []
    with torch.no_grad():
        for images, labels, subject_ids in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)[:, 1]  # probability of class "all" (leukemic)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_subjects.extend(subject_ids)
    return np.array(all_labels), np.array(all_preds), np.array(all_probs), list(all_subjects)


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # recall on positive (all) class
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    pr_auc = average_precision_score(y_true, y_prob)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def plot_confusion_matrix(y_true, y_pred, class_names, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(y_true, y_prob, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr(y_true, y_prob, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"PR curve (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_curves(history_path, out_dir):
    history_path = Path(history_path)
    if not history_path.exists():
        print(f"No history file at {history_path}, skipping training curves.")
        return
    with open(history_path) as f:
        h = json.load(f)

    plt.figure(figsize=(6, 4))
    plt.plot(h["train_loss"], label="train loss")
    plt.plot(h["val_loss"], label="val loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss curves")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "loss_curves.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.plot(h["train_acc"], label="train acc")
    plt.plot(h["val_acc"], label="val acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy curves")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "accuracy_curves.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device)
    run_name = Path(args.checkpoint).stem.replace("_best", "")
    model_type = "baseline" if "baseline" in run_name else "advanced"
    model = build_model(model_type, cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader = val_loader if args.split == "val" else test_loader

    y_true, y_pred, y_prob, subjects = get_predictions(model, loader, device)
    metrics = compute_metrics(y_true, y_pred, y_prob)

    print(json.dumps(metrics, indent=2))
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=cfg["data"]["class_names"]))

    out_dir = Path(cfg["evaluation"]["outputs_dir"]) / "plots"
    ensure_dirs(out_dir)
    plot_confusion_matrix(y_true, y_pred, cfg["data"]["class_names"], out_dir / f"{run_name}_{args.split}_cm.png")
    plot_roc(y_true, y_prob, out_dir / f"{run_name}_{args.split}_roc.png")
    plot_pr(y_true, y_prob, out_dir / f"{run_name}_{args.split}_pr.png")
    plot_training_curves(Path(cfg["training"]["checkpoint_dir"]) / f"{run_name}_history.json", out_dir)

    metrics_path = out_dir / f"{run_name}_{args.split}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics and plots saved under {out_dir}")


if __name__ == "__main__":
    main()
