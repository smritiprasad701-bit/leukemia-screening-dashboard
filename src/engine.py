"""
Shared training engine used by train_baseline.py and train_advanced.py:
loss functions (weighted CE / focal), the train/validate loop, early stopping,
checkpointing, and curve logging.
"""
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from preprocessing import compute_class_weights


class FocalLoss(nn.Module):
    """Focal loss - down-weights easy examples, focuses training on hard/minority
    examples. Alternative to weighted cross-entropy for class imbalance."""
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(cfg: dict, train_labels, device):
    w0, w1 = compute_class_weights(train_labels)
    weight_tensor = torch.tensor([w0, w1], dtype=torch.float32, device=device)
    loss_type = cfg["training"]["loss"]
    if loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif loss_type == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weight_tensor)
    elif loss_type == "focal":
        return FocalLoss(gamma=cfg["training"]["focal_gamma"], weight=weight_tensor)
    else:
        raise ValueError(f"Unknown loss {loss_type}")


def run_epoch(model, loader, criterion, optimizer, device, scaler, train: bool):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for images, labels, _subject_ids in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)

            if train:
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=(scaler is not None)):
                    logits = model(images)
                    loss = criterion(logits, labels)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, cfg, device, run_name: str):
    train_labels = train_loader.dataset.df["label"].values
    criterion = build_loss(cfg, train_labels, device)

    lr = cfg["training"]["lr"]
    wd = cfg["training"]["weight_decay"]
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=wd)

    epochs = cfg["training"]["epochs"]
    if cfg["training"]["scheduler"] == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    elif cfg["training"]["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(1, epochs // 3), gamma=0.3)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler() if (cfg["training"]["amp"] and device.type == "cuda") else None

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{run_name}_best.pt"

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience = cfg["training"]["early_stopping_patience"]
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)

        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"[{run_name}] Epoch {epoch}/{epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({"model_state_dict": model.state_dict(), "cfg": cfg, "epoch": epoch}, best_path)
            print(f"  -> new best model saved to {best_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  -> early stopping triggered at epoch {epoch}")
                break

    history_path = ckpt_dir / f"{run_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    return history, str(best_path)
