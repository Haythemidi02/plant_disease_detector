import os
import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model, count_parameters
from utils import seed_everything, get_device, MetricTracker


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Single epoch ──────────────────────────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer,
    device: torch.device,
    is_train: bool,
) -> tuple[float, float]:
    """
    Runs one full pass over the dataloader.
    Returns (avg_loss, accuracy) for the epoch.
    """
    model.train() if is_train else model.eval()

    total_loss    = 0.0
    correct       = 0
    total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss    += loss.item() * images.size(0)
            preds          = logits.argmax(dim=1)
            correct       += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct   / total_samples
    return avg_loss, accuracy


# ── Training loop ─────────────────────────────────────────────────────────────

def train(config: dict, checkpoint_path: str = None):
    """
    Full training loop driven by a config dict.

    Args:
        config          : hyperparameters (loaded from a YAML file)
        checkpoint_path : optional path to a .pt file to resume from
                          (used in Phase B to start from Phase A weights)
    """
    # ── Setup ─────────────────────────────────────────────────────────────────
    seed_everything(config.get("seed", 42))
    device = get_device()

    print(f"\nDevice      : {device}")
    print(f"Phase       : {config['phase']}")
    print(f"Freeze base : {config['freeze_base']}")
    print(f"LR          : {config['lr']}")
    print(f"Epochs      : {config['epochs']}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        root_dir    = config["data_dir"],
        val_split   = config.get("val_split",  0.15),
        test_split  = config.get("test_split", 0.10),
        batch_size  = config.get("batch_size", 32),
        num_workers = config.get("num_workers", 4),
        seed        = config.get("seed", 42),
    )
    num_classes = loaders["num_classes"]
    print(f"Classes     : {num_classes}")
    print(f"Train size  : {len(loaders['train'].dataset)}")
    print(f"Val size    : {len(loaders['val'].dataset)}\n")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(
        num_classes   = num_classes,
        freeze_base   = config["freeze_base"],
        unfreeze_from = config.get("unfreeze_from", None),
    ).to(device)

    # Load checkpoint if provided (Phase B starts from Phase A weights)
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state["model_state"])
        print(f"Loaded checkpoint : {checkpoint_path}\n")

    params = count_parameters(model)
    print(f"Trainable params : {params['trainable']:,}")
    print(f"Frozen params    : {params['frozen']:,}\n")

    # ── Loss, optimizer, scheduler ────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr           = config["lr"],
        weight_decay = config.get("weight_decay", 1e-4),
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",       # monitor val loss
        factor   = 0.5,         # halve LR on plateau
        patience = config.get("lr_patience", 3),
    )

    # ── Checkpoint dir ────────────────────────────────────────────────────────
    ckpt_dir  = Path(config.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / f"{config['phase']}_best.pt"

    # ── Metric tracker ────────────────────────────────────────────────────────
    tracker   = MetricTracker()
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = config.get("early_stop_patience", 7)

    # ── Epoch loop ────────────────────────────────────────────────────────────
    for epoch in range(1, config["epochs"] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")

        train_loss, train_acc = run_epoch(
            model, loaders["train"], criterion, optimizer, device, is_train=True
        )
        val_loss, val_acc = run_epoch(
            model, loaders["val"], criterion, optimizer, device, is_train=False
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        tracker.update(
            epoch      = epoch,
            train_loss = train_loss,
            train_acc  = train_acc,
            val_loss   = val_loss,
            val_acc    = val_acc,
            lr         = current_lr,
        )

        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            f"  lr={current_lr:.2e}"
        )

        # ── Save best checkpoint ───────────────────────────────────────────────
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            patience_counter = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
                "config":      config,
                "classes":     loaders["classes"],
            }, save_path)
            print(f"  Saved best checkpoint → {save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")

        # ── Early stopping ─────────────────────────────────────────────────────
        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved at: {save_path}")
    tracker.save(ckpt_dir / f"{config['phase']}_history.json")

    return model, tracker


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", default=None,  help="Checkpoint to resume from")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, checkpoint_path=args.checkpoint)