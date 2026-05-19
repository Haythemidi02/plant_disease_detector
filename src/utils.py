import os
import json
import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime


# ── Reproducibility ───────────────────────────────────────────────────────────

def seed_everything(seed: int = 42):
    """
    Seeds all random number generators for full reproducibility.
    Call this once at the start of train.py before anything else.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"]       = str(seed)


# ── Device ────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """
    Returns the best available device in order: CUDA → MPS → CPU.
    MPS is Apple Silicon (M1/M2/M3 Macs).
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    return device


# ── Class name mapping ────────────────────────────────────────────────────────

def get_class_names(root_dir: str) -> list[str]:
    """
    Scans the dataset root and returns class names in sorted order.
    Matches exactly what PlantDataset builds internally so indices
    are always consistent.

    Args:
        root_dir : path to dataset root (e.g. "data/raw")

    Returns:
        list of class name strings, e.g. ["Apple___Apple_scab", ...]
    """
    root = Path(root_dir)
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def idx_to_class(root_dir: str) -> dict:
    """Returns {index: class_name} mapping."""
    names = get_class_names(root_dir)
    return {idx: name for idx, name in enumerate(names)}


def class_to_idx(root_dir: str) -> dict:
    """Returns {class_name: index} mapping."""
    names = get_class_names(root_dir)
    return {name: idx for idx, name in enumerate(names)}


def format_class_name(raw: str) -> str:
    """
    Converts a raw folder name into a human-readable label.
    e.g. "Tomato___Late_blight" → "Tomato — Late blight"
         "Apple___healthy"      → "Apple — Healthy"
    """
    parts = raw.split("___")
    if len(parts) == 2:
        plant, condition = parts
        condition = condition.replace("_", " ").capitalize()
        return f"{plant} — {condition}"
    return raw.replace("_", " ")


# ── Metric tracker ────────────────────────────────────────────────────────────

class MetricTracker:
    """
    Accumulates per-epoch metrics during training and provides
    utilities to print a summary and save history to JSON.

    Usage:
        tracker = MetricTracker()
        tracker.update(epoch=1, train_loss=0.8, train_acc=0.72,
                       val_loss=0.6, val_acc=0.80, lr=1e-3)
        tracker.summary()
        tracker.save("checkpoints/phase_a_history.json")
    """

    def __init__(self):
        self.history = []

    def update(
        self,
        epoch:      int,
        train_loss: float,
        train_acc:  float,
        val_loss:   float,
        val_acc:    float,
        lr:         float,
    ):
        self.history.append({
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "train_acc":  round(train_acc,  6),
            "val_loss":   round(val_loss,   6),
            "val_acc":    round(val_acc,    6),
            "lr":         lr,
        })

    def best_epoch(self) -> dict:
        """Returns the epoch row with the lowest val_loss."""
        return min(self.history, key=lambda r: r["val_loss"])

    def summary(self):
        """Prints a formatted table of all epochs."""
        header = f"{'Epoch':>6}  {'Train loss':>10}  {'Train acc':>9}  {'Val loss':>8}  {'Val acc':>8}  {'LR':>8}"
        print("\n" + header)
        print("─" * len(header))
        for row in self.history:
            print(
                f"{row['epoch']:>6}  "
                f"{row['train_loss']:>10.4f}  "
                f"{row['train_acc']:>9.4f}  "
                f"{row['val_loss']:>8.4f}  "
                f"{row['val_acc']:>8.4f}  "
                f"{row['lr']:>8.2e}"
            )
        best = self.best_epoch()
        print(f"\nBest epoch: {best['epoch']}  val_loss={best['val_loss']:.4f}  val_acc={best['val_acc']:.4f}\n")

    def save(self, path):
        """Saves history + metadata to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "saved_at": datetime.now().isoformat(),
            "best":     self.best_epoch(),
            "history":  self.history,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"History saved → {path}")

    @classmethod
    def load(cls, path) -> "MetricTracker":
        """Loads a previously saved history JSON back into a tracker."""
        with open(path, "r") as f:
            payload = json.load(f)
        tracker = cls()
        tracker.history = payload["history"]
        return tracker


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint(checkpoint_path: str, model, device: torch.device) -> dict:
    """
    Loads a saved checkpoint into a model in-place.
    Returns the full checkpoint dict (contains epoch, val_acc, classes, config).

    Args:
        checkpoint_path : path to .pt file
        model           : nn.Module to load weights into
        device          : torch.device

    Returns:
        checkpoint dict with keys: epoch, model_state, val_loss,
                                   val_acc, config, classes
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch    : {ckpt['epoch']}")
    print(f"  Val loss : {ckpt['val_loss']:.4f}")
    print(f"  Val acc  : {ckpt['val_acc']:.4f}")
    return ckpt


def checkpoint_info(checkpoint_path: str):
    """Prints checkpoint metadata without loading into a model."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    print(f"\nCheckpoint : {checkpoint_path}")
    print(f"  Phase    : {ckpt['config']['phase']}")
    print(f"  Epoch    : {ckpt['epoch']}")
    print(f"  Val loss : {ckpt['val_loss']:.4f}")
    print(f"  Val acc  : {ckpt['val_acc']:.4f}")
    print(f"  Classes  : {len(ckpt['classes'])}")


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Device
    device = get_device()

    # Seeding
    seed_everything(42)
    print("Seeding OK")

    # Class name formatting
    samples = [
        "Apple___Apple_scab",
        "Tomato___Late_blight",
        "Grape___healthy",
    ]
    print("\nClass name formatting:")
    for s in samples:
        print(f"  {s:35s} → {format_class_name(s)}")

    # MetricTracker
    print("\nMetricTracker:")
    tracker = MetricTracker()
    for i in range(1, 5):
        tracker.update(
            epoch      = i,
            train_loss = 1.0 - i * 0.1,
            train_acc  = 0.6 + i * 0.05,
            val_loss   = 1.1 - i * 0.09,
            val_acc    = 0.58 + i * 0.05,
            lr         = 1e-3,
        )
    tracker.summary()

    # Save and reload
    import tempfile
    tmp_path = Path(tempfile.gettempdir()) / "test_history.json"
    tracker.save(str(tmp_path))
    reloaded = MetricTracker.load(str(tmp_path))
    print(f"Reloaded {len(reloaded.history)} epochs from disk. OK")