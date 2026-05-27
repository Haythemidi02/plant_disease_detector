import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from pathlib import Path
from tqdm import tqdm
from torchvision import models

from dataset import get_dataloaders
from model import build_model
from utils import seed_everything, get_device, MetricTracker


def build_student_model(num_classes: int) -> nn.Module:
    """Builds a MobileNetV3-Small student model from scratch."""
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    alpha: float,
) -> torch.Tensor:
    """
    Computes the Knowledge Distillation loss.
    Loss = (1 - alpha) * CrossEntropy(student, labels) +
           alpha * (T^2) * KL_Divergence(student / T, teacher / T)
    """
    # Standard Cross-Entropy with true labels
    ce_loss = F.cross_entropy(student_logits, labels)

    # Soft labels from teacher
    teacher_probs = F.softmax(teacher_logits / temperature, dim=1)
    student_log_probs = F.log_softmax(student_logits / temperature, dim=1)

    # KL Divergence between softened teacher and student probabilities
    kl_div_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')

    return (1. - alpha) * ce_loss + alpha * (temperature ** 2) * kl_div_loss


def run_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    is_train: bool,
    config: dict,
    scaler: GradScaler = None,
) -> tuple[float, float]:
    student.train() if is_train else student.eval()
    
    total_loss = 0.0
    correct = 0
    total_samples = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    temperature = config.get("temperature", 3.0)
    alpha = config.get("alpha", 0.5)

    with ctx:
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            with autocast(device_type=device.type, enabled=scaler is not None):
                # Teacher forward pass (no grad)
                with torch.no_grad():
                    teacher_logits = teacher(images)

                # Student forward pass
                student_logits = student(images)

                # Distillation loss
                loss = distillation_loss(student_logits, teacher_logits, labels, temperature, alpha)

            if is_train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = student_logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy


def train_distillation(config: dict):
    seed_everything(config.get("seed", 42))
    device = get_device()

    print(f"\nPhase: Knowledge Distillation")
    print(f"Device: {device}")
    
    loaders = get_dataloaders(
        root_dir=config["data_dir"],
        val_split=config.get("val_split", 0.15),
        test_split=config.get("test_split", 0.10),
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        seed=config.get("seed", 42),
    )
    num_classes = loaders["num_classes"]

    # Load Teacher Model
    teacher_path = config["teacher_checkpoint"]
    print(f"Loading Teacher from {teacher_path}...")
    teacher = build_model(num_classes=num_classes, freeze_base=True, pretrained=False).to(device)
    state = torch.load(teacher_path, map_location=device, weights_only=False)
    teacher.load_state_dict(state["model_state"])
    teacher.eval() # Teacher is always in eval mode

    # Build Student Model
    print("Building Student (MobileNetV3-Small)...")
    student = build_student_model(num_classes).to(device)

    optimizer = AdamW(student.parameters(), lr=config["student_lr"], weight_decay=config.get("weight_decay", 1e-4))
    
    use_amp = device.type == "cuda"
    scaler = GradScaler("cuda") if use_amp else None

    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=config.get("lr_patience", 3)
    )

    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_path = ckpt_dir / f"{config['phase']}_student_best.pt"

    tracker = MetricTracker()
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = config.get("early_stop_patience", 7)

    for epoch in range(1, config["epochs"] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")

        train_loss, train_acc = run_epoch(
            student, teacher, loaders["train"], optimizer, device, is_train=True, config=config, scaler=scaler
        )
        val_loss, val_acc = run_epoch(
            student, teacher, loaders["val"], optimizer, device, is_train=False, config=config, scaler=scaler
        )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        tracker.update(
            epoch=epoch, train_loss=train_loss, train_acc=train_acc, val_loss=val_loss, val_acc=val_acc, lr=current_lr
        )

        print(
            f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
            f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
            f"  lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state": student.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "config": config,
                "classes": loaders["classes"],
            }, save_path)
            print(f"  Saved best student checkpoint → {save_path}")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")

        if patience_counter >= early_stop_patience:
            print(f"\nEarly stopping triggered at epoch {epoch}.")
            break

    print(f"\nDistillation complete. Best val loss: {best_val_loss:.4f}")
    tracker.summary()
    tracker.save(ckpt_dir / f"{config['phase']}_history.json")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config")
    args = parser.parse_args()
    config = load_config(args.config)
    train_distillation(config)
