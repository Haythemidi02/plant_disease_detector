import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)
from tqdm import tqdm

from dataset import get_dataloaders
from model import build_model
from utils import get_device, format_class_name


# ── Inference pass ────────────────────────────────────────────────────────────

def run_inference(model, loader, device, use_tta: bool = False) -> tuple[list, list, list]:
    """
    Runs the model over a dataloader without gradient computation.

    Returns:
        all_preds  : list of predicted class indices
        all_labels : list of ground-truth class indices
        all_probs  : list of softmax probability vectors (one per image)
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Running inference"):
            images = images.to(device)
            if use_tta:
                # Basic Test-Time Augmentation: average predictions of original and flipped images
                logits1 = model(images)
                logits2 = model(torch.flip(images, dims=[3])) # Horizontal flip
                logits3 = model(torch.flip(images, dims=[2])) # Vertical flip
                logits = (logits1 + logits2 + logits3) / 3.0
            else:
                logits = model(images)
            probs  = torch.softmax(logits, dim=1)
            preds  = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    return all_preds, all_labels, all_probs


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(
    preds:  list,
    labels: list,
    probs:  list,
    classes: list[str],
) -> dict:
    """
    Computes a full set of evaluation metrics.

    Returns a dict with:
        accuracy    : top-1 accuracy
        top5_acc    : top-5 accuracy
        f1_macro    : macro-averaged F1
        f1_weighted : weighted F1
        report      : per-class precision / recall / F1 (dict)
    """
    preds  = np.array(preds)
    labels = np.array(labels)
    probs  = np.array(probs)

    accuracy    = (preds == labels).mean()
    top_k       = min(5, len(classes))
    top5_acc    = top_k_accuracy_score(labels, probs, k=top_k, labels=np.arange(len(classes)))
    f1_macro    = f1_score(labels, preds, average="macro",    zero_division=0)
    f1_weighted = f1_score(labels, preds, average="weighted", zero_division=0)

    report = classification_report(
        labels, preds,
        target_names = classes,
        output_dict  = True,
        zero_division = 0,
    )

    return {
        "accuracy":    round(float(accuracy),    4),
        "top5_acc":    round(float(top5_acc),    4),
        "f1_macro":    round(float(f1_macro),    4),
        "f1_weighted": round(float(f1_weighted), 4),
        "report":      report,
    }


def print_metrics(metrics: dict):
    """Prints a clean summary of the computed metrics."""
    print("\n── Evaluation results ──────────────────────────────")
    print(f"  Top-1 accuracy  : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)")
    print(f"  Top-5 accuracy  : {metrics['top5_acc']:.4f}  ({metrics['top5_acc']*100:.2f}%)")
    print(f"  F1 macro        : {metrics['f1_macro']:.4f}")
    print(f"  F1 weighted     : {metrics['f1_weighted']:.4f}")
    print("────────────────────────────────────────────────────\n")


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(
    preds:   list,
    labels:  list,
    classes: list[str],
    save_path: str = None,
    figsize: tuple = (22, 20),
):
    """
    Plots a normalised confusion matrix using seaborn.
    With 38 classes this gets busy — normalisation keeps it readable.

    Args:
        preds     : predicted class indices
        labels    : ground-truth class indices
        classes   : list of class name strings
        save_path : if provided, saves the figure to this path
        figsize   : matplotlib figure size
    """
    cm = confusion_matrix(labels, preds, normalize="true")
    readable = [format_class_name(c) for c in classes]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm,
        annot      = True,
        fmt        = ".2f",
        cmap       = "Blues",
        xticklabels = readable,
        yticklabels = readable,
        linewidths  = 0.4,
        linecolor   = "lightgray",
        ax          = ax,
        annot_kws   = {"size": 7},
    )
    ax.set_xlabel("Predicted",   fontsize=12, labelpad=10)
    ax.set_ylabel("Ground truth", fontsize=12, labelpad=10)
    ax.set_title("Confusion matrix (normalised)", fontsize=14, pad=14)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Confusion matrix saved → {save_path}")

    plt.show()


# ── Per-class breakdown ───────────────────────────────────────────────────────

def print_worst_classes(metrics: dict, classes: list[str], n: int = 10):
    """
    Prints the N classes with the lowest F1 score.
    Useful for spotting which diseases the model struggles with most.
    """
    report = metrics["report"]
    rows = []
    for cls in classes:
        if cls in report:
            rows.append((cls, report[cls]["f1-score"], report[cls]["support"]))

    rows.sort(key=lambda x: x[1])
    print(f"\n── Bottom {n} classes by F1 ─────────────────────────")
    print(f"  {'Class':<40} {'F1':>6}  {'Support':>8}")
    print("  " + "─" * 58)
    for cls, f1, support in rows[:n]:
        print(f"  {format_class_name(cls):<40} {f1:>6.4f}  {int(support):>8}")
    print()


def print_best_classes(metrics: dict, classes: list[str], n: int = 10):
    """Prints the N classes with the highest F1 score."""
    report = metrics["report"]
    rows = []
    for cls in classes:
        if cls in report:
            rows.append((cls, report[cls]["f1-score"], report[cls]["support"]))

    rows.sort(key=lambda x: x[1], reverse=True)
    print(f"\n── Top {n} classes by F1 ────────────────────────────")
    print(f"  {'Class':<40} {'F1':>6}  {'Support':>8}")
    print("  " + "─" * 58)
    for cls, f1, support in rows[:n]:
        print(f"  {format_class_name(cls):<40} {f1:>6.4f}  {int(support):>8}")
    print()


# ── Compare phases ────────────────────────────────────────────────────────────

def compare_checkpoints(results: dict):
    """
    Prints a side-by-side comparison table of multiple evaluated phases.

    Args:
        results : { "phase_a": metrics_dict, "phase_b": metrics_dict, ... }

    Usage:
        compare_checkpoints({
            "Phase A": metrics_a,
            "Phase B": metrics_b,
            "Baseline": metrics_scratch,
        })
    """
    print("\n── Phase comparison ─────────────────────────────────────────────────")
    print(f"  {'Phase':<12} {'Top-1 acc':>10} {'Top-5 acc':>10} {'F1 macro':>10} {'F1 weighted':>12}")
    print("  " + "─" * 58)
    for phase, m in results.items():
        print(
            f"  {phase:<12}"
            f"  {m['accuracy']:>9.4f}"
            f"  {m['top5_acc']:>9.4f}"
            f"  {m['f1_macro']:>9.4f}"
            f"  {m['f1_weighted']:>11.4f}"
        )
    print()


# ── Save results ──────────────────────────────────────────────────────────────

def save_results(metrics: dict, path: str):
    """Saves evaluation metrics to a JSON file (excluding full report for brevity)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {k: v for k, v in metrics.items() if k != "report"}
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved → {path}")


# ── Main evaluate function ────────────────────────────────────────────────────

def evaluate(checkpoint_path: str, data_dir: str, split: str = "test", use_tta: bool = False):
    """
    Full evaluation pipeline for a single checkpoint.

    Args:
        checkpoint_path : path to a .pt checkpoint file
        data_dir        : dataset root (e.g. "data/raw")
        split           : which split to evaluate — "val" or "test"

    Returns:
        metrics dict
    """
    device = get_device()

    # ── Load checkpoint ───────────────────────────────────────────────────────
    ckpt    = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    config  = ckpt["config"]

    # ── Rebuild model and load weights ────────────────────────────────────────
    model = build_model(
        num_classes   = len(classes),
        freeze_base   = False,
        pretrained    = config.get("pretrained", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    print(f"Loaded checkpoint: {checkpoint_path}  (epoch {ckpt['epoch']}, val_acc={ckpt['val_acc']:.4f})")

    # ── Data ──────────────────────────────────────────────────────────────────
    loaders = get_dataloaders(
        root_dir    = data_dir,
        val_split   = config.get("val_split", 0.15),
        test_split  = config.get("test_split", 0.10),
        batch_size  = config.get("batch_size", 32),
        num_workers = config.get("num_workers", 4),
        seed        = config.get("seed", 42),
    )
    loader = loaders[split]
    print(f"\nEvaluating on {split} set ({len(loader.dataset)} images) ...")

    # ── Inference + metrics ───────────────────────────────────────────────────
    preds, labels, probs = run_inference(model, loader, device, use_tta=use_tta)
    metrics = compute_metrics(preds, labels, probs, classes)

    print_metrics(metrics)
    print_worst_classes(metrics, classes)
    print_best_classes(metrics, classes)

    # ── Confusion matrix ──────────────────────────────────────────────────────
    phase     = config["phase"]
    cm_path   = f"checkpoints/{phase}_confusion_matrix.png"
    json_path = f"checkpoints/{phase}_results.json"

    plot_confusion_matrix(preds, labels, classes, save_path=cm_path)
    save_results(metrics, json_path)

    return metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--data_dir",   required=True, help="Dataset root directory")
    parser.add_argument("--split",      default="test", choices=["val", "test"])
    parser.add_argument("--compare",    nargs="+",      default=None,
                        help="Extra checkpoints to compare against. "
                             "e.g. --compare checkpoints/phase_b_best.pt checkpoints/baseline_best.pt")
    parser.add_argument("--tta",        action="store_true", help="Use Test-Time Augmentation (TTA)")
    args = parser.parse_args()

    # Single checkpoint evaluation
    metrics = evaluate(args.checkpoint, args.data_dir, args.split, use_tta=args.tta)

    # Optional side-by-side comparison
    if args.compare:
        device  = get_device()
        results = {"Primary": metrics}

        for ckpt_path in args.compare:
            ckpt    = torch.load(ckpt_path, map_location=device, weights_only=False)
            classes = ckpt["classes"]
            config  = ckpt["config"]
            model   = build_model(
                num_classes   = len(classes),
                freeze_base   = False,
                pretrained    = config.get("pretrained", True),
            ).to(device)
            model.load_state_dict(ckpt["model_state"])
            loaders = get_dataloaders(
                args.data_dir,
                val_split=config.get("val_split", 0.15),
                test_split=config.get("test_split", 0.10),
                batch_size=config.get("batch_size", 32),
                num_workers=config.get("num_workers", 4),
                seed=config.get("seed", 42),
            )
            preds, labels, probs = run_inference(model, loaders[args.split], device, use_tta=args.tta)
            results[config["phase"]] = compute_metrics(preds, labels, probs, classes)

        compare_checkpoints(results)
