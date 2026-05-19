import torch
import torch.nn as nn
from torchvision import models


# ── Build model ───────────────────────────────────────────────────────────────

def build_model(
    num_classes: int,
    freeze_base: bool = True,
    unfreeze_from: int = None,
    pretrained: bool = True,
) -> nn.Module:
    """
    Loads EfficientNetB0 and replaces its classifier head
    for your number of plant disease classes.

    Args:
        num_classes   : number of output classes (38 for PlantVillage)
        freeze_base   : if True, freeze all base layers (Phase A)
        unfreeze_from : if set, unfreeze layers from this index onward
                        counted from the END of features (Phase B).
                        e.g. unfreeze_from=-20 unfreezes last 20 layers.
        pretrained    : if True, load ImageNet pretrained weights;
                        if False, train from random init (baseline)

    Returns:
        model : nn.Module ready to move to device and train
    """
    # Load backbone (with or without pretrained weights)
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)

    # ── Step 1: freeze the entire base ────────────────────────────────────────
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False

    # ── Step 2: selectively unfreeze from a layer index (Phase B) ─────────────
    if unfreeze_from is not None:
        feature_layers = list(model.features.children())
        layers_to_unfreeze = feature_layers[unfreeze_from:]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True

    # ── Step 3: replace the classifier head ───────────────────────────────────
    # EfficientNetB0's original head: Linear(1280 → 1000)
    # We replace it with: Dropout → Linear(1280 → num_classes)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    # The new head is always trainable
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


# ── Inspect helpers ───────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> dict:
    """
    Returns total and trainable parameter counts.
    Useful for verifying freeze/unfreeze worked correctly.
    """
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """
    Keeps BatchNorm statistics fixed for frozen feature layers.

    Freezing parameters alone does not stop BatchNorm running_mean and
    running_var from changing during model.train(). This makes feature
    extraction behave like a cleaner transfer-learning experiment.
    """
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            params = list(module.parameters())
            if params and not any(param.requires_grad for param in params):
                module.eval()


def describe_trainable_blocks(model: nn.Module) -> list[dict]:
    """Returns feature/head trainability in a notebook-friendly structure."""
    rows = []
    for name, child in model.features.named_children():
        total = sum(p.numel() for p in child.parameters())
        trainable = sum(p.numel() for p in child.parameters() if p.requires_grad)
        rows.append({
            "block": f"features.{name}",
            "trainable": trainable > 0,
            "trainable_params": trainable,
            "total_params": total,
        })

    total = sum(p.numel() for p in model.classifier.parameters())
    trainable = sum(p.numel() for p in model.classifier.parameters() if p.requires_grad)
    rows.append({
        "block": "classifier",
        "trainable": trainable > 0,
        "trainable_params": trainable,
        "total_params": total,
    })
    return rows


def print_layer_status(model: nn.Module):
    """
    Prints each feature block with its trainable status.
    Helps you visually confirm which layers are frozen vs unfrozen.
    """
    print(f"\n{'Block':<10} {'Trainable':<12} {'Params':>10}")
    print("─" * 36)
    for name, child in model.features.named_children():
        trainable = any(p.requires_grad for p in child.parameters())
        n_params  = sum(p.numel() for p in child.parameters())
        status    = "YES" if trainable else "frozen"
        print(f"  [{name}]   {status:<12} {n_params:>10,}")
    # Classifier head
    trainable = any(p.requires_grad for p in model.classifier.parameters())
    n_params  = sum(p.numel() for p in model.classifier.parameters())
    print(f"  [head]   {'YES':<12} {n_params:>10,}")
    print()


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    NUM_CLASSES = 38

    # ── Phase A: only head is trainable ───────────────────────────────────────
    print("=" * 40)
    print("PHASE A — feature extraction (frozen base)")
    print("=" * 40)
    model_a = build_model(num_classes=NUM_CLASSES, freeze_base=True)
    print_layer_status(model_a)
    stats_a = count_parameters(model_a)
    print(f"  Total      : {stats_a['total']:>10,}")
    print(f"  Trainable  : {stats_a['trainable']:>10,}")
    print(f"  Frozen     : {stats_a['frozen']:>10,}")

    # ── Phase B: last 20 layers + head are trainable ──────────────────────────
    print("\n" + "=" * 40)
    print("PHASE B — fine-tuning (last 20 layers unfrozen)")
    print("=" * 40)
    model_b = build_model(
        num_classes=NUM_CLASSES,
        freeze_base=True,
        unfreeze_from=-3,       # EfficientNetB0 has 9 feature blocks;
    )                           # -3 unfreezes the last 3 (equiv. ~20 layers)
    print_layer_status(model_b)
    stats_b = count_parameters(model_b)
    print(f"  Total      : {stats_b['total']:>10,}")
    print(f"  Trainable  : {stats_b['trainable']:>10,}")
    print(f"  Frozen     : {stats_b['frozen']:>10,}")

    # ── Baseline: everything trainable ────────────────────────────────────────
    print("\n" + "=" * 40)
    print("BASELINE — training from scratch")
    print("=" * 40)
    model_scratch = build_model(
        num_classes=NUM_CLASSES,
        freeze_base=False,
        pretrained=False,
    )
    stats_s = count_parameters(model_scratch)
    print(f"  Total      : {stats_s['total']:>10,}")
    print(f"  Trainable  : {stats_s['trainable']:>10,}")

    # ── Forward pass check ────────────────────────────────────────────────────
    print("\n── Forward pass check ──")
    dummy = torch.randn(4, 3, 224, 224)   # batch of 4 images
    out   = model_a(dummy)
    print(f"  Input  : {tuple(dummy.shape)}")
    print(f"  Output : {tuple(out.shape)}")  # → (4, 38)
    assert out.shape == (4, NUM_CLASSES), "Output shape mismatch!"
    print("  Passed.")
