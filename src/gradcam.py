import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from pathlib import Path
from torchvision import transforms

from model import build_model
from utils import get_device, load_checkpoint, format_class_name


# ── Grad-CAM core ─────────────────────────────────────────────────────────────

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Hooks into the last convolutional block of the model and computes
    a heatmap showing which spatial regions of the input image most
    influenced the predicted class.

    Usage:
        gcam    = GradCAM(model)
        heatmap = gcam(image_tensor, class_idx)
        gcam.remove_hooks()
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module = None):
        """
        Args:
            model        : trained EfficientNetB0
            target_layer : the conv layer to hook into.
                           Defaults to the last block of model.features.
        """
        self.model  = model
        self.model.eval()

        self._activations = None
        self._gradients   = None

        # Default: last feature block of EfficientNetB0
        if target_layer is None:
            target_layer = model.features[-1]

        # Forward hook — saves feature map activations
        self._fwd_hook = target_layer.register_forward_hook(
            self._save_activations
        )
        # Backward hook — saves gradients w.r.t. those activations
        self._bwd_hook = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, input, output):
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(
        self,
        image_tensor: torch.Tensor,
        class_idx: int = None,
    ) -> np.ndarray:
        """
        Computes a Grad-CAM heatmap for the given image tensor.

        Args:
            image_tensor : preprocessed image, shape (1, 3, 224, 224)
            class_idx    : class to explain. If None, uses the predicted class.

        Returns:
            heatmap : numpy array of shape (224, 224), values in [0, 1]
        """
        device = next(self.model.parameters()).device
        image_tensor = image_tensor.to(device).requires_grad_(False)

        # Forward pass
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()

        # Backward pass for the target class only
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # Global average pool the gradients → weights per channel
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activation maps
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = torch.relu(cam)                                           # ReLU: keep positives

        # Normalise to [0, 1]
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def remove_hooks(self):
        """Always call this when done to avoid memory leaks."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()


# ── Image preprocessing ───────────────────────────────────────────────────────

def preprocess_image(image_path: str) -> tuple[torch.Tensor, Image.Image]:
    """
    Loads and preprocesses a leaf image for inference.

    Returns:
        tensor    : (1, 3, 224, 224) preprocessed tensor
        original  : PIL Image at original size (for overlay)
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    original = Image.open(image_path).convert("RGB")
    tensor   = transform(original).unsqueeze(0)   # add batch dim
    return tensor, original


# ── Overlay builder ───────────────────────────────────────────────────────────

def overlay_heatmap(
    heatmap:  np.ndarray,
    original: Image.Image,
    alpha:    float = 0.45,
    colormap: str   = "jet",
) -> Image.Image:
    """
    Blends a Grad-CAM heatmap over the original image.

    Args:
        heatmap  : (H, W) float array in [0, 1]
        original : PIL Image
        alpha    : heatmap opacity (0 = invisible, 1 = full heatmap)
        colormap : matplotlib colormap name

    Returns:
        blended PIL Image (RGB)
    """
    # Resize heatmap to match original image dimensions
    h, w = np.array(original).shape[:2]
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(heatmap * 255)).resize((w, h), Image.BILINEAR)
    ) / 255.0

    # Apply colormap
    cmap    = plt.get_cmap(colormap)
    colored = cmap(heatmap_resized)[:, :, :3]           # drop alpha channel
    colored = (colored * 255).astype(np.uint8)
    colored = Image.fromarray(colored)

    # Blend with original
    original_arr = np.array(original.resize((w, h)))
    blended      = (
        (1 - alpha) * original_arr + alpha * np.array(colored)
    ).astype(np.uint8)

    return Image.fromarray(blended)


# ── Prediction with explanation ───────────────────────────────────────────────

def explain(
    image_path:      str,
    checkpoint_path: str,
    save_path:       str  = None,
    top_k:           int  = 5,
    alpha:           float = 0.45,
) -> dict:
    """
    Full pipeline: load model → predict → generate Grad-CAM → plot.

    Args:
        image_path      : path to a leaf image
        checkpoint_path : path to a .pt checkpoint
        save_path       : if provided, saves the plot to this path
        top_k           : number of top predictions to show
        alpha           : heatmap overlay opacity

    Returns:
        dict with keys: predicted_class, confidence, top_k_preds, heatmap
    """
    device = get_device()

    # ── Load model ────────────────────────────────────────────────────────────
    ckpt    = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    config  = ckpt["config"]

    model = build_model(
        num_classes   = len(classes),
        freeze_base   = False,
        pretrained    = config.get("pretrained", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # ── Preprocess ────────────────────────────────────────────────────────────
    tensor, original = preprocess_image(image_path)

    # ── Predict ───────────────────────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs  = torch.softmax(logits, dim=1).squeeze()

    top_probs, top_indices = probs.topk(top_k)
    top_probs   = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    pred_idx   = top_indices[0]
    pred_class = classes[pred_idx]
    confidence = top_probs[0]

    print(f"\nPrediction  : {format_class_name(pred_class)}")
    print(f"Confidence  : {confidence:.4f}  ({confidence*100:.2f}%)")
    print(f"\nTop {top_k} predictions:")
    for prob, idx in zip(top_probs, top_indices):
        print(f"  {format_class_name(classes[idx]):<40}  {prob*100:.2f}%")

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gcam    = GradCAM(model)
    heatmap = gcam(tensor, class_idx=pred_idx)
    gcam.remove_hooks()

    overlay = overlay_heatmap(heatmap, original, alpha=alpha)

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(original.resize((224, 224)))
    axes[0].set_title("Original image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(overlay.resize((224, 224)))
    axes[2].set_title(
        f"Overlay\n{format_class_name(pred_class)}  ({confidence*100:.1f}%)",
        fontsize=11,
    )
    axes[2].axis("off")

    plt.suptitle("Grad-CAM explanation", fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\nPlot saved → {save_path}")

    plt.show()

    return {
        "predicted_class": pred_class,
        "confidence":      float(confidence),
        "top_k_preds":     [
            {"class": classes[i], "prob": float(p)}
            for i, p in zip(top_indices, top_probs)
        ],
        "heatmap": heatmap,
    }


# ── Batch explanation ─────────────────────────────────────────────────────────

def explain_batch(
    image_paths:     list[str],
    checkpoint_path: str,
    save_dir:        str = "checkpoints/gradcam",
):
    """
    Runs explain() on a list of images and saves each plot.
    Useful for building a visual report of model behaviour.

    Args:
        image_paths     : list of paths to leaf images
        checkpoint_path : path to .pt checkpoint
        save_dir        : directory to save output plots
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        stem      = Path(path).stem
        save_path = save_dir / f"{stem}_gradcam.png"
        print(f"\n── {path} ──")
        explain(
            image_path      = path,
            checkpoint_path = checkpoint_path,
            save_path       = str(save_path),
        )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--image",      required=True,  help="Path to a leaf image")
    parser.add_argument("--checkpoint", required=True,  help="Path to .pt checkpoint")
    parser.add_argument("--save",       default=None,   help="Path to save the output plot")
    parser.add_argument("--top_k",      default=5,      type=int)
    parser.add_argument("--alpha",      default=0.45,   type=float,
                        help="Heatmap overlay opacity (0-1)")
    args = parser.parse_args()

    explain(
        image_path      = args.image,
        checkpoint_path = args.checkpoint,
        save_path       = args.save,
        top_k           = args.top_k,
        alpha           = args.alpha,
    )