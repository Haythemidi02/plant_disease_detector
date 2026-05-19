import sys
from pathlib import Path

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import numpy as np
import gradio as gr
from PIL import Image
from torchvision import transforms

from model import build_model
from utils import get_device, format_class_name
from gradcam import GradCAM, overlay_heatmap


# ── Load model once at startup ────────────────────────────────────────────────

def load_model(checkpoint_path: str):
    """
    Loads the model and class list from a checkpoint.
    Called once when the app starts — not on every prediction.

    Returns:
        model   : nn.Module in eval mode
        classes : list of class name strings
        device  : torch.device
    """
    device = get_device()
    ckpt   = torch.load(checkpoint_path, map_location=device, weights_only=False)

    classes = ckpt["classes"]
    config  = ckpt["config"]

    model = build_model(
        num_classes   = len(classes),
        freeze_base   = False,
        pretrained    = config.get("pretrained", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"Model loaded  : {checkpoint_path}")
    print(f"Classes       : {len(classes)}")
    print(f"Device        : {device}")

    return model, classes, device


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess(pil_image: Image.Image) -> torch.Tensor:
    """
    Applies the same val/test transforms used during training.
    Returns a (1, 3, 224, 224) tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std =[0.229, 0.224, 0.225],
        ),
    ])
    return transform(pil_image.convert("RGB")).unsqueeze(0)


# ── Core prediction function ──────────────────────────────────────────────────

def predict(
    pil_image: Image.Image,
    model,
    classes:   list[str],
    device:    torch.device,
    top_k:     int   = 5,
    alpha:     float = 0.45,
) -> tuple:
    """
    Runs inference + Grad-CAM on a PIL image.

    Returns:
        label_probs : dict  {readable_label: confidence}  for Gradio Label
        overlay_img : PIL Image  (original + heatmap blended)
        report      : str  plain-text summary
    """
    tensor = preprocess(pil_image).to(device)

    # ── Inference ─────────────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1).squeeze()

    top_probs, top_indices = probs.topk(top_k)
    top_probs   = top_probs.cpu().numpy()
    top_indices = top_indices.cpu().numpy()

    pred_idx   = int(top_indices[0])
    pred_class = classes[pred_idx]
    confidence = float(top_probs[0])

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    gcam    = GradCAM(model)
    heatmap = gcam(tensor, class_idx=pred_idx)
    gcam.remove_hooks()

    overlay = overlay_heatmap(heatmap, pil_image, alpha=alpha)

    # ── Outputs ───────────────────────────────────────────────────────────────
    label_probs = {
        format_class_name(classes[int(i)]): float(p)
        for i, p in zip(top_indices, top_probs)
    }

    is_healthy = "healthy" in pred_class.lower()
    status     = "Healthy" if is_healthy else "Disease detected"

    report = (
        f"Status      : {status}\n"
        f"Prediction  : {format_class_name(pred_class)}\n"
        f"Confidence  : {confidence*100:.2f}%\n\n"
        f"Top {top_k} predictions:\n"
        + "\n".join(
            f"  {format_class_name(classes[int(i)]):<40}  {float(p)*100:.2f}%"
            for i, p in zip(top_indices, top_probs)
        )
    )

    return label_probs, overlay, report


# ── Gradio interface ──────────────────────────────────────────────────────────

def build_app(checkpoint_path: str) -> gr.Blocks:
    """
    Builds and returns the Gradio Blocks interface.

    Args:
        checkpoint_path : path to the best .pt checkpoint

    Returns:
        gr.Blocks app (call .launch() to start)
    """
    model, classes, device = load_model(checkpoint_path)

    # Wrap predict() so Gradio can call it with just the image + sliders
    def run(image, top_k, alpha):
        if image is None:
            return {}, None, "Upload a leaf image to get started."
        return predict(image, model, classes, device, int(top_k), float(alpha))

    # ── Layout ────────────────────────────────────────────────────────────────
    with gr.Blocks(title="Plant Disease Detector") as app:

        gr.Markdown(
            """
            # Plant Disease Detector
            Upload a photo of a plant leaf.
            The model will identify the disease (or confirm it is healthy)
            and highlight the leaf regions that drove the prediction.
            """
        )

        with gr.Row():

            # Left column — inputs
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type  = "pil",
                    label = "Leaf image",
                )
                top_k_slider = gr.Slider(
                    minimum = 1,
                    maximum = 10,
                    value   = 5,
                    step    = 1,
                    label   = "Top-K predictions to show",
                )
                alpha_slider = gr.Slider(
                    minimum = 0.1,
                    maximum = 0.9,
                    value   = 0.45,
                    step    = 0.05,
                    label   = "Heatmap opacity",
                )
                submit_btn = gr.Button("Analyse", variant="primary")

            # Right column — outputs
            with gr.Column(scale=1):
                label_output = gr.Label(
                    num_top_classes = 5,
                    label           = "Predicted class (confidence)",
                )
                overlay_output = gr.Image(
                    type  = "pil",
                    label = "Grad-CAM overlay  (red = model focused here)",
                )
                report_output = gr.Textbox(
                    label    = "Report",
                    lines    = 10,
                    max_lines= 15,
                )

        # ── Example images ─────────────────────────────────────────────────────
        example_dir = Path("data/raw")
        examples    = []
        if example_dir.exists():
            for cls_dir in sorted(example_dir.iterdir())[:6]:
                imgs = list(cls_dir.glob("*.jpg"))
                if imgs:
                    examples.append([str(imgs[0]), 5, 0.45])

        if examples:
            gr.Examples(
                examples        = examples,
                inputs          = [image_input, top_k_slider, alpha_slider],
                outputs         = [label_output, overlay_output, report_output],
                fn              = run,
                cache_examples  = False,
                label           = "Example leaf images",
            )

        # ── Info accordion ─────────────────────────────────────────────────────
        with gr.Accordion("About this model", open=False):
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            gr.Markdown(
                f"""
                **Backbone**    : EfficientNetB0 pretrained on ImageNet  
                **Phase**       : {ckpt['config']['phase']}  
                **Classes**     : {len(classes)}  
                **Val accuracy**: {ckpt['val_acc']*100:.2f}%  
                **Val loss**    : {ckpt['val_loss']:.4f}  
                **Checkpoint**  : `{checkpoint_path}`  

                The heatmap is generated with **Grad-CAM** — it shows which
                pixels in the leaf most influenced the prediction.
                Red = high attention, blue = low attention.
                """
            )

        # ── Wire up the button ─────────────────────────────────────────────────
        submit_btn.click(
            fn      = run,
            inputs  = [image_input, top_k_slider, alpha_slider],
            outputs = [label_output, overlay_output, report_output],
        )

        # Also trigger on image upload (no need to click the button)
        image_input.change(
            fn      = run,
            inputs  = [image_input, top_k_slider, alpha_slider],
            outputs = [label_output, overlay_output, report_output],
        )

    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default = "checkpoints/phase_b_best.pt",
        help    = "Path to the .pt checkpoint to serve",
    )
    parser.add_argument(
        "--share",
        action  = "store_true",
        help    = "Create a public Gradio share link",
    )
    parser.add_argument(
        "--port",
        default = 7860,
        type    = int,
    )
    args = parser.parse_args()

    app = build_app(args.checkpoint)
    app.launch(
        server_port = args.port,
        share       = args.share,
    )