# Plant Disease Detector

An end-to-end transfer learning project that classifies plant leaf diseases
from photos using EfficientNetB0 pretrained on ImageNet.
Built to learn the three stages of transfer learning hands-on:
feature extraction → fine-tuning → comparison against a from-scratch baseline.

---

## Project structure

```
plant-disease-detector/
├── data/
│   ├── raw/              ← PlantVillage images (downloaded below)
│   └── processed/        ← resized/normalised images (generated)
├── src/
│   ├── dataset.py        ← DataLoader, augmentation, train/val/test split
│   ├── model.py          ← EfficientNetB0 + custom head, freeze logic
│   ├── train.py          ← training loop, checkpointing, early stopping
│   ├── evaluate.py       ← metrics, confusion matrix, phase comparison
│   ├── gradcam.py        ← Grad-CAM heatmap generation
│   └── utils.py          ← seeding, device, MetricTracker, helpers
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_phase_a_feature_extraction.ipynb
│   ├── 03_phase_b_fine_tuning.ipynb
│   └── 04_baseline_comparison.ipynb
├── checkpoints/          ← saved .pt files (gitignored)
├── app/
│   ├── app.py            ← Gradio web UI
│   └── requirements.txt  ← minimal app dependencies
├── configs/
│   ├── phase_a.yaml      ← frozen base, LR=1e-3, 10 epochs
│   ├── phase_b.yaml      ← unfreeze last 3 blocks, LR=1e-5, 15 epochs
│   └── baseline.yaml     ← no pretrained weights, 25 epochs
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1 — Clone and create environment

```bash
git clone https://github.com/YOUR_USERNAME/plant-disease-detector.git
cd plant-disease-detector

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2 — Configure environment variables

```bash
cp .env.example .env
# Open .env and fill in your KAGGLE_USERNAME and KAGGLE_KEY
```

### 3 — Download the dataset

```bash
pip install kaggle

kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d data/raw
```

After unzipping, `data/raw/` should contain 38 sub-folders,
one per disease class (e.g. `Apple___Apple_scab/`, `Tomato___healthy/`).

### 4 — Verify everything loads

```bash
python src/dataset.py   # should print 38 classes + batch shape [32, 3, 224, 224]
python src/model.py     # should print parameter counts + "Passed"
python src/utils.py     # should print device + MetricTracker table
```

---

## Training

Run the three phases in order:

```bash
# Phase A — frozen base, only the head trains (~10 min on GPU)
python src/train.py --config configs/phase_a.yaml

# Phase B — unfreeze last 3 blocks, start from Phase A weights (~20 min on GPU)
python src/train.py --config configs/phase_b.yaml \
       --checkpoint checkpoints/phase_a_best.pt

# Baseline — no pretrained weights, full network trains from scratch (~45 min on GPU)
python src/train.py --config configs/baseline.yaml
```

Checkpoints are saved to `checkpoints/` automatically.
Each phase saves `{phase}_best.pt` (lowest val loss) and `{phase}_history.json`.

---

## Evaluation

```bash
# Evaluate Phase B on the test set
python src/evaluate.py \
       --checkpoint checkpoints/phase_b_best.pt \
       --data_dir   data/raw

# Compare all three phases side by side
python src/evaluate.py \
       --checkpoint checkpoints/phase_a_best.pt \
       --data_dir   data/raw \
       --compare    checkpoints/phase_b_best.pt \
                    checkpoints/baseline_best.pt
```

Expected comparison output:

```
── Phase comparison ──────────────────────────────────────────────────
  Phase        Top-1 acc  Top-5 acc   F1 macro  F1 weighted
  ──────────────────────────────────────────────────────────────────
  phase_a         0.8821     0.9912     0.8803       0.8819
  phase_b         0.9534     0.9981     0.9528       0.9531
  baseline        0.7102     0.9341     0.7089       0.7094
```

---

## Grad-CAM visualisation

```bash
python src/gradcam.py \
       --image      data/raw/Tomato___Late_blight/img_001.jpg \
       --checkpoint checkpoints/phase_b_best.pt \
       --save       checkpoints/gradcam/example.png
```

This produces a three-panel figure:
original image | raw heatmap | heatmap blended over the leaf.
Red regions are where the model looked to make its decision.

---

## Launch the app

```bash
# Local
python app/app.py --checkpoint checkpoints/phase_b_best.pt

# With a public share link
python app/app.py --checkpoint checkpoints/phase_b_best.pt --share
```

Open `http://localhost:7860` in your browser,
upload a leaf photo, and get a prediction + Grad-CAM overlay instantly.

### Deploy to Hugging Face Spaces (free)

```bash
# 1 — Create a new Space at huggingface.co/new-space  (SDK: Gradio)
# 2 — Push
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/plant-disease-detector
git add app/ src/ checkpoints/phase_b_best.pt configs/
git commit -m "deploy"
git push hf main
```

---

## Key concepts learned

| Concept | Where it appears |
|---|---|
| Pretrained weights | `model.py` — `EfficientNet_B0_Weights.IMAGENET1K_V1` |
| Freezing layers | `model.py` — `param.requires_grad = False` |
| Feature extraction | `configs/phase_a.yaml` + Phase A training |
| Fine-tuning | `configs/phase_b.yaml` + `unfreeze_from=-3` |
| Learning rate scheduling | `train.py` — `ReduceLROnPlateau` |
| Early stopping | `train.py` — `patience_counter` |
| Label smoothing | `train.py` — `CrossEntropyLoss(label_smoothing=0.1)` |
| Grad-CAM | `gradcam.py` — forward + backward hooks |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU recommended (CPU works but is slow for training)
- ~3 GB disk space for the dataset
- ~500 MB for model checkpoints