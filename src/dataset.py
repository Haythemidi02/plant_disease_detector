import os
import torch
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(split: str) -> transforms.Compose:
    """
    Returns augmentation pipeline per split.
    - train : random flips, rotation, color jitter → stronger generalization
    - val/test : only resize + normalize (no randomness)
    """
    mean = [0.485, 0.456, 0.406]   # ImageNet stats — required because
    std  = [0.229, 0.224, 0.225]   # we use an ImageNet-pretrained backbone

    if split == "train":
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(30),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class PlantDataset(Dataset):
    """
    Expects PlantVillage folder structure:
        data/raw/
            Apple___Apple_scab/
                image1.jpg
                image2.jpg
            Apple___Black_rot/
                ...
            Tomato___healthy/
                ...

    Each sub-folder name becomes a class label.
    """

    def __init__(self, root_dir: str, split: str = "train"):
        """
        Args:
            root_dir : path to the dataset root (e.g. "data/raw")
            split    : one of "train", "val", "test"
        """
        self.root_dir  = Path(root_dir)
        self.split     = split
        self.transform = get_transforms(split)

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root_dir}. "
                "Download/unzip PlantVillage into data/raw first."
            )

        # Build class list (sorted for reproducibility)
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect (image_path, label) pairs
        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if not self.classes:
            raise ValueError(f"No class folders found under {self.root_dir}.")
        if not self.samples:
            raise ValueError(
                f"No images with extensions {sorted(IMAGE_EXTENSIONS)} found under {self.root_dir}."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")  # ensure 3 channels
        image = self.transform(image)
        return image, label

    def __repr__(self) -> str:
        return (f"PlantDataset(split={self.split}, "
                f"classes={len(self.classes)}, "
                f"samples={len(self.samples)})")


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(
    root_dir: str,
    val_split: float = 0.15,
    test_split: float = 0.10,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
) -> dict:
    """
    Splits the full dataset into train / val / test and returns
    a dict of DataLoaders ready to pass to the training loop.

    Args:
        root_dir    : path to dataset root
        val_split   : fraction of data for validation
        test_split  : fraction of data for testing
        batch_size  : images per batch
        num_workers : parallel workers for loading
        seed        : for reproducible splits

    Returns:
        {
          "train"   : DataLoader,
          "val"     : DataLoader,
          "test"    : DataLoader,
          "classes" : List[str],   ← class names in label order
          "num_classes": int,
        }
    """
    # Build the full dataset with train transforms first (split later)
    full_dataset = PlantDataset(root_dir, split="train")
    n = len(full_dataset)
    if val_split <= 0 or test_split <= 0 or val_split + test_split >= 1:
        raise ValueError("val_split and test_split must be positive and sum to less than 1.")

    # Extract labels for stratification
    labels = [label for _, label in full_dataset.samples]
    class_counts = torch.bincount(torch.tensor(labels), minlength=len(full_dataset.classes))
    if int(class_counts.min()) < 3:
        raise ValueError(
            "Each class needs at least 3 images for stratified train/val/test splitting. "
            f"Smallest class has {int(class_counts.min())}."
        )

    # First split: separate test set
    train_val_idx, test_idx = train_test_split(
        range(n),
        test_size=test_split,
        stratify=labels,
        random_state=seed,
    )

    # Second split: separate val from train
    train_val_labels = [labels[i] for i in train_val_idx]
    val_fraction = val_split / (1 - test_split)  # adjust fraction
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        stratify=train_val_labels,
        random_state=seed,
    )

    # Create subsets with correct transforms
    train_ds = _SubsetWithTransform(full_dataset, train_idx, "train")
    val_ds   = _SubsetWithTransform(full_dataset, val_idx,   "val")
    test_ds  = _SubsetWithTransform(full_dataset, test_idx,  "test")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=True if num_workers > 0 else False,
        )

    return {
        "train":       make_loader(train_ds, shuffle=True),
        "val":         make_loader(val_ds,   shuffle=False),
        "test":        make_loader(test_ds,  shuffle=False),
        "classes":     full_dataset.classes,
        "num_classes": len(full_dataset.classes),
    }


class _SubsetWithTransform(Dataset):
    """
    Wraps a subset of PlantDataset with a different transform.
    Used internally by get_dataloaders to apply val/test transforms
    to the val and test splits without duplicating the file scan.
    """

    def __init__(self, source: PlantDataset, indices, split: str):
        self.samples   = [source.samples[i] for i in indices]
        self.transform = get_transforms(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    loaders = get_dataloaders("data/raw", batch_size=32)

    print(f"Classes  : {loaders['num_classes']}")
    print(f"Train    : {len(loaders['train'].dataset)} images")
    print(f"Val      : {len(loaders['val'].dataset)} images")
    print(f"Test     : {len(loaders['test'].dataset)} images")

    images, labels = next(iter(loaders["train"]))
    print(f"Batch shape : {images.shape}")   # → torch.Size([32, 3, 224, 224])
    print(f"Label shape : {labels.shape}")   # → torch.Size([32])
