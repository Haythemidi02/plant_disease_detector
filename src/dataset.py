import os
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


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
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
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
                if img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

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
    import torch

    # Build the full dataset with train transforms first (split later)
    full_dataset = PlantDataset(root_dir, split="train")
    n = len(full_dataset)

    n_test  = int(n * test_split)
    n_val   = int(n * val_split)
    n_train = n - n_val - n_test

    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [n_train, n_val, n_test], generator=generator
    )

    # Override transforms for val/test subsets
    val_ds.dataset  = _SubsetWithTransform(full_dataset, val_ds.indices,  "val")
    test_ds.dataset = _SubsetWithTransform(full_dataset, test_ds.indices, "test")

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
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