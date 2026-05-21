"""Dataset loading and transforms for bird classification."""

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


def get_transforms(train: bool = True, image_size: int = 224):
    """Get transforms for training or evaluation."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        ])


class BirdDataset(Dataset):
    """Generic image folder dataset for bird classification.

    Expects structure:
        root/
            class_a/
                img1.jpg
                img2.jpg
            class_b/
                img3.jpg
    """

    def __init__(self, root: str, transform=None):
        self.root = Path(root)
        self.transform = transform

        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root / cls
            for img_path in cls_dir.iterdir():
                if img_path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'}:
                    self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Create train and validation dataloaders."""
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    train_dataset = BirdDataset(train_dir, transform=get_transforms(train=True))
    val_dataset = BirdDataset(val_dir, transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset.classes
