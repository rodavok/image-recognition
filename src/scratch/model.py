"""CNN built from scratch for bird classification."""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        # TODO(human): define self.block as an nn.Sequential containing:
        #   1. nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        #   2. nn.BatchNorm2d(out_channels)
        #   3. nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class BirdCNN(nn.Module):
    """Simple CNN for 200-class bird classification.

    Architecture:
        Stage 1: 3   -> 32  channels, 224x224 -> 112x112  (after pool)
        Stage 2: 32  -> 64  channels, 112x112 -> 56x56
        Stage 3: 64  -> 128 channels, 56x56   -> 28x28
        Stage 4: 128 -> 256 channels, 28x28   -> 14x14
        Stage 5: 256 -> 512 channels, 14x14   -> 7x7
        Global average pool: 512 x 7x7 -> 512
        Classifier: 512 -> num_classes
    """

    def __init__(self, num_classes: int = 200):
        super().__init__()

        # TODO(human): build self.features as an nn.Sequential.
        # Each stage = ConvBlock(...) followed by nn.MaxPool2d(2, 2).
        # Use the channel sizes from the docstring above.
        # Tip: ConvBlock(3, 32) means "take a 3-channel image, output 32 channels".
        self.features = nn.Sequential(
                ConvBlock(3, 32), nn.MaxPool2d(2, 2),
                ConvBlock(32, 64), nn.MaxPool2d(2, 2),
                ConvBlock(64, 128), nn.MaxPool2d(2, 2),
                ConvBlock(128, 256), nn.MaxPool2d(2, 2),
                ConvBlock(256, 512), nn.MaxPool2d(2, 2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # TODO(human): build self.classifier as an nn.Sequential:
        #   nn.Dropout(0.5), nn.Linear(512, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def create_model(num_classes: int = 200) -> BirdCNN:
    return BirdCNN(num_classes=num_classes)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, epoch, path, **kwargs):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint
