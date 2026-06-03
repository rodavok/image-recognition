"""Model definitions for bird classification."""

import timm
import torch
import torch.nn as nn


def create_model(
    model_name: str = 'resnet50',
    num_classes: int = 200,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    """Create a model using timm library.

    Args:
        model_name: Any model from timm (resnet50, efficientnet_b0, vit_base_patch16_224, etc.)
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        freeze_backbone: If True, only train the classification head
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier head (timm models use different names)
        classifier = model.get_classifier()
        for param in classifier.parameters():
            param.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_checkpoint(model, optimizer, epoch, path, **kwargs):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load model checkpoint, returning the full checkpoint dict."""
    checkpoint = torch.load(path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint
