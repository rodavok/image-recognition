"""Training loop for bird classification."""

import argparse
import os
from pathlib import Path

import mlflow
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from data import get_dataloaders
from model import create_model, count_parameters, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch, return average loss and accuracy."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=loss.item(), acc=100. * correct / total)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate model, return average loss and accuracy."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Validating'):
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser(description='Train bird classifier')
    parser.add_argument('--data', type=str, default='data', help='Path to data directory')
    parser.add_argument('--model', type=str, default='resnet50', help='Model architecture')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--freeze-backbone', action='store_true', help='Only train classifier head')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--mlflow', action='store_true', help='Enable MLflow tracking')
    parser.add_argument('--experiment', type=str, default='bird-classification', help='MLflow experiment name')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Data
    train_loader, val_loader, classes = get_dataloaders(
        args.data, batch_size=args.batch_size
    )
    num_classes = len(classes)
    print(f'Found {num_classes} classes, {len(train_loader.dataset)} training images')

    # Model
    model = create_model(
        args.model,
        num_classes=num_classes,
        pretrained=True,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)

    total_params, trainable_params = count_parameters(model)
    print(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = 0

    if args.resume:
        from model import load_checkpoint
        start_epoch = load_checkpoint(args.resume, model, optimizer)
        print(f'Resumed from epoch {start_epoch}')

    # Training loop
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    if args.mlflow:
        mlflow.set_experiment(args.experiment)
        mlflow.start_run()
        mlflow.log_params({
            'model': args.model,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'freeze_backbone': args.freeze_backbone,
            'num_classes': num_classes,
            'trainable_params': trainable_params,
        })

    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        if args.mlflow:
            mlflow.log_metrics({
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': scheduler.get_last_lr()[0],
            }, step=epoch)

        # Save checkpoints
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint(
            model, optimizer, epoch + 1,
            os.path.join(args.checkpoint_dir, 'latest.pt'),
            val_acc=val_acc, classes=classes,
        )

        if is_best:
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(args.checkpoint_dir, 'best.pt'),
                val_acc=val_acc, classes=classes,
            )
            print(f'New best accuracy: {val_acc:.4f}')

    if args.mlflow:
        mlflow.log_metric('best_val_acc', best_acc)
        mlflow.log_artifact(os.path.join(args.checkpoint_dir, 'best.pt'))
        mlflow.end_run()

    print(f'\nTraining complete. Best validation accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()
