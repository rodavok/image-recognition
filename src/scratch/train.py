"""Training loop for scratch CNN bird classification."""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from data import get_dataloaders

from model import create_model, count_parameters, save_checkpoint


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

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
    model.eval()
    total_loss, correct, total = 0, 0, 0

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
    parser = argparse.ArgumentParser(description='Train scratch CNN bird classifier')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/scratch')
    parser.add_argument('--resume', type=str, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    train_loader, val_loader, classes = get_dataloaders(args.data, batch_size=args.batch_size)
    num_classes = len(classes)
    print(f'{num_classes} classes, {len(train_loader.dataset)} training images')

    # TODO(human): wire up the model, loss, optimizer, and scheduler.
    #
    # 1. model = create_model(num_classes).to(device)
    # 2. Print total and trainable parameter counts using count_parameters()
    # 3. criterion = nn.CrossEntropyLoss()
    # 4. optimizer = AdamW(model.parameters(), lr=args.lr)
    #    Note: unlike Module 1 we pass ALL parameters — why?
    # 5. scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    total, trainable = count_parameters(model)
    print(f'Parameters: {total:,} total, {trainable:,} trainable')

    model = create_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr) #because we're training from scratch, not fine-tuning
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    start_epoch = 0
    best_acc = 0

    if args.resume:
        from model import load_checkpoint
        checkpoint = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch = checkpoint.get('epoch', 0)
        best_acc = checkpoint.get('best_acc', 0)
        print(f'Resumed from epoch {start_epoch}')

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        print(f'\nEpoch {epoch + 1}/{args.epochs}')

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        save_checkpoint(
            model, optimizer, epoch + 1,
            os.path.join(args.checkpoint_dir, 'latest.pt'),
            val_acc=val_acc, best_acc=best_acc, classes=classes,
            scheduler_state_dict=scheduler.state_dict(),
        )

        if is_best:
            save_checkpoint(
                model, optimizer, epoch + 1,
                os.path.join(args.checkpoint_dir, 'best.pt'),
                val_acc=val_acc, best_acc=best_acc, classes=classes,
                scheduler_state_dict=scheduler.state_dict(),
            )
            print(f'New best: {val_acc:.4f}')

    print(f'\nDone. Best val accuracy: {best_acc:.4f}')


if __name__ == '__main__':
    main()
