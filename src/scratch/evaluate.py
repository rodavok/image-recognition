"""Evaluate trained model and run inference."""

import argparse
from pathlib import Path
import sys

import torch
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))  # src/ for data.py
sys.path.insert(0, str(Path(__file__).parent))          # src/scratch/ for model.py


from data import get_transforms, BirdDataset
from model import create_model


@torch.no_grad()
def evaluate_dataset(model, data_dir: str, device, batch_size: int = 32):
    """Evaluate model on a dataset directory."""
    from torch.utils.data import DataLoader

    dataset = BirdDataset(data_dir, transform=get_transforms(train=False))
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    model.eval()
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc='Evaluating'):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = correct / total
    print(f'Accuracy: {accuracy:.4f} ({correct}/{total})')
    return accuracy


@torch.no_grad()
def predict_image(model, image_path: str, classes: list, device, topk: int = 5):
    """Predict class for a single image."""
    transform = get_transforms(train=False)

    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)

    top_probs, top_indices = probabilities.topk(topk)

    print(f'\nPredictions for {image_path}:')
    for prob, idx in zip(top_probs[0], top_indices[0]):
        print(f'  {classes[idx]}: {prob.item():.4f}')

    return classes[top_indices[0][0].item()]


def main():
    parser = argparse.ArgumentParser(description='Evaluate bird classifier')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--data', type=str, default=None, help='Path to evaluation data directory')
    parser.add_argument('--image', type=str, default=None, help='Path to single image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, weights_only=False, map_location=device)
    classes = checkpoint['classes']
    num_classes = len(classes)
    print(f'Loaded {num_classes} classes')

    # Create and load model
    model = create_model(num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    if args.data:
        evaluate_dataset(model, args.data, device)
    elif args.image:
        predict_image(model, args.image, classes, device)
    else:
        print('Specify --data for dataset evaluation or --image for single prediction')


if __name__ == '__main__':
    main()
