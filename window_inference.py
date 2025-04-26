import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import Transformer_local

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    avg_loss = val_loss / len(val_loader)
    print(f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set up validation data transforms
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_dir = r'data\archive\tiny-imagenet-200'
    val_dir = os.path.join(data_dir, 'val')

    batch_size = 64

    # Prepare validation dataset and loader
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize model architecture
    model = Transformer_local.ViT(
        d_model=256,
        num_layers=6,
        num_heads=4,
        num_classes=200,
        patch_size=8
    ).to(device)

    # Load saved weights
    model.load_state_dict(torch.load('best_local_vit_model.pth', map_location=device))
    print("Loaded model weights from 'best_local_vit_model.pth'.")

    # Set loss function (not for training, just to compute validation loss)
    criterion = nn.CrossEntropyLoss()

    # Run validation
    print("\nRunning inference on validation set...")
    validate(model, val_loader, device, criterion)

if __name__ == '__main__':
    main()
