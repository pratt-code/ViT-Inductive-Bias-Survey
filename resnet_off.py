import torch
import torch.nn as nn
from torchvision.models.vision_transformer import VisionTransformer
from tqdm import tqdm
import multiprocessing
import shutil
from torch.optim.lr_scheduler import SequentialLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import os

def validate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validation", leave=False):
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

def prepare_val_data(val_dir):
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations = os.path.join(val_dir, 'val_annotations.txt')

    val_labels = {}
    with open(val_annotations, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            val_labels[parts[0]] = parts[1]

    for img, label in val_labels.items():
        label_dir = os.path.join(val_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        shutil.move(os.path.join(val_images_dir, img), os.path.join(label_dir, img))

    shutil.rmtree(val_images_dir)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_dir = r'data\archive\tiny-imagenet-200'
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    if not os.path.exists(os.path.join(val_dir, 'n01443537')):
        prepare_val_data(val_dir)

    batch_size = 64

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Initialize ResNet18
    print("\nInitializing ResNet18...")
    model = models.resnet18(num_classes=200).to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=5e-4)

    num_epochs = 100
    best_accuracy = 0.0
    steps_per_epoch = ((len(train_dataset) - 1) // batch_size) + 1
    warmup_epochs = 10
    warmup_steps = warmup_epochs * steps_per_epoch
    cosine_t_max = steps_per_epoch * num_epochs - warmup_steps

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_t_max, eta_min=1e-6
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
    )

    print("\nStarting training...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader):.4f}, Accuracy: {accuracy:.2f}%")

        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            val_accuracy = validate(model, val_loader, device, criterion)
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                print(f"New best accuracy: {best_accuracy:.2f}% - Saving model")
                torch.save(model.state_dict(), 'best_resnet_model.pth')

        for i, param_group in enumerate(optimizer.param_groups):
            print(f"Epoch [{epoch+1}], Param group {i} learning rate = {param_group['lr']:.6f}")

    print('Training complete!')

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
