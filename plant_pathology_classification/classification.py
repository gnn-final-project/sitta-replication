import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import argparse

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="noaug", choices=["noaug", "classic"], help="Augmentation mode")
args = parser.parse_args()
mode = args.mode

# Define transforms
if mode == "noaug":
    print("🔍 Using NO augmentation")
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor()
    ])
elif mode == "classic":
    print("🔍 Using CLASSIC augmentation")
    train_transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor()
    ])

test_transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder('data/plant_pathology/train', transform=train_transform)
test_dataset = datasets.ImageFolder('data/plant_pathology/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Train and evaluate
def train_and_evaluate(model_name="resnet18"):
    print(f"\n📌 Training model: {model_name} ({mode})")

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    else:
        raise ValueError("Only 'resnet18' or 'vgg16' are supported")

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}/10 - Loss: {running_loss:.4f}")

    # Evaluate
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = 100 * correct / total
    print(f"✅ {model_name} Accuracy ({mode}): {acc:.2f}%")
    return acc

# Run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet_acc = train_and_evaluate("resnet18")
vgg_acc = train_and_evaluate("vgg16")

# Save results
df = pd.DataFrame([
    {"Model": "ResNet18", "Mode": mode, "Accuracy (%)": round(resnet_acc, 2)},
    {"Model": "VGG16", "Mode": mode, "Accuracy (%)": round(vgg_acc, 2)}
])

csv_path = f"classification_results_{mode}.csv"
df.to_csv(csv_path, index=False)

print("\n📊 Final Results")
print(df.to_string(index=False))
print(f"\n📁 Saved to: {csv_path}")