import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd

# Script base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map experiment mode to correct data folder
def get_data_path(mode):
    if mode == "funit":
        return os.path.join(SCRIPT_DIR, "../FUNIT/data/plant_pathology")
    elif mode == "tuigan":
        return os.path.join(SCRIPT_DIR, "../TuiGAN/data/plant_pathology")
    elif mode == "singan":
        return os.path.join(SCRIPT_DIR, "../SinGAN/data/plant_pathology")
    elif mode == "classic":
        return os.path.join(SCRIPT_DIR, "./data/plant_pathology")
    elif mode == "none":
        return os.path.join(SCRIPT_DIR, "./data/plant_pathology")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

# Command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="none", help="Options: none | classic | singan | tuigan | funit")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
base_path = get_data_path(args.mode)
data_transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor()
])

train_dataset = datasets.ImageFolder(os.path.join(base_path, 'train'), transform=data_transform)
test_dataset = datasets.ImageFolder(os.path.join(base_path, 'test'), transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Training function
def train_and_evaluate(model_name="resnet18"):
    print(f"\n📌 Training model: {model_name} ({args.mode})")

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    else:
        raise ValueError("Model must be 'resnet18' or 'vgg16'")

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

    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total * 100
    print(f"✅ Accuracy ({model_name}): {accuracy:.2f}%")
    return accuracy

# Run
resnet_acc = train_and_evaluate("resnet18")
vgg_acc = train_and_evaluate("vgg16")

# Save results
result = pd.DataFrame([{
    "Mode": args.mode,
    "ResNet18 Acc": round(resnet_acc, 2),
    "VGG16 Acc": round(vgg_acc, 2)
}])

results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

result_path = os.path.join(results_dir, f"classification_result_{args.mode}.csv")
result.to_csv(result_path, index=False)
print(f"\n📊 Saved results to: {result_path}")