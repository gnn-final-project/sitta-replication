import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pandas as pd
import random
import shutil
from PIL import Image

# Base directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Map mode to data folder
def get_data_path(mode):
    if mode == "funit":
        return os.path.join(SCRIPT_DIR, "../FUNIT/data/plant_pathology")
    elif mode == "tuigan":
        return os.path.join(SCRIPT_DIR, "../TuiGAN/data/plant_pathology")
    elif mode == "singan":
        return os.path.join(SCRIPT_DIR, "../SinGAN/data/plant_pathology")
    elif mode in ["classic", "none"]:
        return os.path.join(SCRIPT_DIR, "./data/plant_pathology")
    else:
        raise ValueError(f"Unsupported mode: {mode}")

# Parse args
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="none", help="Options: none | classic | singan | tuigan | funit")
args = parser.parse_args()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
base_path = get_data_path(args.mode)
train_healthy = os.path.join(base_path, "train", "healthy")
train_sick = os.path.join(base_path, "train", "sick")
test_sick = os.path.join(base_path, "test", "sick")
original_sick = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/plant_pathology/sick"))

# Always ensure 1 real sick image in train/sick
os.makedirs(train_sick, exist_ok=True)
os.makedirs(test_sick, exist_ok=True)
all_sick = sorted(os.listdir(original_sick))
test_sick_imgs = sorted(os.listdir(test_sick))
random.seed(42)
remaining_sick = list(set(all_sick) - set(test_sick_imgs))
if remaining_sick:
    train_sick_sample = random.choice(remaining_sick)
    target_path = os.path.join(train_sick, train_sick_sample)
    if not os.path.exists(target_path):
        shutil.copy(os.path.join(original_sick, train_sick_sample), target_path)
        print(f"✅ Copied 1 real sick image to train/sick: {train_sick_sample}")

# Classic mode: apply SITTA-style augmentations to healthy images
if args.mode == "classic":
    print("🧬 Generating classic augmentations...")

    # Clear previous augmentations (keep real sick)
    for f in os.listdir(train_sick):
        if not f.startswith("classic_sick") and not f == train_sick_sample:
            continue  # real sick image - keep it
        path = os.path.join(train_sick, f)
        if os.path.isfile(path):
            os.remove(path)

    healthy_images = sorted(os.listdir(train_healthy))
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomResizedCrop(size=(288, 288), scale=(0.8, 1.0)),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.3)
    ])

    for idx, fname in enumerate(healthy_images):
        src = os.path.join(train_healthy, fname)
        image = Image.open(src).convert("RGB")
        augmented = transform_aug(image)
        save_name = f"sick_gen_{idx+1:03d}.jpg"
        augmented.save(os.path.join(train_sick, save_name))

    print(f"✅ Generated {len(healthy_images)} synthetic sick images using classic augmentations.")

# Load data
data_transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor()
])
train_dataset = datasets.ImageFolder(os.path.join(base_path, 'train'), transform=data_transform)
test_dataset = datasets.ImageFolder(os.path.join(base_path, 'test'), transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Train + Evaluate
def train_and_evaluate(model_name="resnet18"):
    print(f"\n📌 Training model: {model_name} ({args.mode})")

    if model_name == "resnet18":
        model = models.resnet18(pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name == "vgg16":
        model = models.vgg16(pretrained=True)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 2)
    else:
        raise ValueError("Unsupported model")

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

    acc = correct / total * 100
    print(f"✅ Accuracy ({model_name}): {acc:.2f}%")
    return acc

# Run
resnet_acc = train_and_evaluate("resnet18")
vgg_acc = train_and_evaluate("vgg16")

# Save result
results_dir = os.path.join(SCRIPT_DIR, "results")
os.makedirs(results_dir, exist_ok=True)
df = pd.DataFrame([{
    "Mode": args.mode,
    "ResNet18 Acc": round(resnet_acc, 2),
    "VGG16 Acc": round(vgg_acc, 2)
}])
df.to_csv(os.path.join(results_dir, f"classification_result_{args.mode}.csv"), index=False)
print(f"\n📊 Saved results to: classification_result_{args.mode}.csv")