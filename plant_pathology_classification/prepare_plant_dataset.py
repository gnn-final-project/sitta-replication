import os
import shutil
import random

# Set seed for reproducibility
random.seed(42)

# Input source folders
base_healthy = "../data/plant_pathology/healthy"
base_sick = "../data/plant_pathology/sick"

# Output structure
train_healthy = "./data/plant_pathology/train/healthy"
train_sick = "./data/plant_pathology/train/sick"
test_healthy = "./data/plant_pathology/test/healthy"
test_sick = "./data/plant_pathology/test/sick"

# Create folders
for folder in [train_healthy, train_sick, test_healthy, test_sick]:
    os.makedirs(folder, exist_ok=True)

# Step 1: Split healthy images into train (416) and test (100)
all_healthy = sorted(os.listdir(base_healthy))
random.shuffle(all_healthy)

train_healthy_imgs = all_healthy[:416]
test_healthy_imgs = all_healthy[416:]

for fname in train_healthy_imgs:
    shutil.copy(os.path.join(base_healthy, fname), os.path.join(train_healthy, fname))

for fname in test_healthy_imgs:
    shutil.copy(os.path.join(base_healthy, fname), os.path.join(test_healthy, fname))

print("✅ Copied 416 healthy images to train/healthy and 100 to test/healthy")

# Step 2: Randomly sample 81 sick images for test
all_sick = sorted(os.listdir(base_sick))
test_sick_imgs = random.sample(all_sick, 81)

for fname in test_sick_imgs:
    shutil.copy(os.path.join(base_sick, fname), os.path.join(test_sick, fname))

print("✅ Copied 81 sick images to test/sick")