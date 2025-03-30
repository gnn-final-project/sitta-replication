import os
import torch
import random
import shutil
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from image_utils import normalize_image_to_tensor, construct_scale_pyramid
from train_utils import generate_outputs
from model_utils import load_models

# Constants
NUM_SCALES = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
source_healthy = "../data/plant_pathology/healthy"
source_sick = "../data/plant_pathology/sick"
target_root = "./data/plant_pathology"
os.makedirs(target_root, exist_ok=True)

# Subfolders
train_h = os.path.join(target_root, "train/healthy")
train_s = os.path.join(target_root, "train/sick")
test_h = os.path.join(target_root, "test/healthy")
test_s = os.path.join(target_root, "test/sick")
for path in [train_h, train_s, test_h, test_s]:
    os.makedirs(path, exist_ok=True)

# Random split (seed 42)
healthy_imgs = sorted(os.listdir(source_healthy))
random.seed(42)
random.shuffle(healthy_imgs)
train_imgs = healthy_imgs[:416]
test_imgs = healthy_imgs[416:]

# Copy healthy images
for img in train_imgs:
    shutil.copy(os.path.join(source_healthy, img), os.path.join(train_h, img))
for img in test_imgs:
    shutil.copy(os.path.join(source_healthy, img), os.path.join(test_h, img))

# Copy 1 sick to train and 81 to test
sick_imgs = sorted(os.listdir(source_sick))
random.seed(42)
test_sick = random.sample(sick_imgs, 81)
train_sick_real = list(set(sick_imgs) - set(test_sick))
train_sick_sample = random.choice(train_sick_real)
shutil.copy(os.path.join(source_sick, train_sick_sample), os.path.join(train_s, train_sick_sample))
for img in test_sick:
    shutil.copy(os.path.join(source_sick, img), os.path.join(test_s, img))

print("✅ Dataset folders ready")

# Load trained TuiGAN models
print("\n🎨 Generating synthetic sick images using TuiGAN...")
listg_ab, listg_ba, listd_a, listd_b = load_models("healthy2sick", device=DEVICE, NUM_SCALES=NUM_SCALES)

# Generate synthetic sick images
train_imgs = sorted(os.listdir(train_h))
for idx, fname in enumerate(tqdm(train_imgs, desc="🧬 Generating TuiGAN sick images")):
    path = os.path.join(train_h, fname)
    img = Image.open(path).convert("RGB")

    img_tensor = normalize_image_to_tensor(np.array(img)).to(DEVICE)
    listA = construct_scale_pyramid(img_tensor, N=NUM_SCALES - 1)
    listA.reverse()  # coarse → fine

    # A→B
    fake_img, _ = generate_outputs((listA, listA), (listg_ab, listg_ba, listd_a, listd_b), NUM_SCALES - 1, device=DEVICE, NUM_SCALES=NUM_SCALES)

    save_name = f"sick_gen_{idx+1:03d}.jpg"
    fake_img.save(os.path.join(train_s, save_name))

print("✅ TuiGAN-based data augmentation complete!")