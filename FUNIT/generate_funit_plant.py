import os
import torch
import random
import shutil
from PIL import Image
from torchvision import transforms
from networks import FewShotGen
from tqdm import tqdm

# Paths
source_healthy = "../data/plant_pathology/healthy"
source_sick = "../data/plant_pathology/sick"
target_root = "./data/plant_pathology"
model_path = "./results_funit/healthy2sick_gen.pth"
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

# Copy 81 sick images to test
sick_imgs = sorted(os.listdir(source_sick))
random.seed(42)
sick_sample = random.sample(sick_imgs, 81)
for img in sick_sample:
    shutil.copy(os.path.join(source_sick, img), os.path.join(test_s, img))

# Load trained FUNIT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hparams = {
    'nf': 64, 'nf_mlp': 256, 'n_downs_class': 2, 'n_downs_content': 2,
    'n_mlp_blks': 3, 'n_res_blks': 4, 'latent_dim': 8
}
gen = FewShotGen(hparams).to(device)
gen.load_state_dict(torch.load(model_path))
gen.eval()

# Load reference style image
style = transforms.ToTensor()(Image.open("./data/healthy2sick_B.jpg").convert("RGB").resize((288, 288)))
style = style.unsqueeze(0).to(device)

# Transform and save
transform = transforms.Compose([
    transforms.Resize((288, 288)),
    transforms.ToTensor()
])

train_imgs = sorted(os.listdir(train_h))
for fname in tqdm(train_imgs, desc="🧬 Generating FUNIT sick images"):
    img = transform(Image.open(os.path.join(train_h, fname)).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        output = gen(img, style)
    out_img = transforms.ToPILImage()(output.squeeze().cpu())
    out_img.save(os.path.join(train_s, fname))

print("✅ FUNIT-based data augmentation complete!")