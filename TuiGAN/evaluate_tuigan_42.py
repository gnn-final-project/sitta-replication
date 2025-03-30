import os
import torch
import lpips
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import subprocess
import re
import shutil
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
lpips_model = lpips.LPIPS(net='vgg').eval().to(device)
vgg_model = models.vgg19(pretrained=True).features[:8].eval().to(device)
loss_fn = torch.nn.MSELoss()

# Image preprocessing
def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# FID
def calculate_fid(real_path, fake_path):
    real_dir = "./tmp/real"
    fake_dir = "./tmp/fake"
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)

    for f in os.listdir(real_dir): os.remove(os.path.join(real_dir, f))
    for f in os.listdir(fake_dir): os.remove(os.path.join(fake_dir, f))

    shutil.copy(real_path, os.path.join(real_dir, "real.png"))
    shutil.copy(fake_path, os.path.join(fake_dir, "fake.png"))

    cmd = f"python -m pytorch_fid {real_dir} {fake_dir}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    match = re.search(r'FID:\s*([\d.]+)', result.stdout)
    return float(match.group(1)) if match else "SKIPPED"

# LPIPS
def calculate_lpips(real_path, fake_path):
    img1 = load_image(real_path)
    img2 = load_image(fake_path)
    with torch.no_grad():
        return lpips_model(img1, img2).item()

# VGG
def calculate_vgg_loss(real_path, fake_path):
    img1 = load_image(real_path)
    img2 = load_image(fake_path)
    with torch.no_grad():
        f1 = vgg_model(img1)
        f2 = vgg_model(img2)
        return loss_fn(f1, f2).item()

# Domains to evaluate
DOMAINS = ["apple2orange", "horse2zebra", "milk2bubblemilk", "vanilla2chocolate"]

# Evaluation loop
results = []
for domain in DOMAINS:
    print(f"\n📌 Evaluating domain: {domain}")

    real_B = f"./data/{domain}_B.jpg"
    fake_B = f"./results/{domain}/{domain}_translated_ab.jpg"

    fid = calculate_fid(real_B, fake_B)
    lpips_score = calculate_lpips(real_B, fake_B)
    vgg_loss = calculate_vgg_loss(real_B, fake_B)

    results.append({
        "Domain": domain,
        "FID ↓": round(fid, 2) if isinstance(fid, float) else fid,
        "LPIPS ↓": round(lpips_score, 4),
        "VGG Loss ↓": round(vgg_loss, 4)
    })

# Average row
df = pd.DataFrame(results)
avg_row = {
    "Domain": "Average",
    "FID ↓": round(df["FID ↓"].replace("SKIPPED", pd.NA).dropna().astype(float).mean(), 4),
    "LPIPS ↓": round(df["LPIPS ↓"].mean(), 4),
    "VGG Loss ↓": round(df["VGG Loss ↓"].mean(), 4)
}
df.loc[len(df)] = avg_row

# Save
os.makedirs("./results", exist_ok=True)
df.to_csv("./results/tuigan_eval_results.csv", index=False)
print("\n📊 Evaluation results saved to ./results/tuigan_eval_results.csv")