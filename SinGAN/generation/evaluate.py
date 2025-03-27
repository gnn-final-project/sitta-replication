import torch
import lpips
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os
import subprocess
import re
import shutil
import pandas as pd

# ✅ 디바이스 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로드
lpips_model = lpips.LPIPS(net='vgg').eval().to(device)
vgg_model = models.vgg19(pretrained=True).features[:8].eval().to(device)
loss_fn = torch.nn.MSELoss()

# ✅ 이미지 전처리
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).to(device)

# ✅ FID 계산 함수
def calculate_fid(real_image, generated_image):
    real_dir = "./fid_temp/real"
    gen_dir = "./fid_temp/generated"
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(gen_dir, exist_ok=True)
    for f in os.listdir(real_dir): os.remove(os.path.join(real_dir, f))
    for f in os.listdir(gen_dir): os.remove(os.path.join(gen_dir, f))
    shutil.copy(real_image, os.path.join(real_dir, "real.png"))
    shutil.copy(generated_image, os.path.join(gen_dir, "gen.png"))

    command = f"python -m pytorch_fid {real_dir} {gen_dir}"
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    output = result.stdout.strip()
    match = re.search(r'FID:\s*([\d.]+)', output)
    if match:
        return float(match.group(1))
    else:
        print("⚠️ FID parse failed:\n", output)
        return None

# ✅ LPIPS 계산
def calculate_lpips(real_image, generated_image):
    img1 = load_image(real_image)
    img2 = load_image(generated_image)
    with torch.no_grad():
        return lpips_model(img1, img2).item()

# ✅ VGG perceptual loss 계산
def calculate_vgg_loss(real_image, generated_image):
    img1 = load_image(real_image)
    img2 = load_image(generated_image)
    with torch.no_grad():
        f1 = vgg_model(img1)
        f2 = vgg_model(img2)
        return loss_fn(f1, f2).item()

# ✅ 도메인 정의
domains = ["apple2orange", "horse2zebra", "milk2bubblemilk", "vanilla2chocolate"]

# ✅ 평가 실행
results = []
for domain in domains:
    print(f"\n📌 Evaluating {domain}...")
    real_img = f"../../data/processed/singan/{domain}_B.jpg"
    gen_img = f"./results/{domain}/s11/s9_sampled.png"

    fid = calculate_fid(real_img, gen_img)
    lpips_score = calculate_lpips(real_img, gen_img)
    vgg_loss = calculate_vgg_loss(real_img, gen_img)

    results.append({
        "Domain": domain,
        "FID ↓": round(fid, 2) if fid is not None else "N/A",
        "LPIPS ↓": round(lpips_score, 4),
        "VGG Loss ↓": round(vgg_loss, 4)
    })

# ✅ 평균 추가
df = pd.DataFrame(results)
avg_row = {
    "Domain": "Average",
    "FID ↓": round(df["FID ↓"].replace("N/A", pd.NA).dropna().astype(float).mean(), 2) if "N/A" not in df["FID ↓"].values else "N/A",
    "LPIPS ↓": round(df["LPIPS ↓"].mean(), 4),
    "VGG Loss ↓": round(df["VGG Loss ↓"].mean(), 4)
}
df.loc[len(df)] = avg_row

# ✅ 결과 출력
print("\n📊 Evaluation Results (SITTA Table 1 style)")
print(df.to_string(index=False))