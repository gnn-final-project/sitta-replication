import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms
from networks import FewShotGen
from tqdm import tqdm
import pandas as pd

# Config
DATA_FOLDER = "./data"
SAVE_FOLDER = "./results"
os.makedirs(SAVE_FOLDER, exist_ok=True)

DOMAINS = [
    "apple2orange",
    "horse2zebra",
    "milk2bubblemilk",
    "vanilla2chocolate"
]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator config
hparams = {
    'nf': 64,
    'nf_mlp': 256,
    'n_downs_class': 2,
    'n_downs_content': 2,
    'n_mlp_blks': 3,
    'n_res_blks': 4,
    'latent_dim': 8
}

# Image loader
def load_image(path):
    transform = transforms.Compose([
        transforms.Resize((288, 288)),
        transforms.ToTensor()
    ])
    return transform(Image.open(path).convert("RGB")).unsqueeze(0)

# Logging training times
train_times = []

# Training loop
for domain in DOMAINS:
    print(f"\n📌 Training FUNIT for domain: {domain}")
    
    content_path = os.path.join(DATA_FOLDER, f"{domain}_A.jpg")
    style_path = os.path.join(DATA_FOLDER, f"{domain}_B.jpg")
    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)

    gen = FewShotGen(hparams).to(device)
    gen.train()
    optimizer = optim.Adam(gen.parameters(), lr=1e-4)
    loss_fn = nn.L1Loss()

    steps = 1000
    start_time = time.time()

    for step in tqdm(range(steps), desc=f"🔧 Training {domain}"):
        optimizer.zero_grad()
        output = gen(content, style)
        loss_content = loss_fn(output, content)
        loss = loss_content
        loss.backward()
        optimizer.step()

    elapsed = time.time() - start_time
    train_times.append({"domain": domain, "total_seconds": elapsed, "per_iter": elapsed / steps})

    save_path = os.path.join(SAVE_FOLDER, f"{domain}_gen.pth")
    torch.save(gen.state_dict(), save_path)
    print(f"✅ Saved trained generator: {save_path}")
    print(f"⏱ Training time: {elapsed:.2f} sec")

# Save training times
df = pd.DataFrame(train_times)
df.to_csv(os.path.join(SAVE_FOLDER, "funit_time_results.csv"), index=False)
print("\n📊 Saved training time results to funit_time_results.csv")
