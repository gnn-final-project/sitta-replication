import os
import torch
from PIL import Image
from torchvision import transforms
from networks import FewShotGen
from tqdm import tqdm

# Paths
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

# Generator config (same as during training)
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

# Image saver
def save_image(tensor, path):
    img = tensor.squeeze(0).detach().cpu()
    img = transforms.ToPILImage()(img)
    img.save(path)

# Inference
for domain in tqdm(DOMAINS, desc="🖼 Translating with FUNIT"):
    # Load images
    content_path = os.path.join(DATA_FOLDER, f"{domain}_A.jpg")
    style_path = os.path.join(DATA_FOLDER, f"{domain}_B.jpg")
    content = load_image(content_path).to(device)
    style = load_image(style_path).to(device)

    # Load trained generator
    gen = FewShotGen(hparams).to(device)
    gen.load_state_dict(torch.load(os.path.join(SAVE_FOLDER, f"{domain}_gen.pth")))
    gen.eval()

    # Translate
    with torch.no_grad():
        output = gen(content, style)

    # Save output image
    save_path = os.path.join(SAVE_FOLDER, f"{domain}_translated.jpg")
    save_image(output, save_path)
    print(f"✅ Saved: {save_path}")
