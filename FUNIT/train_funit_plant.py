import os, torch, torch.nn as nn, torch.optim as optim
from PIL import Image
from torchvision import transforms
from networks import FewShotGen
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hparams = {
    'nf': 64, 'nf_mlp': 256, 'n_downs_class': 2, 'n_downs_content': 2,
    'n_mlp_blks': 3, 'n_res_blks': 4, 'latent_dim': 8
}

def load_image(path):
    return transforms.Compose([
        transforms.Resize((288, 288)), transforms.ToTensor()
    ])(Image.open(path).convert("RGB")).unsqueeze(0)

A_path = "./data/healthy2sick_A.jpg"
B_path = "./data/healthy2sick_B.jpg"

content = load_image(A_path).to(device)
style = load_image(B_path).to(device)

gen = FewShotGen(hparams).to(device)
optimizer = optim.Adam(gen.parameters(), lr=1e-4)
loss_fn = nn.L1Loss()

for step in tqdm(range(1000), desc="🧪 Training FUNIT"):
    optimizer.zero_grad()
    output = gen(content, style)
    loss = loss_fn(output, content)
    loss.backward()
    optimizer.step()

os.makedirs("./results", exist_ok=True)
torch.save(gen.state_dict(), "./results/healthy2sick_gen.pth")
print("✅ Saved generator.")