import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
from pathlib import Path
from image_utils import read_domains, normalize_image_to_tensor, construct_scale_pyramid
from model_utils import create_models, save_models, total_variation_loss, gradient_penalty
from train_utils import generate_input, generate_identity

# Device and hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LR = 0.0005
LR_DECAY_STEP = 1600
NUM_ITERS = 4000
NUM_SCALES = 5
LAMBDA_CYC = 1
LAMBDA_IDT = 1
LAMBDA_TV = 0.1
LAMBDA_PEN = 0.1
DISC_STEP = 3
GEN_STEP = 3
LOG_FREQ = 500

torch.manual_seed(1641)
random.seed(1641)
np.random.seed(1641)

DOMAINS = ["apple2orange", "horse2zebra", "milk2bubblemilk", "vanilla2chocolate"]

train_times = []

def train_scale(listA, listB, models, model_idx):
    inpA = listA[model_idx]
    inpB = listB[model_idx]

    gen_ab = models[0][model_idx]
    gen_ba = models[1][model_idx]
    disc_a = models[2][model_idx]
    disc_b = models[3][model_idx]

    gen_ab.train()
    gen_ba.train()
    disc_a.train()
    disc_b.train()

    opt_g = optim.Adam(gen_ab.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_g2 = optim.Adam(gen_ba.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d = optim.Adam(disc_a.parameters(), lr=LR, betas=(0.5, 0.999))
    opt_d2 = optim.Adam(disc_b.parameters(), lr=LR, betas=(0.5, 0.999))

    sched_d = optim.lr_scheduler.MultiStepLR(opt_d, milestones=[LR_DECAY_STEP])
    sched_d2 = optim.lr_scheduler.MultiStepLR(opt_d2, milestones=[LR_DECAY_STEP])
    sched_g = optim.lr_scheduler.MultiStepLR(opt_g, milestones=[LR_DECAY_STEP])
    sched_g2 = optim.lr_scheduler.MultiStepLR(opt_g2, milestones=[LR_DECAY_STEP])

    loss_fn = nn.L1Loss()

    for i in range(NUM_ITERS + 1):
        gc.collect()
        torch.cuda.empty_cache()

        for _ in range(DISC_STEP):
            opt_d.zero_grad()
            opt_d2.zero_grad()

            _, _, _, _, curr_ab, curr_ba = generate_input((listA, listB), models, model_idx, device=DEVICE)

            out_A = -disc_a(inpA).mean()
            out_B = -disc_b(inpB).mean()
            out_gB = disc_b(curr_ab).mean()
            out_gA = disc_a(curr_ba).mean()

            gp_a = LAMBDA_PEN * gradient_penalty(disc_a, inpA, curr_ba, device=DEVICE)
            gp_b = LAMBDA_PEN * gradient_penalty(disc_b, inpB, curr_ab, device=DEVICE)

            total_d_loss = out_A + out_B + out_gA + out_gB + gp_a + gp_b
            total_d_loss.backward()
            opt_d.step()
            opt_d2.step()

        for _ in range(GEN_STEP):
            opt_g.zero_grad()
            opt_g2.zero_grad()

            _, _, _, _, curr_ab, curr_ba = generate_input((listA, listB), models, model_idx, device=DEVICE)

            adv_loss_ab = -disc_b(curr_ab).mean()
            adv_loss_ba = -disc_a(curr_ba).mean()

            curr_aba = gen_ba(curr_ab, inpA)
            curr_bab = gen_ab(curr_ba, inpB)

            cyc_aba = LAMBDA_CYC * loss_fn(inpA, curr_aba)
            cyc_bab = LAMBDA_CYC * loss_fn(inpB, curr_bab)

            idt_aa = LAMBDA_IDT * loss_fn(inpA, generate_identity(listA, models[1], model_idx, device=DEVICE))
            idt_bb = LAMBDA_IDT * loss_fn(inpB, generate_identity(listB, models[0], model_idx, device=DEVICE))

            tv_ab = LAMBDA_TV * total_variation_loss(curr_ab)
            tv_ba = LAMBDA_TV * total_variation_loss(curr_ba)

            total_g_loss = adv_loss_ab + adv_loss_ba + cyc_aba + cyc_bab + idt_aa + idt_bb + tv_ab + tv_ba
            total_g_loss.backward()
            opt_g.step()
            opt_g2.step()

        sched_d.step()
        sched_d2.step()
        sched_g.step()
        sched_g2.step()

        if i % LOG_FREQ == 0:
            print(f"[Scale {model_idx} | Iter {i}] D Loss: {total_d_loss.item():.4f} | G Loss: {total_g_loss.item():.4f}")

    for model in [gen_ab, gen_ba, disc_a, disc_b]:
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

def train_domain(domain):
    print(f"\n🚀 Training domain: {domain}")
    image_A = f"./data/{domain}_A.jpg"
    image_B = f"./data/{domain}_B.jpg"

    imgA, imgB = read_domains("data", image_A_file=image_A, image_B_file=image_B, resize=True)

    normed_imgA = normalize_image_to_tensor(imgA).to(DEVICE)
    normed_imgB = normalize_image_to_tensor(imgB).to(DEVICE)

    listA = construct_scale_pyramid(normed_imgA, N=NUM_SCALES - 1)
    listB = construct_scale_pyramid(normed_imgB, N=NUM_SCALES - 1)
    listA.reverse()
    listB.reverse()

    models = create_models(num_scale=NUM_SCALES, device=DEVICE)

    start = time.time()
    for scale_idx in range(NUM_SCALES):
        train_scale(listA, listB, models, scale_idx)
    end = time.time()
    elapsed = end - start

    save_models(*models, name=domain)
    print(f"✅ Saved models to models/{domain}/ | Time: {elapsed:.2f}s")

    train_times.append({
        "Domain": domain,
        "Total Time (sec)": round(elapsed, 2),
        "Time per Scale": round(elapsed / NUM_SCALES, 2)
    })

# Train all domains
for domain in DOMAINS:
    train_domain(domain)

# Save training times
df = pd.DataFrame(train_times)
avg_row = {
    "Domain": "Average",
    "Total Time (sec)": round(df["Total Time (sec)"].mean(), 2),
    "Time per Scale": round(df["Time per Scale"].mean(), 2)
}
df.loc[len(df)] = avg_row

os.makedirs("./results", exist_ok=True)
df.to_csv("./results/train_time_results.csv", index=False)
print("\n📊 Training time saved to ./results/train_time_results.csv")