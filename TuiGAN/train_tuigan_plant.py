import os
import time
import random
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from image_utils import read_domains, normalize_image_to_tensor, construct_scale_pyramid, get_output_image
from model_utils import create_models, save_models, total_variation_loss, gradient_penalty
from train_utils import generate_input, generate_identity

# --- Step 1: Configuration ---
device_number = 0
DEVICE = f'cuda:{device_number}' if torch.cuda.is_available() else 'cpu'

torch.cuda.empty_cache()
gc.collect()

# Hyperparameters
LR = 0.0005
LR_DECAY_STEP = 1600
SCALE_FACTOR = 0.75
NUM_ITERS = 4000
NUM_SCALES = 5
PHI_BLOCK_COUNT = 5
PSI_BLOCK_COUNT = 4
LAMBDA_CYC = 1
LAMBDA_IDT = 1
LAMBDA_TV = 0.1
LAMBDA_PEN = 0.1
LOG_FREQ = 500
DISC_STEP = 3
GEN_STEP = 3

torch.manual_seed(1641)
random.seed(1641)
np.random.seed(1641)

# --- Step 2: Data loading ---
pathA = "healthy2sick_A.jpg"  # single healthy
pathB = "healthy2sick_B.jpg"  # single sick
data_name = "healthy2sick"

imgA, imgB = read_domains("data", image_A_file=pathA, image_B_file=pathB, resize=True)

normed_imgA = normalize_image_to_tensor(imgA).to(DEVICE)
normed_imgB = normalize_image_to_tensor(imgB).to(DEVICE)

listA = construct_scale_pyramid(normed_imgA, N=NUM_SCALES - 1)
listB = construct_scale_pyramid(normed_imgB, N=NUM_SCALES - 1)

listA.reverse()
listB.reverse()

# --- Step 3: Model Initialization ---
listg_ab, listg_ba, listd_a, listd_b = create_models(num_scale=NUM_SCALES, device=DEVICE)

# --- Step 4: Training ---
def reset_grads(model, require_grad):
    for p in model.parameters():
        p.requires_grad_(require_grad)
    return model

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

    # After training, freeze gradients and eval mode
    for model in [gen_ab, gen_ba, disc_a, disc_b]:
        model.eval()
        reset_grads(model, False)
    torch.cuda.empty_cache()

def train_all_scales(listA, listB, models):
    for i in range(NUM_SCALES):
        train_scale(listA, listB, models, i)

print("\n🚀 Training TuiGAN on single healthy-sick pair...")
start_time = time.time()
train_all_scales(listA, listB, (listg_ab, listg_ba, listd_a, listd_b))
end_time = time.time()

# --- Step 5: Save models ---
save_models(listg_ab, listg_ba, listd_a, listd_b, data_name)
print(f"✅ Training completed in {(end_time - start_time):.2f} seconds and models saved to models/{data_name}/")