import os
import shutil
import random
import subprocess

# Set seed for reproducibility
random.seed(42)

# === Step 1: Prepare dataset folders ===
base_healthy = "../../data/plant_pathology/healthy"
base_sick = "../../data/plant_pathology/sick"

train_healthy = "./data/plant_pathology/train/healthy"
train_sick = "./data/plant_pathology/train/sick"
test_healthy = "./data/plant_pathology/test/healthy"
test_sick = "./data/plant_pathology/test/sick"

for folder in [train_healthy, train_sick, test_healthy, test_sick]:
    os.makedirs(folder, exist_ok=True)

# === Step 2: Random split healthy images ===
all_healthy = sorted(os.listdir(base_healthy))
random.shuffle(all_healthy)
train_healthy_imgs = all_healthy[:416]
test_healthy_imgs = all_healthy[416:]

for fname in train_healthy_imgs:
    shutil.copy(os.path.join(base_healthy, fname), os.path.join(train_healthy, fname))

for fname in test_healthy_imgs:
    shutil.copy(os.path.join(base_healthy, fname), os.path.join(test_healthy, fname))

print("✅ Copied 416 healthy images to train and 100 to test.")

# === Step 3: Random select 81 sick test images ===
all_sick = sorted(os.listdir(base_sick))
test_sick_imgs = random.sample(all_sick, 81)

for fname in test_sick_imgs:
    shutil.copy(os.path.join(base_sick, fname), os.path.join(test_sick, fname))

print("✅ Copied 81 sick images to test set.")

# === Step 4: Generate fake sick images using SinGAN ===
model_dir = "./results/healthy2sick"
generated_output = train_sick
os.makedirs(generated_output, exist_ok=True)

for i, fname in enumerate(sorted(os.listdir(train_healthy))):
    input_path = os.path.join(train_healthy, fname)
    output_name = f"sick_gen_{i+1:03d}.jpg"
    output_path = os.path.join(generated_output, output_name)

    print(f"📌 Generating fake sick image: {output_name}")

    subprocess.run([
        "python", "main.py",
        "--root", input_path,
        "--evaluation",
        "--model-to-load", os.path.join(model_dir, "g_multivanilla.pt"),
        "--amps-to-load", os.path.join(model_dir, "amps.pt"),
        "--save", "temp_gen",
        "--results-dir", "./results",
        "--num-steps", "10"
    ])

    gen_img = "./results/temp_gen/s11/s9_sampled.png"
    if os.path.exists(gen_img):
        shutil.copy(gen_img, output_path)

print("✅ Finished generating 416 fake sick images using SinGAN.")