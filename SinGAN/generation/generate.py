import os
import subprocess
import shutil
import time

# Domains to generate
domains = ["apple2orange", "horse2zebra", "milk2bubblemilk", "vanilla2chocolate"]

# Configuration
data_dir = "./data"
results_dir = "./results"

generation_times = []

for domain in domains:
    print(f"\n📌 Generating translated image for domain: {domain}")

    a_image = os.path.join(data_dir, f"{domain}_A.jpg")
    model_folder = os.path.join(results_dir, domain)
    model_g = os.path.join(model_folder, "g_multivanilla.pt")
    model_amps = os.path.join(model_folder, "amps.pt")

    # Start timing
    start_time = time.time()

    # Run SinGAN in evaluation mode
    subprocess.run([
        "python", "main.py",
        "--root", a_image,
        "--evaluation",
        "--model-to-load", model_g,
        "--amps-to-load", model_amps,
        "--save", domain,
        "--results-dir", results_dir,
        "--num-steps", "10"
    ])

    end_time = time.time()
    duration = end_time - start_time
    generation_times.append(duration)

    gen_img = os.path.join(results_dir, domain, "s11", "s9_sampled.png")
    if os.path.exists(gen_img):
        print(f"✅ Saved: {gen_img} | ⏱️ Time: {duration:.2f}s")
    else:
        print(f"❌ Generation failed for {domain}")

# Average time
avg_time = sum(generation_times) / len(generation_times)
print("\n📊 Generation Summary")
for d, t in zip(domains, generation_times):
    print(f"{d:<20} : {t:.2f} seconds")
print(f"\n⏱️ Average generation time: {avg_time:.2f} seconds")
