import subprocess

domain = "healthy2sick"

# Common configuration
data_root = "./data"
num_steps = 2000

# Loop over all domains and train SinGAN
input_path = f"{data_root}/{domain}_B.jpg"
print(f"\nStarting training for domain: {domain}")

cmd = [
    "python", "main.py",
    "--root", input_path,
    "--save", domain,
    "--num-steps", str(num_steps),
    "--max-size", "288"
]

# Run the training command
result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print(f"Training completed for {domain}")
else:
    print(f"Training failed for {domain}")
    print(result.stderr)