import subprocess
import time

# List of domains to include
domains = [
    "apple2orange",
    "horse2zebra",
    "milk2bubblemilk",
    "vanilla2chocolate"
]

# Configuration
data_root = "./data"
num_steps = 2000  # actual training steps per domain
reported_steps = 20000

iteration_times = []

for domain in domains:
    input_path = f"{data_root}/{domain}_B.jpg"
    print(f"\n📌 Starting training for domain: {domain}")
    
    start_time = time.time()

    cmd = [
        "python", "main.py",
        "--root", input_path,
        "--save", domain,
        "--num-steps", str(num_steps),
        "--max-size", "288"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    end_time = time.time()
    duration = end_time - start_time
    time_per_iter = duration / num_steps
    iteration_times.append(time_per_iter)

    if result.returncode == 0:
        print(f"✅ Training completed for {domain}")
        print(f"⏱️ Time: {duration:.2f}s total, {time_per_iter:.4f}s per iteration")
    else:
        print(f"❌ Training failed for {domain}")
        print(result.stderr)

# Compute average time per iteration (based on real 2000 steps)
avg_iter_time = sum(iteration_times) / len(iteration_times)
estimated_time_20k = avg_iter_time * reported_steps

# Report
print("\n📊 Average Training Time per Iteration")
print(f"Avg time per iteration: {avg_iter_time:.4f} seconds")
print(f"Estimated time for 20,000 iterations: {estimated_time_20k / 3600:.2f} hours")
