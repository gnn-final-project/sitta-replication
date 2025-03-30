import subprocess
import time
import pandas as pd
import os

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
log_rows = []

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

    log_rows.append({
        "Domain": domain,
        "Total Time (sec)": round(duration, 2),
        "Time per Step (sec)": round(time_per_iter, 4)
    })

    if result.returncode == 0:
        print(f"✅ Training completed for {domain}")
        print(f"⏱️ Time: {duration:.2f}s total, {time_per_iter:.4f}s per iteration")
    else:
        print(f"❌ Training failed for {domain}")
        print(result.stderr)

# Compute average time
avg_row = {
    "Domain": "Average",
    "Total Time (sec)": round(sum(r["Total Time (sec)"] for r in log_rows) / len(log_rows), 2),
    "Time per Step (sec)": round(sum(r["Time per Step (sec)"] for r in log_rows) / len(log_rows), 4)
}
log_rows.append(avg_row)

# Save to CSV
os.makedirs("./results", exist_ok=True)
df = pd.DataFrame(log_rows)
df.to_csv("./results/singan_time_results.csv", index=False)

# Report
print("\n📊 Average Training Time per Iteration")
print(f"Avg time per iteration: {avg_row['Time per Step (sec)']} seconds")
print(f"Estimated time for 20,000 iterations: {avg_row['Time per Step (sec)'] * reported_steps / 3600:.2f} hours")
print("✅ Saved training time log to ./results/singan_time_results.csv")