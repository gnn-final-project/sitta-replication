import os
import pandas as pd
import matplotlib.pyplot as plt

# Path
RESULTS_DIR = "./results"
OUT_CSV = os.path.join(RESULTS_DIR, "classification_summary.csv")
OUT_PLOT = os.path.join(RESULTS_DIR, "classification_plot.png")

# Load all classification result CSVs
csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("classification_result_")]
all_results = []

for file in csv_files:
    path = os.path.join(RESULTS_DIR, file)
    df = pd.read_csv(path)
    all_results.append(df)

# Merge all into single DataFrame
summary = pd.concat(all_results, ignore_index=True)
summary = summary.sort_values(by="ResNet18 Acc", ascending=False)

# Save merged table
summary.to_csv(OUT_CSV, index=False)
print(f"📊 Combined results saved to: {OUT_CSV}")

# Print table
print("\n📋 Classification Summary:")
print(summary.to_string(index=False))

# Plot
modes = summary["Mode"]
resnet_scores = summary["ResNet18 Acc"]
vgg_scores = summary["VGG16 Acc"]

x = range(len(modes))
bar_width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x, resnet_scores, width=bar_width, label='ResNet18', align='center')
plt.bar([i + bar_width for i in x], vgg_scores, width=bar_width, label='VGG16', align='center')

plt.xlabel("Experiment Mode")
plt.ylabel("Accuracy (%)")
plt.title("Classification Accuracy by Mode")
plt.xticks([i + bar_width / 2 for i in x], modes, rotation=15)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PLOT)
print(f"📈 Plot saved to: {OUT_PLOT}")
plt.show()