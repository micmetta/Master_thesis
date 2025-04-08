import os
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pandas as pd

load_dotenv("configuration.env")

BASE_PATH = os.getenv("BASE_PATH")
RESULTS_MODEL_1 = os.path.join(BASE_PATH, os.getenv("RESULTS_MODEL_1"))
RESULTS_MODEL_2 = os.path.join(BASE_PATH, os.getenv("RESULTS_MODEL_2"))
RESULTS_MODEL_3 = os.path.join(BASE_PATH, os.getenv("RESULTS_MODEL_3"))
NAME_OUTPUT_FILE_GRAPH = os.getenv("NAME_OUTPUT_FILE_GRAPH")

LABEL_MODEL_1 = os.getenv("LABEL_MODEL_1")
LABEL_MODEL_2 = os.getenv("LABEL_MODEL_2")
LABEL_MODEL_3 = os.getenv("LABEL_MODEL_3")

TITLE_GRAPH = os.getenv("TITLE_GRAPH")

file_paths = {
    LABEL_MODEL_1: RESULTS_MODEL_1,
    LABEL_MODEL_2: RESULTS_MODEL_2,
    LABEL_MODEL_3: RESULTS_MODEL_3
}

# Function to calculate the mean and standard deviation of memory allocation
def extract_memory_stats(file_path):
    data = pd.read_csv(file_path, skiprows=2, skipinitialspace=True)  # the first 2 lines are skipped
    allocated_memory = data["allocated_memory_GB"]
    mean_memory = allocated_memory.mean()
    std_memory = allocated_memory.std()
    return mean_memory, std_memory

# Extract memory data for each model
memory_stats = {model: extract_memory_stats(path) for model, path in file_paths.items()}

# Creating the plot with error bars
fig, ax = plt.subplots(figsize=(16, 11))

models = list(memory_stats.keys())
means = [memory_stats[model][0] for model in models]
std_devs = [memory_stats[model][1] for model in models]

colors = ["#6baed6", "#74c476", "#fdae61"]

# Creating bars without error bars
bars = ax.bar(models, means, color=colors, edgecolor="black", linewidth=0.8)

# Added labels with mean and standard deviation directly above the bars
for bar, mean, std in zip(bars, means, std_devs):
    ax.text(bar.get_x() + bar.get_width()/2, mean + 0.3, 
            f"{mean:.2f} GB\n(±{std:.2f}) GB", 
            ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel("Models", fontsize=14, fontweight='bold')
ax.set_ylabel("Allocated Memory (GB)", fontsize=14, fontweight='bold')
ax.set_title(TITLE_GRAPH, fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Save the graph
path_save = os.path.join(BASE_PATH, "venvText2SQL/models/Graphs/single_model_allocated_memory_fineTuning/Stackplot", NAME_OUTPUT_FILE_GRAPH + ".png")
os.makedirs(os.path.dirname(path_save), exist_ok=True)
plt.savefig(path_save, format='png')
print(f"\n\nGraph saved in: {path_save}\n")