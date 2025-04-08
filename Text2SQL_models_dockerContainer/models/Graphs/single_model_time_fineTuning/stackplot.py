import os
import re
import numpy as np
import matplotlib.pyplot as plt
from dotenv import load_dotenv

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
    LABEL_MODEL_3: RESULTS_MODEL_3,
}

# Function to extract "elapsed_time" and convert it to hours and minutes
def extract_elapsed_time(file_path):
    with open(file_path, "r") as file:
        for line in file:
            match = re.search(r"elapsed_time:\s*([\d\.]+)", line)
            if match:
                elapsed_seconds = float(match.group(1))
                total_hours = elapsed_seconds // 3600
                total_minutes = (elapsed_seconds % 3600) // 60
                return total_hours, total_minutes
    return None, None

# Extract fine-tuning times for each model
elapsed_times = {model: extract_elapsed_time(path) for model, path in file_paths.items()}

fig, ax = plt.subplots(figsize=(12, 7))

models = list(elapsed_times.keys())
times = [elapsed_times[model][0] + elapsed_times[model][1] / 60 for model in models]  # convert everything to decimal hours

ax.bar(models, times, color=["#6baed6", "#74c476", "#fdae61"], edgecolor="black", linewidth=0.8)

# Adding labels above bars (format hh:mm)
for i, (hours, minutes) in enumerate(elapsed_times.values()):
    ax.text(i, times[i] + 0.01, f"{int(hours)}h {int(minutes)}m", ha='center', va='bottom', fontsize=12, fontweight='bold')

ax.set_xlabel("Models", fontsize=14, fontweight='bold')
ax.set_ylabel("Fine-tuning Time (hours)", fontsize=14, fontweight='bold')
ax.set_title(TITLE_GRAPH, fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(range(len(models)))
ax.set_xticklabels(models, fontsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Save the graph
path_save = os.path.join(BASE_PATH, "venvText2SQL/models/Graphs/single_model_time_fineTuning/Stackplot", NAME_OUTPUT_FILE_GRAPH + ".png")
os.makedirs(os.path.dirname(path_save), exist_ok=True)
plt.savefig(path_save, format='png')
print(f"\n\nGraph saved in: {path_save}\n")