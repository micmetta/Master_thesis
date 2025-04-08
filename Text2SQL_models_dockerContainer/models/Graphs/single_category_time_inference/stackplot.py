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
LANGUAGE = os.getenv("LANGUAGE")
if LANGUAGE not in {"en", "it"}:
    raise ValueError(f"Invalid LANGUAGE value: {LANGUAGE}. Expected 'en' or 'it'.")

# Dictionary associating models with their file paths
file_paths = {
    LABEL_MODEL_1: RESULTS_MODEL_1,
    LABEL_MODEL_2: RESULTS_MODEL_2,
    LABEL_MODEL_3: RESULTS_MODEL_3
}

# Function to extract average and variance of inference times per category
def extract_avg_var_times(file_path):
    category_times = {}
    current_category = None

    print("\n\nfile_path of the current model: ", file_path)
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith("Current category:"):
                current_category = line.split(":")[1].strip()
                category_times[current_category] = []
                print("\ncurrent_category: ", current_category)
            elif "Time taken:" in line:
                match = re.search(r"Time taken: ([\d\.]+) seconds", line)
                if match and current_category:
                    category_times[current_category].append(float(match.group(1)))
                    print("Time taken: ", float(match.group(1)))
    print("\n")
    # Compute mean and variance for each category
    stats = {cat: (np.mean(times), np.var(times)) for cat, times in category_times.items() if times}
    print("final stats: ", stats)
    return stats

# Extracting average inference times and variances for each model
stats = {model: extract_avg_var_times(path) for model, path in file_paths.items()}

# Defining categories (considering all those present in at least one model)
categories = sorted(set().union(*[stats[model].keys() for model in stats]))

# Extracting means and standard deviations for visualization
times = {model: [stats[model].get(cat, (0, 0))[0] for cat in categories] for model in stats}
print("\ntimes: ", times)
variances = {model: [stats[model].get(cat, (0, 0))[1] for cat in categories] for model in stats}
print("\nvariances: ", variances)
std_devs = {model: [np.sqrt(var) for var in variances[model]] for model in variances}
print("\nstd_devs: ", std_devs)

# Creating the plot with error bars
fig, ax = plt.subplots(figsize=(12, 7))
bar_width = 0.25
x = np.arange(len(categories))

# Define colors for models
colors = {
    LABEL_MODEL_1: "#6baed6",  # light blue
    LABEL_MODEL_2: "#74c476",  # light green
    LABEL_MODEL_3: "#fdae61",  # light orange
}

# Creating bars with error bars (std deviation)
bars = {}
for i, (model, color) in enumerate(colors.items()):
    bars[model] = ax.bar(x + (i - 1) * bar_width, times[model], bar_width, 
                          yerr=std_devs[model], capsize=5, label=model, color=color, 
                          edgecolor='black', linewidth=0.8)
    ax.errorbar(x + (i - 1) * bar_width, times[model], yerr=std_devs[model], 
                fmt='none', ecolor=color, elinewidth=2, alpha=0.7, capsize=5)

# Annotating bars with mean values
for model in bars:
    for bar in bars[model]:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.set_xlabel("Categories", fontsize=14, fontweight='bold')
ax.set_ylabel("Average Time (seconds)", fontsize=14, fontweight='bold')
if(LANGUAGE == "en"):
    ax.set_title("Average Inference Time per Category with std Deviation (Comparison of 3 Models) - English", fontsize=13, fontweight='bold', pad=10)
else:
    ax.set_title("Average Inference Time per Category with std Deviation (Comparison of 3 Models) - Italian", fontsize=13, fontweight='bold', pad=10)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, rotation=20)
ax.legend(fontsize=12, loc='upper left', bbox_to_anchor=(1, 1))
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Save the graph
path_save = os.path.join(BASE_PATH, "venvText2SQL/models/Graphs/single_category_time_inference/Stackplot", NAME_OUTPUT_FILE_GRAPH + ".png")
os.makedirs(os.path.dirname(path_save), exist_ok=True)
plt.savefig(path_save, format='png')
print(f"\n\nGraph saved in: {path_save}\n")