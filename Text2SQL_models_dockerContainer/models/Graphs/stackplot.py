import os
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np

# Function to load data from results file
def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)


# Load variables from .env file
load_dotenv("configuration.env")

# File paths with results
BASE_PATH = os.getenv("BASE_PATH")
RESULTS_MODEL_1 = os.getenv("RESULTS_MODEL_1")
RESULTS_MODEL_2 = os.getenv("RESULTS_MODEL_2")
NAME_OUTPUT_FILE_GRAPH = os.getenv("NAME_OUTPUT_FILE_GRAPH")
LABEL_MODEL_1 = os.getenv("LABEL_MODEL_1")
LABEL_MODEL_2 = os.getenv("LABEL_MODEL_2")

# Load data
RESULTS_MODEL_1 = load_results(BASE_PATH+"/"+RESULTS_MODEL_1)
RESULTS_MODEL_2 = load_results(BASE_PATH+"/"+RESULTS_MODEL_2)

# Prepare data for stackplot
categories = list(RESULTS_MODEL_1.keys())
correct_model_1 = [RESULTS_MODEL_1[cat]["total"] - RESULTS_MODEL_1[cat]["incorrect"] for cat in categories]
incorrect_model_1 = [RESULTS_MODEL_1[cat]["incorrect"] for cat in categories]

correct_model_2 = [RESULTS_MODEL_2[cat]["total"] - RESULTS_MODEL_2[cat]["incorrect"] for cat in categories]
incorrect_model_2 = [RESULTS_MODEL_2[cat]["incorrect"] for cat in categories]

x = np.arange(len(categories)) # category positions

# Creating the stackplot
bar_width = 0.35  # width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

# Stackplot for model 1
ax.bar(x - bar_width/2, correct_model_1, bar_width, label=LABEL_MODEL_1 + " - Correct", color="green")
ax.bar(x - bar_width/2, incorrect_model_1, bar_width, bottom=correct_model_1, label=LABEL_MODEL_1 + " - Incorrect", color="red")

# Stackplot for model 2
ax.bar(x + bar_width/2, correct_model_2, bar_width, label=LABEL_MODEL_2 + " - Correct", color="blue")
ax.bar(x + bar_width/2, incorrect_model_2, bar_width, bottom=correct_model_2, label=LABEL_MODEL_2 + " - Incorrect", color="orange")

ax.set_xlabel("Categories")
ax.set_ylabel("Number of Queries")
ax.set_title("Comparison of SQL Query Results by Model and Category")
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()

plt.tight_layout()
plt.show()

# Save the graph
path_save = BASE_PATH+"/venvText2SQL/models/Graphs/Stackplot/"+NAME_OUTPUT_FILE_GRAPH+".png"
os.makedirs(os.path.dirname(path_save), exist_ok=True)
plt.savefig(path_save, format='png')
print(f"\nGraph saved in: {NAME_OUTPUT_FILE_GRAPH}\n")
