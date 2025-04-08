import os
import json
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerPatch

def load_results(filepath):
    with open(filepath, "r") as f:
        return json.load(f)
    
# Class to correctly handle circles in the legend
class HandlerCircle(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        center = (width / 2, height / 2)
        radius = 2  # Riduzione del cerchio
        p = mpatches.Circle(center, radius, facecolor=orig_handle.get_facecolor(),
                            edgecolor=orig_handle.get_edgecolor(), transform=trans)
        return [p]

# Class to correctly handle rectangles in the legend
class HandlerRectangle(HandlerPatch):
    def create_artists(self, legend, orig_handle, xdescent, ydescent,
                       width, height, fontsize, trans):
        rect = mpatches.Rectangle((xdescent, ydescent), width * 2, height,  # increase width
                                  facecolor=orig_handle.get_facecolor(),
                                  edgecolor=orig_handle.get_edgecolor(),
                                  linewidth=orig_handle.get_linewidth(), transform=trans)
        return [rect]

load_dotenv("configuration.env")

# File paths with results
BASE_PATH = os.getenv("BASE_PATH")
RESULTS_MODEL_1 = os.getenv("RESULTS_MODEL_1")
RESULTS_MODEL_2 = os.getenv("RESULTS_MODEL_2")
RESULTS_MODEL_3 = os.getenv("RESULTS_MODEL_3")
NAME_OUTPUT_FILE_GRAPH = os.getenv("NAME_OUTPUT_FILE_GRAPH")
LABEL_MODEL_1 = os.getenv("LABEL_MODEL_1")
LABEL_MODEL_2 = os.getenv("LABEL_MODEL_2")
LABEL_MODEL_3 = os.getenv("LABEL_MODEL_3")
LANGUAGE = os.getenv("LANGUAGE")
if LANGUAGE not in {"en", "it"}:
    raise ValueError(f"Invalid LANGUAGE value: {LANGUAGE}. Expected 'en' or 'it'.")

# Load data
RESULTS_MODEL_1 = load_results(BASE_PATH+"/"+RESULTS_MODEL_1)
RESULTS_MODEL_2 = load_results(BASE_PATH+"/"+RESULTS_MODEL_2)
RESULTS_MODEL_3 = load_results(BASE_PATH+"/"+RESULTS_MODEL_3)

# Prepare data for stackplot
categories = list(RESULTS_MODEL_1.keys())
correct_model_1 = [RESULTS_MODEL_1[cat]["total"] - RESULTS_MODEL_1[cat]["incorrect"] for cat in categories]
incorrect_model_1 = [RESULTS_MODEL_1[cat]["incorrect"] for cat in categories]

correct_model_2 = [RESULTS_MODEL_2[cat]["total"] - RESULTS_MODEL_2[cat]["incorrect"] for cat in categories]
incorrect_model_2 = [RESULTS_MODEL_2[cat]["incorrect"] for cat in categories]

correct_model_3 = [RESULTS_MODEL_3[cat]["total"] - RESULTS_MODEL_3[cat]["incorrect"] for cat in categories]
incorrect_model_3 = [RESULTS_MODEL_3[cat]["incorrect"] for cat in categories]

x = np.arange(len(categories)) # category positions

# Creating the stackplot
bar_width = 0.25  # width of the bars

fig, ax = plt.subplots(figsize=(12, 7))

# Defining colors
color_correct_1 = "#6baed6"  # light blue
color_incorrect_1 = "#2171b5"  # dark blue

color_correct_2 = "#74c476"  # light green
color_incorrect_2 = "#238b45"  # dark green

color_correct_3 = "#fdae61"  # light orange
color_incorrect_3 = "#d73027"  # dark red

# Model 1
bars1 = ax.bar(x - bar_width, correct_model_1, bar_width, color=color_correct_1, edgecolor='black', linewidth=0.6)
bars1_incorrect = ax.bar(x - bar_width, incorrect_model_1, bar_width, bottom=correct_model_1, color=color_incorrect_1, edgecolor='black', linewidth=0.6)

# Model 2
bars2 = ax.bar(x, correct_model_2, bar_width, color=color_correct_2, edgecolor='black', linewidth=0.6)
bars2_incorrect = ax.bar(x, incorrect_model_2, bar_width, bottom=correct_model_2, color=color_incorrect_2, edgecolor='black', linewidth=0.6)

# Model 3
bars3 = ax.bar(x + bar_width, correct_model_3, bar_width, color=color_correct_3, edgecolor='black', linewidth=0.6)
bars3_incorrect = ax.bar(x + bar_width, incorrect_model_3, bar_width, bottom=correct_model_3, color=color_incorrect_3, edgecolor='black', linewidth=0.6)

# Labels for each bar (numeric value)
for bars in [bars1, bars1_incorrect, bars2, bars2_incorrect, bars3, bars3_incorrect]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.annotate(f'{height}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_y() + height / 2),
                        xytext=(0, 5),
                        textcoords="offset points",
                        ha='center', va='top', fontsize=10, color='white', fontweight='bold')


spacer = mpatches.Patch(color='none', label='')

# Creating dots for models
dot_model_1 = mpatches.Circle((0, 0), radius=0.1, facecolor="black", edgecolor="black", label=LABEL_MODEL_1)
dot_model_2 = mpatches.Circle((0, 0), radius=0.1, facecolor="black", edgecolor="black", label=LABEL_MODEL_2)
dot_model_3 = mpatches.Circle((0, 0), radius=0.1, facecolor="black", edgecolor="black", label=LABEL_MODEL_3)

# Creating colored bars with black border for the legend
bar_correct_1 = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_correct_1, edgecolor='black', linewidth=0.8, label="   Correct")
bar_incorrect_1 = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_incorrect_1, edgecolor='black', linewidth=0.8, label="   Incorrect")

bar_correct_2 = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_correct_2, edgecolor='black', linewidth=0.8, label="   Correct")
bar_incorrect_2 = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_incorrect_2, edgecolor='black', linewidth=0.8, label="   Incorrect")

bar_correct_3 = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_correct_3, edgecolor='black', linewidth=0.8, label="   Correct")
bar_incorrect_3 = mpatches.Rectangle((0, 0), 1, 1, facecolor=color_incorrect_3, edgecolor='black', linewidth=0.8, label="   Incorrect")


# Combine model legends with spaces
#legend_patches = dot_model_1 + bar_correct_1 + bar_incorrect_1 + [spacer1] + dot_model_2 + bar_correct_2 + bar_incorrect_2 + [spacer1] + dot_model_3 + bar_correct_3 + bar_incorrect_3

ax.set_xlabel("Categories", fontsize=14, fontweight='bold')
ax.set_ylabel("Number of Queries", fontsize=14, fontweight='bold')
if(LANGUAGE == "en"):
    ax.set_title("Comparison of SQL Query Results by Model and Category (English)", fontsize=16, fontweight='bold', pad=20)
else:
    ax.set_title("Comparison of SQL Query Results by Model and Category (Italian)", fontsize=16, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=12, rotation=20)

#ax.legend(handles=legend_patches, fontsize=12, loc='upper left', bbox_to_anchor=(1,1))
legend = ax.legend(
    handles=[dot_model_1, bar_correct_1, bar_incorrect_1, spacer,
             dot_model_2, bar_correct_2, bar_incorrect_2, spacer,
             dot_model_3, bar_correct_3, bar_incorrect_3],
    loc="upper left",
    bbox_to_anchor=(1, 1),
    handlelength=1,
    handletextpad=0.5,
    fontsize=12,
    handler_map={
        mpatches.Circle: HandlerCircle(),  # customisation circles
        mpatches.Rectangle: HandlerRectangle()  # customisation rectangles
    }
)

# Make only model labels bold
for text in legend.get_texts():
    if text.get_text() in {LABEL_MODEL_1, LABEL_MODEL_2, LABEL_MODEL_3}:
        text.set_fontweight('bold')

ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# Save the graph
path_save = BASE_PATH+"/venvText2SQL/models/Graphs/single_category_performance/Stackplot/"+NAME_OUTPUT_FILE_GRAPH+".png"
os.makedirs(os.path.dirname(path_save), exist_ok=True)
plt.savefig(path_save, format='png')
print(f"\nGraph saved in: {NAME_OUTPUT_FILE_GRAPH}\n")