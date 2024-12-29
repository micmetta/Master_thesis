from datasets import load_dataset
import json
from dotenv import load_dotenv
import os

# Load file .env
load_dotenv("configuration.env")

# Retrieve all configuration variables:
BASE_PATH = os.getenv("BASE_PATH")
NAME_DOWNLOAD_DATASET = os.getenv("NAME_DOWNLOAD_DATASET")
NAME_DATASET = os.getenv("NAME_DATASET")

# Load dataset from Hugging Face
dataset = load_dataset(NAME_DOWNLOAD_DATASET)

# Path where you want to save the dataset in JSON format
save_path = BASE_PATH+"/venvText2SQL/models/Fine_tuning_dataset/dataset/training/"+NAME_DATASET+".json"

# Converts the dataset into the desired structure (such as "BioInfoDataset.json") and saves it in JSON format
converted_data = [{"text": entry} for entry in dataset['train']['text']]

# Converts the dataset into a list of dictionaries and saves it in JSON format
with open(save_path, "w") as f:
    json.dump(converted_data, f, indent=4)

print("")
print("- Number of rows in dataset:: ", len(converted_data))
print(f"- Dataset downloaded and saved successfully in: {save_path}")
print("")
