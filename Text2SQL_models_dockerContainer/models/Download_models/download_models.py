from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer
)
import os
from dotenv import load_dotenv

# Load file .env
load_dotenv("configuration.env")

# Retrieve all configuration variables:
LOGIN_TOKEN_HUGGING_FACE = os.getenv("LOGIN_TOKEN_HUGGING_FACE")
BASE_PATH = os.getenv("BASE_PATH")
BASE_MODELS_DIR = os.getenv("BASE_MODELS_DIR")
MODEL_NAME = os.getenv("MODEL_NAME")
NAME_CACHE_DIR = os.getenv("NAME_CACHE_DIR")
NAME_OFFLOAD_CACHE_DIR = os.getenv("NAME_OFFLOAD_CACHE_DIR")

print("")
print("- Summary of all configuration variables present in 'configuration.env':")
print("")
print("LOGIN_TOKEN_HUGGING_FACE: ", LOGIN_TOKEN_HUGGING_FACE)
print("BASE_PATH: ", BASE_PATH)
print("BASE_MODELS_DIR: ", BASE_MODELS_DIR)
print("MODEL_NAME: ", MODEL_NAME)
print("NAME_CACHE_DIR: ", NAME_CACHE_DIR)
print("NAME_OFFLOAD_CACHE_DIR: ", NAME_OFFLOAD_CACHE_DIR)
print("")
print("")

# Login con token huggingface:
login(token=LOGIN_TOKEN_HUGGING_FACE)

# Definition folder paths
cache_dir = os.path.join(BASE_PATH, "venvText2SQL/models/Download_models/Base_models/Cache_dir", NAME_CACHE_DIR) # path cache folder where Hugging Face will save downloaded model files (weights, configuration, tokenizer, etc.) to avoid having to re-download them every time
offload_cache_dir = os.path.join(BASE_PATH, "venvText2SQL/models/Download_models/Base_models/Offload_dir", NAME_OFFLOAD_CACHE_DIR) # path folder in which "dynamic offloading" of parts of the model will be applied, that is, temporarily moving pieces of the model to disk when they are not in active use, rather than keeping them in RAM or VRAM memory

print("")
print("- Full path to the files of the new downloaded model: ", cache_dir)
print("- Full path to offload_cache_dir folder of new model: ", offload_cache_dir)
print("")
print("")

# Create folders if they don't exist.
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(offload_cache_dir, exist_ok=True)

# Download model:
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map='auto', offload_folder=offload_cache_dir, cache_dir=cache_dir)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=cache_dir)

print("")
print("Model download completed successfully.")
print("")
