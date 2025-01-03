
#############################################################################################################################################
# - Script to execute the second fine-tuning of an LLM on a certain dataset.
#############################################################################################################################################

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer
import threading
import csv
from datetime import datetime
import time
from dotenv import load_dotenv

# Memory monitoring function:
def monitor_memory(interval, stop_event, csv_writer):
    while not stop_event.is_set():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # in GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3    # in GB
        datetime_obj = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S") # convert the timestamp to a datetime object
        csv_writer.writerow([datetime_obj, round(allocated, 2), round(reserved, 2)])
        time.sleep(interval)


##########################################################################################
# STEP 1 - Loading configuration variables

# Load file .env
load_dotenv("configuration.env")

# Retrieve all configuration variables:
BASE_PATH = os.getenv("BASE_PATH")
PATH_DATASET = os.getenv("PATH_DATASET")
PATH_BASE_MODEL = os.getenv("PATH_BASE_MODEL")
PATH_BASE_MODEL = BASE_PATH+"/"+PATH_BASE_MODEL
PATH_OFFLOAD_BASE_MODEL_DIR = os.getenv("PATH_OFFLOAD_BASE_MODEL_DIR")
NAME_NEW_MODEL = os.getenv("NAME_NEW_MODEL")
NAME_OFFLOAD_CACHE_DIR_NEW_MODEL = os.getenv("NAME_OFFLOAD_CACHE_DIR_NEW_MODEL")
WAIT = int(os.getenv("WAIT")) # number of seconds between memory reads
QUANTIZATION_BASE_MODEL = bool(os.getenv("QUANTIZATION_BASE_MODEL", "").lower() == "true")

# Retrieve all variables for LLM hyperparameter setting:

################################################################################
# QLoRA parameters
################################################################################
# LoRA attention dimension (Low-Rank Adaptation)
LORA_R = int(os.getenv("LORA_R"))
# Alpha parameter for LoRA scaling
LORA_ALPHA = int(os.getenv("LORA_ALPHA"))
# Dropout probability for LoRA layers
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT"))

################################################################################
# TrainingArguments parameters
################################################################################
# Number of training epochs
NUM_TRAIN_EPOCHS = int(os.getenv("NUM_TRAIN_EPOCHS"))

# Enable fp16/bf16 training (set bf16 to True with an A100)
FP16 = bool(os.getenv("FP16", "").lower() == "true")
BF16 = bool(os.getenv("BF16", "").lower() == "true")

# Batch size per GPU for training
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("PER_DEVICE_TRAIN_BATCH_SIZE"))

# Batch size per GPU for evaluation
PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("PER_DEVICE_EVAL_BATCH_SIZE"))

# Number of update steps to accumulate the gradients for
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS"))

# Enable gradient checkpointing
GRADIENT_CHECKPOINTING = bool(os.getenv("GRADIENT_CHECKPOINTING", "").lower() == "true")

# Maximum gradient normal (gradient clipping)
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM"))

# Initial learning rate (AdamW optimizer)
LEARNING_RATE = float(os.getenv("LEARNING_RATE"))

# Weight decay to apply to all layers except bias/LayerNorm weights
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY"))

# Optimizer to use
OPTIM = os.getenv("OPTIM")

# Learning rate schedule
LR_SCHEDULER_TYPE = os.getenv("LR_SCHEDULER_TYPE")

# Number of training steps (overrides num_train_epochs)
MAX_STEPS = int(os.getenv("MAX_STEPS"))

# Ratio of steps for a linear warmup (from 0 to learning rate)
WARMPUT_RATIO = float(os.getenv("WARMPUT_RATIO"))

# Group sequences into batches with same length
# Saves memory and speeds up training considerably
GROUP_BY_LENGTH = bool(os.getenv("GROUP_BY_LENGTH", "").lower() == "true")

# Save checkpoint every X updates steps
SAVE_STEPS = int(os.getenv("SAVE_STEPS"))

# Log every X updates steps
LOGGING_STEPS = int(os.getenv("LOGGING_STEPS"))

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
MAX_SEQ_LENGTH = os.getenv("MAX_SEQ_LENGTH")
MAX_SEQ_LENGTH = int(MAX_SEQ_LENGTH) if MAX_SEQ_LENGTH and MAX_SEQ_LENGTH.lower() != "none" else None

# Pack multiple short examples in the same input sequence to increase efficiency
PACKING = bool(os.getenv("PACKING", "").lower() == "true")

# Load the entire model on the GPU 0 (or use "auto" to distribute model loading between CPU and GPU)
DEVICE_MAP = os.getenv("DEVICE_MAP")
##########################################################################################


print("")
print("- Summary of all configuration variables present in 'configuration.env':")
print("")
print("BASE_PATH: ", BASE_PATH)
print("PATH_DATASET: ", PATH_DATASET)
print("PATH_BASE_MODEL: ", PATH_BASE_MODEL)
print("PATH_OFFLOAD_BASE_MODEL_DIR: ", PATH_OFFLOAD_BASE_MODEL_DIR)
print("NAME_NEW_MODEL: ", NAME_NEW_MODEL)
print("NAME_OFFLOAD_CACHE_DIR_NEW_MODEL: ", NAME_OFFLOAD_CACHE_DIR_NEW_MODEL)
print("")
print("")
print("- Summary of hyperparameter configuration variables present in 'configuration.env':")
print("")
print("WAIT: ", WAIT)
print("QUANTIZATION_BASE_MODEL: ", QUANTIZATION_BASE_MODEL)
print("LORA_R: ", LORA_R)
print("LORA_ALPHA: ", LORA_ALPHA)
print("LORA_DROPOUT: ", LORA_DROPOUT)
print("NUM_TRAIN_EPOCHS: ", NUM_TRAIN_EPOCHS)
print("FP16: ", FP16)
print("BF16: ", BF16)
print("PER_DEVICE_TRAIN_BATCH_SIZE: ", PER_DEVICE_TRAIN_BATCH_SIZE)
print("PER_DEVICE_EVAL_BATCH_SIZE: ", PER_DEVICE_EVAL_BATCH_SIZE)
print("GRADIENT_ACCUMULATION_STEPS: ", GRADIENT_ACCUMULATION_STEPS)
print("GRADIENT_CHECKPOINTING: ", GRADIENT_CHECKPOINTING)
print("MAX_GRAD_NORM: ", MAX_GRAD_NORM)
print("LEARNING_RATE: ", LEARNING_RATE)
print("WEIGHT_DECAY: ", WEIGHT_DECAY)
print("OPTIM: ", OPTIM)
print("LR_SCHEDULER_TYPE: ", LR_SCHEDULER_TYPE)
print("MAX_STEPS: ", MAX_STEPS)
print("WARMPUT_RATIO: ", WARMPUT_RATIO)
print("GROUP_BY_LENGTH: ", GROUP_BY_LENGTH)
print("SAVE_STEPS: ", SAVE_STEPS)
print("LOGGING_STEPS: ", LOGGING_STEPS)
print("MAX_SEQ_LENGTH: ", MAX_SEQ_LENGTH)
print("PACKING: ", PACKING)
print("DEVICE_MAP: ", DEVICE_MAP)
print("")
print("")
##########################################################################################


##########################################################################################
# STEP 2 - SETTING paths needed for fine-tuning the model:

# I create the full path of the fine-tuning dataset:
path_dataset = os.path.join(BASE_PATH, PATH_DATASET)
print("")
print("Full path to the FT dataset:", path_dataset)
print("")
print("")
print("Complete path of the basic model:", PATH_BASE_MODEL)
print("")
print("")

# Directory where the new fine-tuned model will be saved:
new_model = BASE_PATH+"/venvText2SQL/models/FineTuning_models_double/Cache_dir/"+NAME_NEW_MODEL

# Directory offload of new model:
new_model_offload_dir = BASE_PATH+"/venvText2SQL/models/FineTuning_models_double/Offload_dir/"+NAME_OFFLOAD_CACHE_DIR_NEW_MODEL

# Memory Monitor Data Folder Path:
PATH_csv = BASE_PATH+"/venvText2SQL/models/FineTuning_models_double/Memory_occupation/"+NAME_NEW_MODEL+"/memory_usage.csv"

# Extract only the path part of the directory
PATH_csv_dir = os.path.dirname(PATH_csv)

# Create folders if they do not exist
os.makedirs(new_model, exist_ok=True)
os.makedirs(new_model_offload_dir, exist_ok=True)
os.makedirs(PATH_csv_dir, exist_ok=True)
##########################################################################################


##########################################################################################
# STEP 3 - SETTING output folder:

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"
##########################################################################################


##########################################################################################
# Step 4: Loading dataset for fine-tuning:

dataset = load_dataset('json', data_files=path_dataset, split="train")
print(f"Numero totale righe del dataset: {len(dataset)}")
print("")
##########################################################################################


##########################################################################################
# Step 5: Fully load the LLM with all the chosen hyperparameter values:

# Quantize or not the base model:
base_model = None
bnb_config = None
if(QUANTIZATION_BASE_MODEL):
    print("- I PERFORM 4-BIT QUANTIZATION.")
    print("")
    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=False
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        PATH_BASE_MODEL,
        device_map = {"": 0},
        offload_folder=PATH_OFFLOAD_BASE_MODEL_DIR,
        cache_dir=PATH_BASE_MODEL,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
else:
    print("- I DO NOT PERFORM 4-BIT QUANTIZATION.")
    print("")
    base_model = AutoModelForCausalLM.from_pretrained(
        PATH_BASE_MODEL,
        device_map = {"": 0},
        offload_folder=PATH_OFFLOAD_BASE_MODEL_DIR,
        cache_dir=PATH_BASE_MODEL,
        torch_dtype=torch.float16,
)
base_model.config.use_cache = False
base_model.config.pretraining_tp = 1


# Load LLaMA tokenizer
tokenizer = AutoTokenizer.from_pretrained(PATH_BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
)

# Set training parameters
training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIM,
    save_steps=SAVE_STEPS,
    logging_steps=LOGGING_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=FP16,
    bf16=BF16,
    max_grad_norm=MAX_GRAD_NORM,
    max_steps=MAX_STEPS,
    warmup_ratio=WARMPUT_RATIO,
    group_by_length=GROUP_BY_LENGTH,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    report_to="tensorboard"
)

# Set supervised fine-tuning parameters
trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=PACKING,
)
##########################################################################################


##########################################################################################
# Step 6: Start fine-tuning process and save new model and memory footprint file

# Opening .csv files:
with open(PATH_csv, mode="w", newline="") as file:
    csv_writer = csv.writer(file)
    csv_writer.writerow([])
    csv_writer.writerow(["NAME_MODEL_TO_PATH_CSV: " + NAME_NEW_MODEL])
    csv_writer.writerow([])
    csv_writer.writerow(["timestamp", "allocated_memory_GB", "reserved_memory_GB"])

    stop_event = threading.Event() # I create an event to stop the monitoring thread
    monitor_thread = threading.Thread(target=monitor_memory, args=(WAIT, stop_event, csv_writer))
    monitor_thread.start() # start thread
    start_time = time.time()


    # Train model
    print("Start double fine-tuning model.")
    trainer.train()

    end_time = time.time()
    stop_event.set() # signal the thread to stop
    monitor_thread.join() # I'm waiting for the thread to end

    print("End double fine-tuning model.")
    elapsed_time = end_time - start_time
    csv_writer.writerow([])
    csv_writer.writerow([])
    csv_writer.writerow(["elapsed_time: " + str(elapsed_time)])

    # Save trained model
    trainer.model.save_pretrained(new_model)
    print("Saving new model completed.")

    # Save tokenizer
    tokenizer.save_pretrained(new_model)
    print("Saving new model tokenizer completed.")


print("")
print("- The new model obtained after the second FT has this path: ", new_model)
print("")
print("- Full path to new model offloading folder after second FT: ", new_model_offload_dir)
print("")
print("- The file containing the memory usage info of the new model after the second FT has this path: ", PATH_csv)
print("")
print("Second fine-tuning successfully completed.")
print("")
##########################################################################################
