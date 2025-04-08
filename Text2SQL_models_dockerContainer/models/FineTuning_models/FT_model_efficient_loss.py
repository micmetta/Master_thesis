#############################################################################################################################################
# - Script to fine-tune a generic LLM on a given dataset with LoRA and Loss Masking.
#############################################################################################################################################

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig
from trl import SFTTrainer
import threading
import csv
from datetime import datetime
import time
from dotenv import load_dotenv
import shutil
import bitsandbytes as bnb

# Memory monitoring function:
def monitor_memory(interval, stop_event, csv_writer):
    while not stop_event.is_set():
        allocated = torch.cuda.memory_allocated(0) / 1024**3  # in GB
        reserved = torch.cuda.memory_reserved(0) / 1024**3    # in GB
        datetime_obj = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S") # convert timestamp to datetime
        csv_writer.writerow([datetime_obj, round(allocated, 2), round(reserved, 2)])
        time.sleep(interval)

##########################################################################################
# STEP 1 - Loading configuration variables

# Load file .env
load_dotenv("configuration.env")

# Retrieve all configuration variables:
BASE_PATH = os.getenv("BASE_PATH")
PATH_DATASET_TRAINING = os.getenv("PATH_DATASET_TRAINING")
PATH_DATASET_VALIDATION = os.getenv("PATH_DATASET_VALIDATION")
PATH_BASE_MODEL = os.getenv("PATH_BASE_MODEL")
PATH_BASE_MODEL = os.path.join(BASE_PATH, PATH_BASE_MODEL)
PATH_OFFLOAD_BASE_MODEL_DIR = os.getenv("PATH_OFFLOAD_BASE_MODEL_DIR")
NAME_NEW_MODEL = os.getenv("NAME_NEW_MODEL")
NAME_OFFLOAD_CACHE_DIR_NEW_MODEL = os.getenv("NAME_OFFLOAD_CACHE_DIR_NEW_MODEL")
WAIT = int(os.getenv("WAIT")) # number of seconds between memory reads
QUANTIZATION_BASE_MODEL = os.getenv("QUANTIZATION_BASE_MODEL", "").lower() == "true"

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

################################################################################
# SFT parameters
################################################################################

# Maximum sequence length to use
#MAX_SEQ_LENGTH = os.getenv("MAX_SEQ_LENGTH")
#MAX_SEQ_LENGTH = int(MAX_SEQ_LENGTH) if MAX_SEQ_LENGTH and MAX_SEQ_LENGTH.lower() != "none" else None

# Pack multiple short examples in the same input sequence to increase efficiency
PACKING = bool(os.getenv("PACKING", "").lower() == "true")

# Load the entire model on the GPU 0 (or use "auto" to distribute model loading between CPU and GPU)
DEVICE_MAP = os.getenv("DEVICE_MAP")
##########################################################################################

print("")
print("- Summary of all configuration variables present in 'configuration.env':")
print("")
print("BASE_PATH: ", BASE_PATH)
print("PATH_DATASET_TRAINING: ", PATH_DATASET_TRAINING)
print("PATH_DATASET_VALIDATION: ", PATH_DATASET_VALIDATION)
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
#print("LOGGING_STEPS: ", LOGGING_STEPS)
#print("MAX_SEQ_LENGTH: ", MAX_SEQ_LENGTH)
print("PACKING: ", PACKING)
print("DEVICE_MAP: ", DEVICE_MAP)
print("")
print("")
##########################################################################################


# Directory where the new fine-tuned model will be saved:
new_model = BASE_PATH+"/venvText2SQL/models/FineTuning_models/Fine_tuned_models/Cache_dir/"+NAME_NEW_MODEL

# Directory offload of new model:
new_model_offload_dir = BASE_PATH+"/venvText2SQL/models/FineTuning_models/Fine_tuned_models/Offload_dir/"+NAME_OFFLOAD_CACHE_DIR_NEW_MODEL

# Create folders if they do not exist
os.makedirs(new_model, exist_ok=True)
os.makedirs(new_model_offload_dir, exist_ok=True)

##########################################################################################
# STEP 2 - SETTING output folder:

# cleaning old checkpoints before training
if os.path.exists("./results"):
    shutil.rmtree("./results")

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"
##########################################################################################

##########################################################################################
# STEP 3 - Load LLM with quantization (if enabled)

if QUANTIZATION_BASE_MODEL:
    print("- Applying 4-bit quantization.")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=False
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        PATH_BASE_MODEL,
        device_map={"": 0},
        offload_folder=PATH_OFFLOAD_BASE_MODEL_DIR,
        torch_dtype=torch.float16,
        quantization_config=bnb_config
    )
else:
    print("- Using full precision model.")
    base_model = AutoModelForCausalLM.from_pretrained(
        PATH_BASE_MODEL,
        device_map={"": 0},
        offload_folder=PATH_OFFLOAD_BASE_MODEL_DIR,
        torch_dtype=torch.float16
    )

base_model.config.use_cache = False

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(PATH_BASE_MODEL)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right" # Fix weird overflow issue with fp16 training

##########################################################################################
# STEP 4 - Loading dataset and applying loss masking

# Function to apply loss masking (ignore loss for prompt tokens)
def preprocess_function(example):
    #print("example: ", example)
    text = example["text"]
    
    # Tokenization
    #tokenized = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH, padding="max_length")
    tokenized = tokenizer(text, padding="longest")

    # Find the start of the SQL query
    #sql_start = text.find("```")  # Locate query start
    # Trova tutte le occorrenze di "```"
    sql_positions = [i for i in range(len(text)) if text.startswith("```", i)]
    sql_start = sql_positions[1]

    sql_tokens_start = len(tokenizer(text[:sql_start])["input_ids"]) if sql_start != -1 else len(tokenized["input_ids"])

    # Create labels with loss masking (-100 for ignored tokens)
    labels = tokenized["input_ids"][:]
    for i in range(sql_tokens_start):  # Masking non-SQL tokens
        labels[i] = -100  # Transformers ignores -100 in loss calculation

    tokenized["labels"] = labels
    return tokenized

# Load dataset
dataset_training = load_dataset('json', data_files=os.path.join(BASE_PATH, PATH_DATASET_TRAINING), split="train").shuffle(seed=42)
dataset_validation = load_dataset('json', data_files=os.path.join(BASE_PATH, PATH_DATASET_VALIDATION), split="train")

# Apply loss masking
#tokenized_dataset_training = dataset_training.map(preprocess_function, remove_columns=["text"])
#tokenized_dataset_validation = dataset_validation.map(preprocess_function, remove_columns=["text"])
tokenized_dataset_training = dataset_training.map(preprocess_function)
tokenized_dataset_validation = dataset_validation.map(preprocess_function)

print("")
print("")
print("tokenized_dataset_training[0]:", tokenized_dataset_training[0])
print("")
print("")
print("tokenized_dataset_validation[0]:", tokenized_dataset_validation[0])
print("")
print("")

# Data Collator for padding and correct loss computation
#data_collator = DataCollatorForSeq2Seq(tokenizer, model=base_model, label_pad_token_id=-100)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False) # False per il causal LM

print(f"Dataset Training length: {len(dataset_training)}")
print(f"Dataset Validation length: {len(dataset_validation)}")
print(f"Tokenized Training length: {len(tokenized_dataset_training)}")
print(f"Tokenized Validation length: {len(tokenized_dataset_validation)}\n")

lengths_dataset_training= [len(tokenizer(example["text"])["input_ids"]) for example in dataset_training]
print("Lunghezza media (dataset_training):", sum(lengths_dataset_training)/len(lengths_dataset_training))
print("Lunghezza massima (dataset_training):", max(lengths_dataset_training))

lengths_dataset_validation= [len(tokenizer(example["text"])["input_ids"]) for example in dataset_validation]
print("Lunghezza media (dataset_validation):", sum(lengths_dataset_validation)/len(lengths_dataset_validation))
print("Lunghezza massima (dataset_validation):", max(lengths_dataset_validation))
##########################################################################################
# STEP 5 - Configure TrainingArguments and Trainer

'''
training_arguments = TrainingArguments(
    output_dir="./results",
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    fp16=FP16,
    bf16=BF16,
    evaluation_strategy="epoch",   # Assicura che l'early stopping abbia metriche per valutare il modello
    save_strategy="epoch",         # Salvataggio ad ogni epoca
    load_best_model_at_end=True,   # NECESSARIO PER EARLY STOPPING
    metric_for_best_model="eval_loss",  # Indica la metrica da usare per selezionare il miglior modello
    greater_is_better=False,       # Per `eval_loss`, minore è meglio
    report_to="tensorboard"
)
'''

training_arguments = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    optim=OPTIM,
    save_steps=SAVE_STEPS,
    logging_strategy="epoch",
    learning_rate=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    fp16=FP16,
    bf16=BF16,
    max_grad_norm=MAX_GRAD_NORM,
    max_steps=MAX_STEPS,
    warmup_ratio=WARMPUT_RATIO,
    group_by_length=GROUP_BY_LENGTH,
    lr_scheduler_type=LR_SCHEDULER_TYPE,
    evaluation_strategy="epoch",  # Può essere 'steps' o 'epoch'
    save_strategy="epoch",  # Salvataggio ad ogni epoca per evitare il conflitto
    load_best_model_at_end=True,  # Carica il miglior modello al termine dell'addestramento
    metric_for_best_model="eval_loss",  # Monitora la metrica 'eval_loss'
    greater_is_better=False,  # 'eval_loss' è meglio quando è minore
    report_to="tensorboard"
)

# Integrazione del callback per l'early stopping
early_stopping = EarlyStoppingCallback(
    early_stopping_patience=2,  # Numero di valutazioni senza miglioramento prima dello stop
    early_stopping_threshold=0.0  # Miglioramento minimo richiesto per reset del contatore
)

# Load LoRA configuration
peft_config = LoraConfig(
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    r=LORA_R,
    bias="none",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model=base_model,
    train_dataset=tokenized_dataset_training,
    eval_dataset=tokenized_dataset_validation,
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=4096,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=PACKING,
    callbacks=[early_stopping],
    data_collator=data_collator  # Ensure correct loss computation
)


# Calculate the total number of model parameters
if not QUANTIZATION_BASE_MODEL:
  total_params = sum(p.numel() for p in trainer.model.parameters())
else:
  total_params = 0
  for name, module in base_model.named_modules():
      if isinstance(module, bnb.nn.Linear4bit):  # quantized layers only
          total_params += module.in_features * module.out_features
      elif hasattr(module, 'weight') and module.weight is not None:
          total_params += module.weight.numel()

# Calcola il numero di parametri addestrabili (solo quelli con requires_grad=True)
trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)

# Calcola la percentuale di parametri addestrabili rispetto al totale
trainable_percentage = (trainable_params / total_params) * 100

print(f"Pretraining TP: {base_model.config.pretraining_tp}")
print(f"\nTotal Parameters: {total_params:,}")
print(f"Trainable Parameters (LoRA): {trainable_params:,}")
print(f"Percentage of Trainable Parameters (LoRA): {trainable_percentage:.4f}%\n")
##########################################################################################

##########################################################################################
# STEP 6 - Start Fine-Tuning Process

print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning completed.")

# Save fine-tuned model
new_model = BASE_PATH+"/venvText2SQL/models/FineTuning_models/Fine_tuned_models/Cache_dir/"+NAME_NEW_MODEL
trainer.model.save_pretrained(new_model)
tokenizer.save_pretrained(new_model)

print("Model and tokenizer saved successfully.")

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss

/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 629 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████                                                                                                      | 4083/15000 [11:30<26:48,  6.79it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 630 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████                                                                                                      | 4084/15000 [11:30<26:57,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 631 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4085/15000 [11:31<27:02,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 632 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4086/15000 [11:31<27:07,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 633 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4087/15000 [11:31<27:05,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 634 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4088/15000 [11:31<26:58,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 635 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4089/15000 [11:31<27:00,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 636 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4090/15000 [11:31<27:03,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 637 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4091/15000 [11:32<27:11,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 638 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4092/15000 [11:32<27:14,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 639 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4093/15000 [11:32<27:17,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 640 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4094/15000 [11:32<27:06,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 641 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4095/15000 [11:32<26:56,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 642 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4096/15000 [11:32<27:00,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 643 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4097/15000 [11:32<27:02,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 644 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▏                                                                                                     | 4098/15000 [11:33<27:02,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 645 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4099/15000 [11:33<27:04,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 646 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4100/15000 [11:33<27:08,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 647 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4101/15000 [11:33<26:41,  6.81it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 648 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4102/15000 [11:33<26:41,  6.80it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 649 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4103/15000 [11:33<26:46,  6.78it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 650 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4104/15000 [11:33<26:54,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 651 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4105/15000 [11:34<27:02,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 652 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4106/15000 [11:34<27:08,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 653 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4107/15000 [11:34<27:16,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 654 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4108/15000 [11:34<26:49,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 655 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4109/15000 [11:34<26:50,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 656 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4110/15000 [11:34<26:55,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 657 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▎                                                                                                     | 4111/15000 [11:34<27:00,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 658 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4112/15000 [11:35<27:05,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 659 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4113/15000 [11:35<26:51,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 660 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4114/15000 [11:35<26:48,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 661 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4115/15000 [11:35<26:42,  6.79it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 662 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4116/15000 [11:35<26:46,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 663 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4117/15000 [11:35<26:52,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 664 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4118/15000 [11:36<27:01,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 665 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4119/15000 [11:36<27:12,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 666 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4120/15000 [11:36<27:15,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 667 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4121/15000 [11:36<26:58,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 668 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4122/15000 [11:36<26:55,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 669 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4123/15000 [11:36<26:58,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 670 samples have been fetched. 
  warnings.warn(warn_msg)
 27%|██████████████████████████████████████▍                                                                                                     | 4124/15000 [11:36<27:03,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 671 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4125/15000 [11:37<27:09,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 672 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4126/15000 [11:37<27:09,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 673 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4127/15000 [11:37<27:14,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 674 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4128/15000 [11:37<26:37,  6.81it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 675 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4129/15000 [11:37<26:39,  6.80it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 676 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4130/15000 [11:37<26:46,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 677 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4131/15000 [11:37<26:54,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 678 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4132/15000 [11:38<27:04,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 679 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4133/15000 [11:38<27:08,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 680 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4134/15000 [11:38<27:11,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 681 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4135/15000 [11:38<26:53,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 682 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4136/15000 [11:38<26:54,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 683 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4137/15000 [11:38<27:01,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 684 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▌                                                                                                     | 4138/15000 [11:39<27:09,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 685 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▋                                                                                                     | 4139/15000 [11:39<27:15,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 686 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▋                                                                                                     | 4140/15000 [11:39<27:15,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 687 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▋                                                                                                     | 4141/15000 [11:39<27:04,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 688 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▋                                                                                                     | 4142/15000 [11:39<26:55,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 689 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▋                                                                                                     | 4143/15000 [11:39<26:56,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 690 samples have been fetched. 
  warnings.warn(warn_msg)
 28%|██████████████████████████████████████▋                                                                                                     | 4144/15000 [11:39<26:58,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 691 samples have been fetched. 
  warnings.warn(warn_msg)
{'eval_loss': 0.06093303859233856, 'eval_runtime': 15.0186, 'eval_samples_per_second': 19.975, 'eval_steps_per_second': 4.994, 'epoch': 6.38}                                         
{'loss': 0.0272, 'learning_rate': 3.500666666666667e-05, 'epoch': 6.71}                                                                                                               
 31%|███████████████████████████████████████████▎                                                                                                | 4645/15000 [13:10<25:46,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 501 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▎                                                                                                | 4646/15000 [13:10<25:51,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 502 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▎                                                                                                | 4647/15000 [13:11<25:55,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 503 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4648/15000 [13:11<25:56,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 504 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4649/15000 [13:11<26:00,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 505 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4650/15000 [13:11<25:26,  6.78it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 506 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4651/15000 [13:11<25:27,  6.78it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 507 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4652/15000 [13:11<25:34,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 508 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4653/15000 [13:11<25:41,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 509 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4654/15000 [13:12<25:51,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 510 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4655/15000 [13:12<26:01,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 511 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4656/15000 [13:12<25:59,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 512 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4657/15000 [13:12<25:46,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 513 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4658/15000 [13:12<25:45,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 514 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4659/15000 [13:12<25:51,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 515 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▍                                                                                                | 4660/15000 [13:13<25:56,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 516 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4661/15000 [13:13<25:57,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 517 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4662/15000 [13:13<26:00,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 518 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4663/15000 [13:13<25:49,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 519 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4664/15000 [13:13<25:38,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 520 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4665/15000 [13:13<25:44,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 521 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4666/15000 [13:13<25:48,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 522 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4667/15000 [13:14<25:55,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 523 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4668/15000 [13:14<26:01,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 524 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4669/15000 [13:14<26:01,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 525 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4670/15000 [13:14<25:26,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 526 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4671/15000 [13:14<25:27,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 527 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4672/15000 [13:14<25:34,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 528 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4673/15000 [13:14<25:43,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 529 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▌                                                                                                | 4674/15000 [13:15<25:50,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 530 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4675/15000 [13:15<25:58,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 531 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4676/15000 [13:15<25:58,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 532 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4677/15000 [13:15<25:44,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 533 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4678/15000 [13:15<25:45,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 534 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4679/15000 [13:15<25:47,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 535 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4680/15000 [13:16<25:54,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 536 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4681/15000 [13:16<25:56,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 537 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4682/15000 [13:16<25:58,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 538 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4683/15000 [13:16<25:45,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 539 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4684/15000 [13:16<25:38,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 540 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4685/15000 [13:16<25:39,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 541 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4686/15000 [13:16<25:45,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 542 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▋                                                                                                | 4687/15000 [13:17<25:52,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 543 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4688/15000 [13:17<25:56,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 544 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4689/15000 [13:17<26:01,  6.60it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 545 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4690/15000 [13:17<25:26,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 546 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4691/15000 [13:17<25:26,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 547 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4692/15000 [13:17<25:33,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 548 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4693/15000 [13:17<25:41,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 549 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4694/15000 [13:18<25:48,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 550 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4695/15000 [13:18<25:51,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 551 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4696/15000 [13:18<25:51,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 552 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4697/15000 [13:18<25:37,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 553 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4698/15000 [13:18<25:33,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 554 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4699/15000 [13:18<25:38,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 555 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▊                                                                                                | 4700/15000 [13:19<25:40,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 556 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4701/15000 [13:19<25:42,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 557 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4702/15000 [13:19<25:47,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 558 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4703/15000 [13:19<25:39,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 559 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4704/15000 [13:19<25:28,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 560 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4705/15000 [13:19<25:28,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 561 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4706/15000 [13:19<25:38,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 562 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4707/15000 [13:20<25:42,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 563 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4708/15000 [13:20<25:51,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 564 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4709/15000 [13:20<25:55,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 565 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4710/15000 [13:20<25:26,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 566 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4711/15000 [13:20<25:29,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 567 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4712/15000 [13:20<25:34,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 568 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4713/15000 [13:20<25:39,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 569 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|███████████████████████████████████████████▉                                                                                                | 4714/15000 [13:21<25:49,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 570 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4715/15000 [13:21<25:50,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 571 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4716/15000 [13:21<25:48,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 572 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4717/15000 [13:21<25:35,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 573 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4718/15000 [13:21<25:33,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 574 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4719/15000 [13:21<25:34,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 575 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4720/15000 [13:22<25:39,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 576 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4721/15000 [13:22<25:42,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 577 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4722/15000 [13:22<25:47,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 578 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4723/15000 [13:22<25:41,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 579 samples have been fetched. 
  warnings.warn(warn_msg)
 31%|████████████████████████████████████████████                                                                                                | 4724/15000 [13:22<25:31,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 580 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████                                                                                                | 4725/15000 [13:22<25:36,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 581 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████                                                                                                | 4726/15000 [13:22<25:40,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 582 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████                                                                                                | 4727/15000 [13:23<25:45,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 583 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4728/15000 [13:23<25:51,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 584 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4729/15000 [13:23<25:52,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 585 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4730/15000 [13:23<25:25,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 586 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4731/15000 [13:23<25:20,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 587 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4732/15000 [13:23<25:28,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 588 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4733/15000 [13:23<25:32,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 589 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4734/15000 [13:24<25:41,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 590 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4735/15000 [13:24<25:43,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 591 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4736/15000 [13:24<25:47,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 592 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4737/15000 [13:24<25:27,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 593 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4738/15000 [13:24<25:26,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 594 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4739/15000 [13:24<25:34,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 595 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4740/15000 [13:25<25:39,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 596 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▏                                                                                               | 4741/15000 [13:25<25:47,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 597 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4742/15000 [13:25<25:47,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 598 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4743/15000 [13:25<25:33,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 599 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4744/15000 [13:25<25:25,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 600 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4745/15000 [13:25<25:24,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 601 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4746/15000 [13:25<25:31,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 602 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4747/15000 [13:26<25:36,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 603 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4748/15000 [13:26<25:40,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 604 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4749/15000 [13:26<25:44,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 605 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4750/15000 [13:26<25:22,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 606 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4751/15000 [13:26<25:21,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 607 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4752/15000 [13:26<25:26,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 608 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4753/15000 [13:26<25:31,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 609 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▎                                                                                               | 4754/15000 [13:27<25:39,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 610 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4755/15000 [13:27<25:44,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 611 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4756/15000 [13:27<25:45,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 612 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4757/15000 [13:27<25:11,  6.78it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 613 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4758/15000 [13:27<25:12,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 614 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4759/15000 [13:27<25:21,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 615 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4760/15000 [13:27<25:30,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 616 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4761/15000 [13:28<25:36,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 617 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4762/15000 [13:28<25:42,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 618 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4763/15000 [13:28<25:38,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 619 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4764/15000 [13:28<25:26,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 620 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4765/15000 [13:28<25:26,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 621 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4766/15000 [13:28<25:27,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 622 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▍                                                                                               | 4767/15000 [13:29<25:33,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 623 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4768/15000 [13:29<25:34,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 624 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4769/15000 [13:29<25:42,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 625 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4770/15000 [13:29<25:24,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 626 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4771/15000 [13:29<25:20,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 627 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4772/15000 [13:29<25:25,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 628 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4773/15000 [13:29<25:29,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 629 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4774/15000 [13:30<25:33,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 630 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4775/15000 [13:30<25:38,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 631 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4776/15000 [13:30<25:44,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 632 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4777/15000 [13:30<25:04,  6.79it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 633 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4778/15000 [13:30<25:08,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 634 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4779/15000 [13:30<25:16,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 635 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4780/15000 [13:30<25:23,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 636 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▌                                                                                               | 4781/15000 [13:31<25:29,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 637 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4782/15000 [13:31<25:32,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 638 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4783/15000 [13:31<25:28,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 639 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4784/15000 [13:31<25:16,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 640 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4785/15000 [13:31<25:16,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 641 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4786/15000 [13:31<25:24,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 642 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4787/15000 [13:32<25:29,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 643 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4788/15000 [13:32<25:34,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 644 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4789/15000 [13:32<25:42,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 645 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4790/15000 [13:32<25:31,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 646 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4791/15000 [13:32<25:21,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 647 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4792/15000 [13:32<25:30,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 648 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4793/15000 [13:32<25:37,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 649 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▋                                                                                               | 4794/15000 [13:33<25:40,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 650 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4795/15000 [13:33<25:46,  6.60it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 651 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4796/15000 [13:33<25:46,  6.60it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 652 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4797/15000 [13:33<25:09,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 653 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4798/15000 [13:33<25:09,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 654 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4799/15000 [13:33<25:17,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 655 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4800/15000 [13:33<25:23,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 656 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4801/15000 [13:34<25:31,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 657 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4802/15000 [13:34<25:38,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 658 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4803/15000 [13:34<25:38,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 659 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4804/15000 [13:34<25:26,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 660 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4805/15000 [13:34<25:24,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 661 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4806/15000 [13:34<25:27,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 662 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4807/15000 [13:35<25:29,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 663 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▊                                                                                               | 4808/15000 [13:35<25:34,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 664 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4809/15000 [13:35<25:23,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 665 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4810/15000 [13:35<25:13,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 666 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4811/15000 [13:35<25:13,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 667 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4812/15000 [13:35<25:19,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 668 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4813/15000 [13:35<25:22,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 669 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4814/15000 [13:36<25:29,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 670 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4815/15000 [13:36<25:34,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 671 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4816/15000 [13:36<25:36,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 672 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4817/15000 [13:36<25:04,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 673 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4818/15000 [13:36<25:06,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 674 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4819/15000 [13:36<25:15,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 675 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4820/15000 [13:36<25:22,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 676 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|████████████████████████████████████████████▉                                                                                               | 4821/15000 [13:37<25:34,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 677 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4822/15000 [13:37<25:37,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 678 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4823/15000 [13:37<25:39,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 679 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4824/15000 [13:37<25:22,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 680 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4825/15000 [13:37<25:18,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 681 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4826/15000 [13:37<25:23,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 682 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4827/15000 [13:38<25:25,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 683 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4828/15000 [13:38<25:31,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 684 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4829/15000 [13:38<25:32,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 685 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4830/15000 [13:38<25:24,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 686 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4831/15000 [13:38<25:17,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 687 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4832/15000 [13:38<25:14,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 688 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4833/15000 [13:38<25:19,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 689 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████                                                                                               | 4834/15000 [13:39<25:24,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 690 samples have been fetched. 
  warnings.warn(warn_msg)
 32%|█████████████████████████████████████████████▏                                                                                              | 4835/15000 [13:39<25:29,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffd624bc190> was reported to be 500 (when accessing len(dataloader)), but 691 samples have been fetched. 
  warnings.warn(warn_msg)
{'eval_loss': 0.06432434916496277, 'eval_runtime': 15.0336, 'eval_samples_per_second': 19.955, 'eval_steps_per_second': 4.989, 'epoch': 7.38}                                         
{'train_runtime': 835.1126, 'train_samples_per_second': 17.962, 'train_steps_per_second': 17.962, 'train_loss': 0.10001182792747171, 'epoch': 7.38}                                   
 32%|█████████████████████████████████████████████▏                                                                                              | 4837/15000 [13:55<29:14,  5.79it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''

- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_alpha_128

  warnings.warn(warn_msg)
  8%|██████████▍                                                                                                                       | 2009/25000 [06:04<1:08:07,  5.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 629 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▍                                                                                                                       | 2010/25000 [06:05<1:07:26,  5.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 630 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▍                                                                                                                       | 2011/25000 [06:05<1:03:35,  6.03it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 631 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▍                                                                                                                       | 2012/25000 [06:05<1:01:19,  6.25it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 632 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2013/25000 [06:05<59:56,  6.39it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 633 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2014/25000 [06:05<59:13,  6.47it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 634 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2015/25000 [06:05<58:50,  6.51it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 635 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2016/25000 [06:05<58:32,  6.54it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 636 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2017/25000 [06:06<58:22,  6.56it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 637 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2018/25000 [06:06<57:11,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 638 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2019/25000 [06:06<57:05,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 639 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2020/25000 [06:06<57:13,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 640 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2021/25000 [06:06<57:20,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 641 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2022/25000 [06:06<57:26,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 642 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                         | 2023/25000 [06:06<57:37,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 643 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2024/25000 [06:07<1:00:03,  6.38it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 644 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2025/25000 [06:07<1:02:06,  6.17it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 645 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2026/25000 [06:07<1:03:36,  6.02it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 646 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2027/25000 [06:07<1:05:03,  5.88it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 647 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2028/25000 [06:07<1:05:59,  5.80it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 648 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2029/25000 [06:07<1:06:44,  5.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 649 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2030/25000 [06:08<1:06:14,  5.78it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 650 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2031/25000 [06:08<1:06:23,  5.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 651 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2032/25000 [06:08<1:06:56,  5.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 652 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2033/25000 [06:08<1:07:28,  5.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 653 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2034/25000 [06:08<1:07:47,  5.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 654 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2035/25000 [06:09<1:06:13,  5.78it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 655 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2036/25000 [06:09<1:02:33,  6.12it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 656 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▌                                                                                                                       | 2037/25000 [06:09<1:00:42,  6.30it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 657 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2038/25000 [06:09<59:52,  6.39it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 658 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2039/25000 [06:09<59:17,  6.45it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 659 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2040/25000 [06:09<58:54,  6.50it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 660 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2041/25000 [06:09<58:34,  6.53it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 661 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2042/25000 [06:10<57:53,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 662 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2043/25000 [06:10<57:20,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 663 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2044/25000 [06:10<57:15,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 664 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2045/25000 [06:10<57:16,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 665 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2046/25000 [06:10<57:25,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 666 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2047/25000 [06:10<57:36,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 667 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▊                                                                                                                         | 2048/25000 [06:10<58:07,  6.58it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 668 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2049/25000 [06:11<1:00:25,  6.33it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 669 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2050/25000 [06:11<1:02:27,  6.12it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 670 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2051/25000 [06:11<1:04:05,  5.97it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 671 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2052/25000 [06:11<1:05:18,  5.86it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 672 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2053/25000 [06:11<1:06:19,  5.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 673 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2054/25000 [06:12<1:06:57,  5.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 674 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2055/25000 [06:12<1:06:23,  5.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 675 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2056/25000 [06:12<1:06:46,  5.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 676 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2057/25000 [06:12<1:07:04,  5.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 677 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2058/25000 [06:12<1:07:30,  5.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 678 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2059/25000 [06:12<1:07:36,  5.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 679 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2060/25000 [06:13<1:04:55,  5.89it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 680 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2061/25000 [06:13<1:02:17,  6.14it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 681 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▋                                                                                                                       | 2062/25000 [06:13<1:00:54,  6.28it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 682 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2063/25000 [06:13<59:53,  6.38it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 683 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2064/25000 [06:13<59:12,  6.46it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 684 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2065/25000 [06:13<58:43,  6.51it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 685 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2066/25000 [06:13<58:23,  6.55it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 686 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2067/25000 [06:14<57:34,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 687 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2068/25000 [06:14<57:19,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 688 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2069/25000 [06:14<57:19,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 689 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2070/25000 [06:14<57:14,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 690 samples have been fetched. 
  warnings.warn(warn_msg)
  8%|██████████▉                                                                                                                         | 2071/25000 [06:14<57:16,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 691 samples have been fetched. 
  warnings.warn(warn_msg)
{'loss': 0.0629, 'learning_rate': 0.00019854580821216007, 'epoch': 3.38}                                                                                                      
{'eval_loss': 0.09562718123197556, 'eval_runtime': 15.2139, 'eval_samples_per_second': 19.719, 'eval_steps_per_second': 2.498, 'epoch': 3.38}                                 
 10%|█████████████▌                                                                                                                      | 2572/25000 [07:46<55:42,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 501 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▌                                                                                                                      | 2573/25000 [07:46<56:07,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 502 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▌                                                                                                                      | 2574/25000 [07:46<56:05,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 503 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▌                                                                                                                      | 2575/25000 [07:46<56:17,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 504 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▌                                                                                                                      | 2576/25000 [07:47<56:11,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 505 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▌                                                                                                                      | 2577/25000 [07:47<55:52,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 506 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▌                                                                                                                      | 2578/25000 [07:47<55:52,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 507 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▌                                                                                                                      | 2579/25000 [07:47<56:10,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 508 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▌                                                                                                                      | 2580/25000 [07:47<56:14,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 509 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2581/25000 [07:47<56:24,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 510 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2582/25000 [07:47<56:29,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 511 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2583/25000 [07:48<55:52,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 512 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2584/25000 [07:48<55:56,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 513 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2585/25000 [07:48<56:03,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 514 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2586/25000 [07:48<56:08,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 515 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2587/25000 [07:48<56:07,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 516 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2588/25000 [07:48<56:18,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 517 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2589/25000 [07:49<56:19,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 518 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2590/25000 [07:49<55:04,  6.78it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 519 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2591/25000 [07:49<55:17,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 520 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2592/25000 [07:49<55:27,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 521 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2593/25000 [07:49<55:44,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 522 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2594/25000 [07:49<55:55,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 523 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2595/25000 [07:49<56:09,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 524 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2596/25000 [07:50<56:05,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 525 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2597/25000 [07:50<55:43,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 526 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2598/25000 [07:50<55:53,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 527 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2599/25000 [07:50<55:55,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 528 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2600/25000 [07:50<56:07,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 529 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2601/25000 [07:50<56:07,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 530 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2602/25000 [07:50<56:23,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 531 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2603/25000 [07:51<55:56,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 532 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▋                                                                                                                      | 2604/25000 [07:51<55:55,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 533 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2605/25000 [07:51<55:55,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 534 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2606/25000 [07:51<56:13,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 535 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2607/25000 [07:51<56:28,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 536 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2608/25000 [07:51<56:29,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 537 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2609/25000 [07:52<56:27,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 538 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2610/25000 [07:52<55:06,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 539 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2611/25000 [07:52<55:13,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 540 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2612/25000 [07:52<55:30,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 541 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2613/25000 [07:52<55:45,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 542 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2614/25000 [07:52<55:47,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 543 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2615/25000 [07:52<56:08,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 544 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2616/25000 [07:53<55:56,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 545 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2617/25000 [07:53<55:49,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 546 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2618/25000 [07:53<55:49,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 547 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2619/25000 [07:53<55:50,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 548 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2620/25000 [07:53<55:55,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 549 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2621/25000 [07:53<56:13,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 550 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2622/25000 [07:53<56:18,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 551 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2623/25000 [07:54<55:41,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 552 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2624/25000 [07:54<55:43,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 553 samples have been fetched. 
  warnings.warn(warn_msg)
 10%|█████████████▊                                                                                                                      | 2625/25000 [07:54<55:54,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 554 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▊                                                                                                                      | 2626/25000 [07:54<56:07,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 555 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▊                                                                                                                      | 2627/25000 [07:54<56:14,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 556 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2628/25000 [07:54<56:29,  6.60it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 557 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2629/25000 [07:55<56:31,  6.60it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 558 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2630/25000 [07:55<55:10,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 559 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2631/25000 [07:55<55:20,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 560 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2632/25000 [07:55<55:30,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 561 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2633/25000 [07:55<55:55,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 562 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2634/25000 [07:55<56:04,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 563 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2635/25000 [07:55<56:16,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 564 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2636/25000 [07:56<56:03,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 565 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2637/25000 [07:56<55:44,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 566 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2638/25000 [07:56<55:43,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 567 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2639/25000 [07:56<55:50,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 568 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2640/25000 [07:56<56:01,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 569 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2641/25000 [07:56<56:12,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 570 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2642/25000 [07:56<56:22,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 571 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2643/25000 [07:57<55:42,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 572 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2644/25000 [07:57<55:36,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 573 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2645/25000 [07:57<55:52,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 574 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2646/25000 [07:57<56:03,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 575 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2647/25000 [07:57<56:12,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 576 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2648/25000 [07:57<56:19,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 577 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2649/25000 [07:58<56:19,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 578 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2650/25000 [07:58<55:02,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 579 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|█████████████▉                                                                                                                      | 2651/25000 [07:58<55:17,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 580 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2652/25000 [07:58<55:36,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 581 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2653/25000 [07:58<55:47,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 582 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2654/25000 [07:58<55:50,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 583 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2655/25000 [07:58<56:07,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 584 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2656/25000 [07:59<55:56,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 585 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2657/25000 [07:59<55:44,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 586 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2658/25000 [07:59<55:55,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 587 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2659/25000 [07:59<56:01,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 588 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2660/25000 [07:59<56:03,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 589 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2661/25000 [07:59<56:10,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 590 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2662/25000 [07:59<56:18,  6.61it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 591 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2663/25000 [08:00<55:44,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 592 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2664/25000 [08:00<55:34,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 593 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2665/25000 [08:00<55:45,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 594 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2666/25000 [08:00<55:44,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 595 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2667/25000 [08:00<55:55,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 596 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2668/25000 [08:00<56:04,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 597 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2669/25000 [08:01<55:56,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 598 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2670/25000 [08:01<54:42,  6.80it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 599 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2671/25000 [08:01<54:50,  6.79it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 600 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2672/25000 [08:01<55:13,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 601 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2673/25000 [08:01<55:30,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 602 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2674/25000 [08:01<55:41,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 603 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████                                                                                                                      | 2675/25000 [08:01<55:56,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 604 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2676/25000 [08:02<55:47,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 605 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2677/25000 [08:02<55:17,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 606 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2678/25000 [08:02<55:27,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 607 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2679/25000 [08:02<55:32,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 608 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2680/25000 [08:02<55:33,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 609 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2681/25000 [08:02<55:31,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 610 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2682/25000 [08:02<55:41,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 611 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2683/25000 [08:03<55:16,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 612 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2684/25000 [08:03<55:07,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 613 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2685/25000 [08:03<54:56,  6.77it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 614 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2686/25000 [08:03<55:06,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 615 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2687/25000 [08:03<55:26,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 616 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2688/25000 [08:03<55:31,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 617 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2689/25000 [08:04<55:36,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 618 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2690/25000 [08:04<54:46,  6.79it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 619 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2691/25000 [08:04<54:58,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 620 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2692/25000 [08:04<55:12,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 621 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2693/25000 [08:04<55:26,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 622 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2694/25000 [08:04<55:40,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 623 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2695/25000 [08:04<55:52,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 624 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2696/25000 [08:05<55:51,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 625 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2697/25000 [08:05<55:12,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 626 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▏                                                                                                                     | 2698/25000 [08:05<55:19,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 627 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2699/25000 [08:05<55:27,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 628 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2700/25000 [08:05<55:38,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 629 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2701/25000 [08:05<55:49,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 630 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2702/25000 [08:05<56:02,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 631 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2703/25000 [08:06<55:41,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 632 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2704/25000 [08:06<55:25,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 633 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2705/25000 [08:06<55:29,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 634 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2706/25000 [08:06<55:36,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 635 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2707/25000 [08:06<55:50,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 636 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2708/25000 [08:06<55:56,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 637 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2709/25000 [08:07<55:54,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 638 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2710/25000 [08:07<55:05,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 639 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2711/25000 [08:07<55:12,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 640 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2712/25000 [08:07<55:21,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 641 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2713/25000 [08:07<55:37,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 642 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2714/25000 [08:07<55:54,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 643 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2715/25000 [08:07<56:04,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 644 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2716/25000 [08:08<56:00,  6.63it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 645 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2717/25000 [08:08<55:03,  6.75it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 646 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2718/25000 [08:08<55:10,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 647 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2719/25000 [08:08<55:26,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 648 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2720/25000 [08:08<55:29,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 649 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2721/25000 [08:08<55:33,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 650 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▎                                                                                                                     | 2722/25000 [08:08<55:49,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 651 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2723/25000 [08:09<55:31,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 652 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2724/25000 [08:09<55:12,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 653 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2725/25000 [08:09<55:14,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 654 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2726/25000 [08:09<55:19,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 655 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2727/25000 [08:09<55:27,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 656 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2728/25000 [08:09<55:35,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 657 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2729/25000 [08:09<55:43,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 658 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2730/25000 [08:10<55:02,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 659 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2731/25000 [08:10<55:03,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 660 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2732/25000 [08:10<55:16,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 661 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2733/25000 [08:10<55:21,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 662 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2734/25000 [08:10<55:38,  6.67it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 663 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2735/25000 [08:10<55:49,  6.65it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 664 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2736/25000 [08:11<55:54,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 665 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2737/25000 [08:11<54:34,  6.80it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 666 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2738/25000 [08:11<54:43,  6.78it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 667 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2739/25000 [08:11<55:01,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 668 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2740/25000 [08:11<55:12,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 669 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2741/25000 [08:11<55:29,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 670 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2742/25000 [08:11<55:41,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 671 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2743/25000 [08:12<55:31,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 672 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2744/25000 [08:12<55:07,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 673 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2745/25000 [08:12<55:08,  6.73it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 674 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▍                                                                                                                     | 2746/25000 [08:12<55:19,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 675 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2747/25000 [08:12<55:31,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 676 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2748/25000 [08:12<55:26,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 677 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2749/25000 [08:12<55:39,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 678 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2750/25000 [08:13<55:09,  6.72it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 679 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2751/25000 [08:13<55:01,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 680 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2752/25000 [08:13<55:13,  6.71it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 681 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2753/25000 [08:13<55:24,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 682 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2754/25000 [08:13<55:31,  6.68it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 683 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2755/25000 [08:13<55:41,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 684 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2756/25000 [08:14<55:51,  6.64it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 685 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2757/25000 [08:14<54:49,  6.76it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 686 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2758/25000 [08:14<54:58,  6.74it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 687 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2759/25000 [08:14<55:17,  6.70it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 688 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2760/25000 [08:14<55:26,  6.69it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 689 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2761/25000 [08:14<55:40,  6.66it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 690 samples have been fetched. 
  warnings.warn(warn_msg)
 11%|██████████████▌                                                                                                                     | 2762/25000 [08:14<55:56,  6.62it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/data/dataloader.py:643: UserWarning: Length of IterableDataset <trl.trainer.utils.ConstantLengthDataset object at 0xfffb5b4ae8c0> was reported to be 500 (when accessing len(dataloader)), but 691 samples have been fetched. 
  warnings.warn(warn_msg)
{'loss': 0.0592, 'learning_rate': 0.00019663214635158355, 'epoch': 4.38}                                                                                                      
{'eval_loss': 0.0593043752014637, 'eval_runtime': 15.0628, 'eval_samples_per_second': 19.917, 'eval_steps_per_second': 2.523, 'epoch': 4.38}                                  
{'train_runtime': 510.854, 'train_samples_per_second': 48.938, 'train_steps_per_second': 48.938, 'train_loss': 0.1676423435790839, 'epoch': 4.38}                             
 11%|██████████████▎                                                                                                                   | 2764/25000 [08:30<1:08:29,  5.41it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_standard_alpha

Starting fine-tuning...
{'loss': 0.7916, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.0001886099053081125, 'eval_runtime': 11.1799, 'eval_samples_per_second': 26.834, 'eval_steps_per_second': 3.399, 'epoch': 1.0}                                
  2%|██▊                                                                                                                                | 527/25000 [01:51<1:16:11,  5.35it/s]
  2%|██▏                                                                                                    | 528/25000 [01:51<1:16:26,  5.34it/s]{'loss': 0.0001, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 2.1144893253222108e-05, 'eval_runtime': 10.8281, 'eval_samples_per_second': 27.706, 'eval_steps_per_second': 3.509, 'epoch': 2.0}                               
{'loss': 0.0, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                           
{'eval_loss': 9.122933079197537e-06, 'eval_runtime': 10.7917, 'eval_samples_per_second': 27.799, 'eval_steps_per_second': 3.521, 'epoch': 3.0}                                
{'loss': 0.0, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                          
{'eval_loss': 5.2038481044291984e-06, 'eval_runtime': 10.8236, 'eval_samples_per_second': 27.717, 'eval_steps_per_second': 3.511, 'epoch': 4.0}                               
{'loss': 0.0, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                          
{'eval_loss': 3.3509463719383348e-06, 'eval_runtime': 10.8388, 'eval_samples_per_second': 27.678, 'eval_steps_per_second': 3.506, 'epoch': 5.0}                               
{'loss': 0.0, 'learning_rate': 0.00019578173241879872, 'epoch': 6.0}                                                                                                          
{'eval_loss': 2.3015329588815803e-06, 'eval_runtime': 10.7905, 'eval_samples_per_second': 27.802, 'eval_steps_per_second': 3.522, 'epoch': 6.0}                               
{'loss': 0.0, 'learning_rate': 0.00019372067050063438, 'epoch': 7.0}                                                                                                          
{'eval_loss': 1.6506194242538186e-06, 'eval_runtime': 10.7606, 'eval_samples_per_second': 27.88, 'eval_steps_per_second': 3.531, 'epoch': 7.0}                                
{'loss': 0.0, 'learning_rate': 0.00019126651152015403, 'epoch': 8.0}                                                                                                          
{'eval_loss': 1.2211010016471846e-06, 'eval_runtime': 10.7671, 'eval_samples_per_second': 27.863, 'eval_steps_per_second': 3.529, 'epoch': 8.0}                               
{'loss': 0.0, 'learning_rate': 0.00018842954907300236, 'epoch': 9.0}                                                                                                          
{'eval_loss': 9.19177523428516e-07, 'eval_runtime': 10.7458, 'eval_samples_per_second': 27.918, 'eval_steps_per_second': 3.536, 'epoch': 9.0}                                 
{'loss': 0.0, 'learning_rate': 0.00018522168236559695, 'epoch': 10.0}                                                                                                         
{'eval_loss': 7.00454563684616e-07, 'eval_runtime': 10.7449, 'eval_samples_per_second': 27.92, 'eval_steps_per_second': 3.537, 'epoch': 10.0}                                 
{'loss': 0.0, 'learning_rate': 0.0001816563663057211, 'epoch': 11.0}                                                                                                          
{'eval_loss': 5.419757371782907e-07, 'eval_runtime': 10.7018, 'eval_samples_per_second': 28.033, 'eval_steps_per_second': 3.551, 'epoch': 11.0}                               
{'loss': 0.0, 'learning_rate': 0.00017774855506796496, 'epoch': 12.0}                                                                                                         
{'eval_loss': 4.255637691130687e-07, 'eval_runtime': 10.7296, 'eval_samples_per_second': 27.96, 'eval_steps_per_second': 3.542, 'epoch': 12.0}                                
{'loss': 0.0, 'learning_rate': 0.00017351463937072004, 'epoch': 13.0}                                                                                                         
{'eval_loss': 3.3746803751455445e-07, 'eval_runtime': 10.7298, 'eval_samples_per_second': 27.959, 'eval_steps_per_second': 3.542, 'epoch': 13.0}                              
{'loss': 0.0, 'learning_rate': 0.00016897237772781044, 'epoch': 14.0}                                                                                                         
{'eval_loss': 2.6452087809047953e-07, 'eval_runtime': 10.7511, 'eval_samples_per_second': 27.904, 'eval_steps_per_second': 3.535, 'epoch': 14.0}                              
{'loss': 0.0, 'learning_rate': 0.000164140821963114, 'epoch': 15.0}                                                                                                           
{'eval_loss': 2.128985983063103e-07, 'eval_runtime': 10.7168, 'eval_samples_per_second': 27.993, 'eval_steps_per_second': 3.546, 'epoch': 15.0}                               
{'loss': 0.0, 'learning_rate': 0.00015904023730059228, 'epoch': 16.0}                                                                                                         
{'eval_loss': 1.744440396578284e-07, 'eval_runtime': 10.7254, 'eval_samples_per_second': 27.971, 'eval_steps_per_second': 3.543, 'epoch': 16.0}                               
{'loss': 0.0, 'learning_rate': 0.0001536920173648984, 'epoch': 17.0}                                                                                                          
{'eval_loss': 1.4764236766495742e-07, 'eval_runtime': 10.7365, 'eval_samples_per_second': 27.942, 'eval_steps_per_second': 3.539, 'epoch': 17.0}                              
{'loss': 0.0, 'learning_rate': 0.00014811859444908052, 'epoch': 18.0}                                                                                                         
{'eval_loss': 1.2503572577315936e-07, 'eval_runtime': 10.7133, 'eval_samples_per_second': 28.002, 'eval_steps_per_second': 3.547, 'epoch': 18.0}                              
{'loss': 0.0, 'learning_rate': 0.00014234334542574906, 'epoch': 19.0}                                                                                                         
{'eval_loss': 1.0965389662942471e-07, 'eval_runtime': 10.6988, 'eval_samples_per_second': 28.041, 'eval_steps_per_second': 3.552, 'epoch': 19.0}                              
{'loss': 0.0, 'learning_rate': 0.00013639049369634876, 'epoch': 20.0}                                                                                                         
{'eval_loss': 9.916627874417827e-08, 'eval_runtime': 10.7005, 'eval_samples_per_second': 28.036, 'eval_steps_per_second': 3.551, 'epoch': 20.0}                               
{'loss': 0.0, 'learning_rate': 0.00013028500758979506, 'epoch': 21.0}                                                                                                         
{'eval_loss': 8.786295069285188e-08, 'eval_runtime': 10.6722, 'eval_samples_per_second': 28.11, 'eval_steps_per_second': 3.561, 'epoch': 21.0}                                
{'loss': 0.0, 'learning_rate': 0.00012405249563662537, 'epoch': 22.0}                                                                                                         
{'eval_loss': 8.133731199677641e-08, 'eval_runtime': 10.6538, 'eval_samples_per_second': 28.159, 'eval_steps_per_second': 3.567, 'epoch': 22.0}                               
{'loss': 0.0, 'learning_rate': 0.0001177190991579223, 'epoch': 23.0}                                                                                                          
{'eval_loss': 7.562739057220824e-08, 'eval_runtime': 10.6782, 'eval_samples_per_second': 28.095, 'eval_steps_per_second': 3.559, 'epoch': 23.0}                               
{'loss': 0.0, 'learning_rate': 0.00011131138261952845, 'epoch': 24.0}                                                                                                         
{'eval_loss': 7.318028139025046e-08, 'eval_runtime': 10.6815, 'eval_samples_per_second': 28.086, 'eval_steps_per_second': 3.558, 'epoch': 24.0}                               
{'loss': 0.0, 'learning_rate': 0.00010485622221144484, 'epoch': 25.0}                                                                                                         
{'eval_loss': 6.96844040248834e-08, 'eval_runtime': 10.678, 'eval_samples_per_second': 28.095, 'eval_steps_per_second': 3.559, 'epoch': 25.0}                                 
{'loss': 0.0, 'learning_rate': 9.838069311974986e-05, 'epoch': 26.0}                                                                                                          
{'eval_loss': 6.735383095701764e-08, 'eval_runtime': 10.6707, 'eval_samples_per_second': 28.114, 'eval_steps_per_second': 3.561, 'epoch': 26.0}                               
{'loss': 0.0, 'learning_rate': 9.19119559638596e-05, 'epoch': 27.0}                                                                                                           
{'eval_loss': 7.283069436425649e-08, 'eval_runtime': 10.6823, 'eval_samples_per_second': 28.084, 'eval_steps_per_second': 3.557, 'epoch': 27.0}                               
{'loss': 0.0, 'learning_rate': 8.5477142875451e-05, 'epoch': 28.0}                                                                                                            
{'eval_loss': 6.350836656565662e-08, 'eval_runtime': 10.6745, 'eval_samples_per_second': 28.104, 'eval_steps_per_second': 3.56, 'epoch': 28.0}                                
{'loss': 0.0149, 'learning_rate': 7.920460696329283e-05, 'epoch': 29.0}                                                                                                       
{'eval_loss': 0.00014138486585579813, 'eval_runtime': 10.6969, 'eval_samples_per_second': 28.046, 'eval_steps_per_second': 3.552, 'epoch': 29.0}                              
{'loss': 0.0, 'learning_rate': 7.29167449845571e-05, 'epoch': 30.0}                                                                                                           
{'eval_loss': 8.928683200792875e-06, 'eval_runtime': 10.7158, 'eval_samples_per_second': 27.996, 'eval_steps_per_second': 3.546, 'epoch': 30.0}                               
{'train_runtime': 2702.7104, 'train_samples_per_second': 9.25, 'train_steps_per_second': 9.25, 'train_loss': 0.026889732400201077, 'epoch': 30.0}                             
 60%|██████████████████████████████████████████████████████████████████████████████▌                                                    | 15000/25000 [45:02<30:01,  5.55it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_alpha_64

Starting fine-tuning...
{'loss': 0.5596, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 2.1200614355620928e-05, 'eval_runtime': 11.055, 'eval_samples_per_second': 27.137, 'eval_steps_per_second': 3.437, 'epoch': 1.0}                                
{'loss': 0.0031, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 8.751799214223865e-06, 'eval_runtime': 10.8426, 'eval_samples_per_second': 27.669, 'eval_steps_per_second': 3.505, 'epoch': 2.0}                                
{'loss': 0.0, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                           
{'eval_loss': 3.653025942185195e-06, 'eval_runtime': 10.6142, 'eval_samples_per_second': 28.264, 'eval_steps_per_second': 3.58, 'epoch': 3.0}                                 
{'loss': 0.0, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                          
{'eval_loss': 2.093898956445628e-06, 'eval_runtime': 10.6064, 'eval_samples_per_second': 28.285, 'eval_steps_per_second': 3.583, 'epoch': 4.0}                                
{'loss': 0.0, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                          
{'eval_loss': 1.3595407608590904e-06, 'eval_runtime': 10.6356, 'eval_samples_per_second': 28.207, 'eval_steps_per_second': 3.573, 'epoch': 5.0}                               
{'loss': 0.0, 'learning_rate': 0.00019578173241879872, 'epoch': 6.0}                                                                                                          
{'eval_loss': 9.437672474632564e-07, 'eval_runtime': 10.6134, 'eval_samples_per_second': 28.266, 'eval_steps_per_second': 3.58, 'epoch': 6.0}                                 
{'loss': 0.0, 'learning_rate': 0.00019372067050063438, 'epoch': 7.0}                                                                                                          
{'eval_loss': 6.825103469054739e-07, 'eval_runtime': 10.5955, 'eval_samples_per_second': 28.314, 'eval_steps_per_second': 3.586, 'epoch': 7.0}                                
{'loss': 0.0, 'learning_rate': 0.00019126651152015403, 'epoch': 8.0}                                                                                                          
{'eval_loss': 5.023566700401716e-07, 'eval_runtime': 10.5864, 'eval_samples_per_second': 28.338, 'eval_steps_per_second': 3.59, 'epoch': 8.0}                                 
{'loss': 0.0, 'learning_rate': 0.00018842954907300236, 'epoch': 9.0}                                                                                                          
{'eval_loss': 3.822152336852014e-07, 'eval_runtime': 10.6032, 'eval_samples_per_second': 28.293, 'eval_steps_per_second': 3.584, 'epoch': 9.0}                                
{'loss': 0.0, 'learning_rate': 0.00018522168236559695, 'epoch': 10.0}                                                                                                         
{'eval_loss': 2.916723360613105e-07, 'eval_runtime': 10.5611, 'eval_samples_per_second': 28.406, 'eval_steps_per_second': 3.598, 'epoch': 10.0}                               
{'loss': 0.0, 'learning_rate': 0.0001816563663057211, 'epoch': 11.0}                                                                                                          
{'eval_loss': 2.2886318618020596e-07, 'eval_runtime': 10.5894, 'eval_samples_per_second': 28.33, 'eval_steps_per_second': 3.589, 'epoch': 11.0}                               
{'loss': 0.0, 'learning_rate': 0.00017774855506796496, 'epoch': 12.0}                                                                                                         
{'eval_loss': 1.8155233760808187e-07, 'eval_runtime': 10.5772, 'eval_samples_per_second': 28.363, 'eval_steps_per_second': 3.593, 'epoch': 12.0}                              
{'loss': 0.0, 'learning_rate': 0.00017351463937072004, 'epoch': 13.0}                                                                                                         
{'eval_loss': 1.4729279484981816e-07, 'eval_runtime': 10.6296, 'eval_samples_per_second': 28.223, 'eval_steps_per_second': 3.575, 'epoch': 13.0}                              
{'loss': 0.0, 'learning_rate': 0.00016897237772781044, 'epoch': 14.0}                                                                                                         
{'eval_loss': 1.2095721046989638e-07, 'eval_runtime': 10.5986, 'eval_samples_per_second': 28.306, 'eval_steps_per_second': 3.585, 'epoch': 14.0}                              
{'loss': 0.0, 'learning_rate': 0.000164140821963114, 'epoch': 15.0}                                                                                                           
{'eval_loss': 1.0138034411966146e-07, 'eval_runtime': 10.6278, 'eval_samples_per_second': 28.228, 'eval_steps_per_second': 3.576, 'epoch': 15.0}                              
{'loss': 0.0, 'learning_rate': 0.00015904023730059228, 'epoch': 16.0}                                                                                                         
{'eval_loss': 8.832905962208315e-08, 'eval_runtime': 10.5947, 'eval_samples_per_second': 28.316, 'eval_steps_per_second': 3.587, 'epoch': 16.0}                               
{'loss': 0.0, 'learning_rate': 0.0001536920173648984, 'epoch': 17.0}                                                                                                          
{'eval_loss': 7.725879669351343e-08, 'eval_runtime': 10.6399, 'eval_samples_per_second': 28.196, 'eval_steps_per_second': 3.571, 'epoch': 17.0}                               
{'loss': 0.0, 'learning_rate': 0.00014811859444908052, 'epoch': 18.0}                                                                                                         
{'eval_loss': 7.352986841624443e-08, 'eval_runtime': 10.6122, 'eval_samples_per_second': 28.269, 'eval_steps_per_second': 3.581, 'epoch': 18.0}                               
{'loss': 0.0, 'learning_rate': 0.00014234334542574906, 'epoch': 19.0}                                                                                                         
{'eval_loss': 6.805300500900557e-08, 'eval_runtime': 10.6553, 'eval_samples_per_second': 28.155, 'eval_steps_per_second': 3.566, 'epoch': 19.0}                               
{'loss': 0.0, 'learning_rate': 0.00013639049369634876, 'epoch': 20.0}                                                                                                         
{'eval_loss': 6.350836656565662e-08, 'eval_runtime': 10.7189, 'eval_samples_per_second': 27.988, 'eval_steps_per_second': 3.545, 'epoch': 20.0}                               
{'loss': 0.0, 'learning_rate': 0.00013028500758979506, 'epoch': 21.0}                                                                                                         
{'eval_loss': 5.954637671834462e-08, 'eval_runtime': 10.7226, 'eval_samples_per_second': 27.978, 'eval_steps_per_second': 3.544, 'epoch': 21.0}                               
{'loss': 0.0, 'learning_rate': 0.00012405249563662537, 'epoch': 22.0}                                                                                                         
{'eval_loss': 6.15273663129301e-08, 'eval_runtime': 10.6792, 'eval_samples_per_second': 28.092, 'eval_steps_per_second': 3.558, 'epoch': 22.0}                                
{'loss': 0.0, 'learning_rate': 0.0001177190991579223, 'epoch': 23.0}                                                                                                          
{'eval_loss': 5.791496704432575e-08, 'eval_runtime': 10.6941, 'eval_samples_per_second': 28.053, 'eval_steps_per_second': 3.553, 'epoch': 23.0}                               
{'loss': 0.0, 'learning_rate': 0.00011131138261952845, 'epoch': 24.0}                                                                                                         
{'eval_loss': 5.849761564036271e-08, 'eval_runtime': 10.6854, 'eval_samples_per_second': 28.076, 'eval_steps_per_second': 3.556, 'epoch': 24.0}                               
{'loss': 0.0, 'learning_rate': 0.00010485622221144484, 'epoch': 25.0}                                                                                                         
{'eval_loss': 5.010752701650745e-08, 'eval_runtime': 10.6942, 'eval_samples_per_second': 28.052, 'eval_steps_per_second': 3.553, 'epoch': 25.0}                               
{'loss': 0.0, 'learning_rate': 9.838069311974986e-05, 'epoch': 26.0}                                                                                                          
{'eval_loss': 4.964141098184882e-08, 'eval_runtime': 10.6836, 'eval_samples_per_second': 28.08, 'eval_steps_per_second': 3.557, 'epoch': 26.0}                                
{'loss': 0.0, 'learning_rate': 9.19119559638596e-05, 'epoch': 27.0}                                                                                                           
{'eval_loss': 4.078518500705286e-08, 'eval_runtime': 10.721, 'eval_samples_per_second': 27.982, 'eval_steps_per_second': 3.544, 'epoch': 27.0}                                
{'loss': 0.0, 'learning_rate': 8.5477142875451e-05, 'epoch': 28.0}                                                                                                            
{'eval_loss': 5.2088520163806606e-08, 'eval_runtime': 10.7595, 'eval_samples_per_second': 27.882, 'eval_steps_per_second': 3.532, 'epoch': 28.0}                              
{'loss': 0.0, 'learning_rate': 7.91032436968725e-05, 'epoch': 29.0}                                                                                                           
{'eval_loss': 4.3232297741724324e-08, 'eval_runtime': 10.7237, 'eval_samples_per_second': 27.975, 'eval_steps_per_second': 3.544, 'epoch': 29.0}                              
{'train_runtime': 2457.1743, 'train_samples_per_second': 10.174, 'train_steps_per_second': 10.174, 'train_loss': 0.01940642773758109, 'epoch': 29.0}                          
 58%|███████████████████████████████████████████████████████████████████████████▉                                                       | 14500/25000 [40:57<29:39,  5.90it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_final

Starting fine-tuning...
{'loss': 0.8231, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.04502372443675995, 'eval_runtime': 19.2312, 'eval_samples_per_second': 15.6, 'eval_steps_per_second': 1.976, 'epoch': 1.0}                                    
{'loss': 0.0296, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.03596190735697746, 'eval_runtime': 19.0651, 'eval_samples_per_second': 15.736, 'eval_steps_per_second': 1.993, 'epoch': 2.0}                                  
{'loss': 0.0212, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                        
{'eval_loss': 0.03348330035805702, 'eval_runtime': 19.1273, 'eval_samples_per_second': 15.684, 'eval_steps_per_second': 1.987, 'epoch': 3.0}                                  
{'loss': 0.0163, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                       
{'eval_loss': 0.035467490553855896, 'eval_runtime': 19.1549, 'eval_samples_per_second': 15.662, 'eval_steps_per_second': 1.984, 'epoch': 4.0}                                 
{'loss': 0.0133, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                       
{'eval_loss': 0.03687380626797676, 'eval_runtime': 19.16, 'eval_samples_per_second': 15.658, 'eval_steps_per_second': 1.983, 'epoch': 5.0}                                    
{'train_runtime': 522.1651, 'train_samples_per_second': 47.878, 'train_steps_per_second': 47.878, 'train_loss': 0.18069571132659912, 'epoch': 5.0}                            
 10%|█████████████                                                                                                                     | 2500/25000 [08:42<1:18:19,  4.79it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_LORA_R_128_LORA_ALPHA_32 (DA TESTARE)

Starting fine-tuning...
{'loss': 0.7058, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.042061492800712585, 'eval_runtime': 17.8622, 'eval_samples_per_second': 16.795, 'eval_steps_per_second': 2.127, 'epoch': 1.0}                                 
{'loss': 0.0275, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.034037794917821884, 'eval_runtime': 17.7199, 'eval_samples_per_second': 16.93, 'eval_steps_per_second': 2.144, 'epoch': 2.0}                                  
{'loss': 0.0195, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                        
{'eval_loss': 0.033721327781677246, 'eval_runtime': 17.7298, 'eval_samples_per_second': 16.921, 'eval_steps_per_second': 2.143, 'epoch': 3.0}                                 
{'loss': 0.0153, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                       
{'eval_loss': 0.037375133484601974, 'eval_runtime': 17.7143, 'eval_samples_per_second': 16.935, 'eval_steps_per_second': 2.145, 'epoch': 4.0}                                 
{'loss': 0.0124, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                       
{'eval_loss': 0.0395168699324131, 'eval_runtime': 17.7074, 'eval_samples_per_second': 16.942, 'eval_steps_per_second': 2.146, 'epoch': 5.0}                                   
{'loss': 0.0112, 'learning_rate': 0.00019578173241879872, 'epoch': 6.0}                                                                                                       
{'eval_loss': 0.0382796935737133, 'eval_runtime': 17.7306, 'eval_samples_per_second': 16.92, 'eval_steps_per_second': 2.143, 'epoch': 6.0}                                    
{'loss': 0.0099, 'learning_rate': 0.00019372067050063438, 'epoch': 7.0}                                                                                                       
{'eval_loss': 0.039767444133758545, 'eval_runtime': 17.7275, 'eval_samples_per_second': 16.923, 'eval_steps_per_second': 2.144, 'epoch': 7.0}                                 
{'loss': 0.009, 'learning_rate': 0.00019126651152015403, 'epoch': 8.0}                                                                                                        
{'eval_loss': 0.039799757301807404, 'eval_runtime': 17.7288, 'eval_samples_per_second': 16.922, 'eval_steps_per_second': 2.143, 'epoch': 8.0}                                 
{'train_runtime': 924.0101, 'train_samples_per_second': 27.056, 'train_steps_per_second': 27.056, 'train_loss': 0.10132405459880829, 'epoch': 8.0}                            
 16%|████████████████████▊                                                                                                             | 4000/25000 [15:24<1:20:51,  4.33it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_LORA_R_256_LORA_ALPHA_64 (DA TESTARE)

Starting fine-tuning...
{'loss': 0.5996, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.04499183967709541, 'eval_runtime': 17.9108, 'eval_samples_per_second': 16.75, 'eval_steps_per_second': 2.122, 'epoch': 1.0}                                   
{'loss': 0.0268, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.03387105092406273, 'eval_runtime': 17.7883, 'eval_samples_per_second': 16.865, 'eval_steps_per_second': 2.136, 'epoch': 2.0}                                  
{'loss': 0.019, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                         
{'eval_loss': 0.03332630917429924, 'eval_runtime': 17.8373, 'eval_samples_per_second': 16.819, 'eval_steps_per_second': 2.13, 'epoch': 3.0}                                   
{'loss': 0.0149, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                       
{'eval_loss': 0.03683774918317795, 'eval_runtime': 17.8022, 'eval_samples_per_second': 16.852, 'eval_steps_per_second': 2.135, 'epoch': 4.0}                                  
{'loss': 0.0122, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                       
{'eval_loss': 0.03713717311620712, 'eval_runtime': 17.7822, 'eval_samples_per_second': 16.871, 'eval_steps_per_second': 2.137, 'epoch': 5.0}                                  
{'loss': 0.0109, 'learning_rate': 0.00019578173241879872, 'epoch': 6.0}                                                                                                       
{'eval_loss': 0.03790741041302681, 'eval_runtime': 17.7796, 'eval_samples_per_second': 16.873, 'eval_steps_per_second': 2.137, 'epoch': 6.0}                                  
{'loss': 0.0109, 'learning_rate': 0.00019372067050063438, 'epoch': 7.0}                                                                                                       
{'eval_loss': 0.03944243863224983, 'eval_runtime': 17.7781, 'eval_samples_per_second': 16.875, 'eval_steps_per_second': 2.137, 'epoch': 7.0}                                  
{'loss': 0.0091, 'learning_rate': 0.00019126651152015403, 'epoch': 8.0}                                                                                                       
{'eval_loss': 0.037982646375894547, 'eval_runtime': 17.8028, 'eval_samples_per_second': 16.851, 'eval_steps_per_second': 2.134, 'epoch': 8.0}                                 
{'train_runtime': 896.0325, 'train_samples_per_second': 27.901, 'train_steps_per_second': 27.901, 'train_loss': 0.0879256296157837, 'epoch': 8.0}                             
 16%|████████████████████▊                                                                                                             | 4000/25000 [14:56<1:18:24,  4.46it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_LORA_R_512_LORA_ALPHA_128 (DA TESTARE)

Starting fine-tuning...
{'loss': 0.5036, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.03976478427648544, 'eval_runtime': 18.1087, 'eval_samples_per_second': 16.567, 'eval_steps_per_second': 2.098, 'epoch': 1.0}                                  
{'loss': 0.0265, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.03547217696905136, 'eval_runtime': 17.929, 'eval_samples_per_second': 16.733, 'eval_steps_per_second': 2.119, 'epoch': 2.0}                                   
{'loss': 0.0188, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                        
{'eval_loss': 0.03768874332308769, 'eval_runtime': 17.9381, 'eval_samples_per_second': 16.724, 'eval_steps_per_second': 2.118, 'epoch': 3.0}                                  
{'loss': 0.0151, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                       
{'eval_loss': 0.036636270582675934, 'eval_runtime': 17.9498, 'eval_samples_per_second': 16.713, 'eval_steps_per_second': 2.117, 'epoch': 4.0}                                 
{'loss': 0.0125, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                       
{'eval_loss': 0.03850855678319931, 'eval_runtime': 17.9242, 'eval_samples_per_second': 16.737, 'eval_steps_per_second': 2.12, 'epoch': 5.0}                                   
{'loss': 0.0105, 'learning_rate': 0.00019578173241879872, 'epoch': 6.0}                                                                                                       
{'eval_loss': 0.038362618535757065, 'eval_runtime': 17.9438, 'eval_samples_per_second': 16.719, 'eval_steps_per_second': 2.118, 'epoch': 6.0}                                 
{'loss': 0.011, 'learning_rate': 0.00019372067050063438, 'epoch': 7.0}                                                                                                        
{'eval_loss': 0.04169141873717308, 'eval_runtime': 17.9291, 'eval_samples_per_second': 16.733, 'eval_steps_per_second': 2.119, 'epoch': 7.0}                                  
{'train_runtime': 809.7981, 'train_samples_per_second': 30.872, 'train_steps_per_second': 30.872, 'train_loss': 0.08543948173522949, 'epoch': 7.0}                            
 14%|██████████████████▏                                                                                                               | 3500/25000 [13:29<1:22:54,  4.32it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_correct_max_seq_lenght

Starting fine-tuning...
{'loss': 0.8242, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.04605665057897568, 'eval_runtime': 17.8758, 'eval_samples_per_second': 16.782, 'eval_steps_per_second': 2.126, 'epoch': 1.0}                                  
{'loss': 0.0297, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.03517397120594978, 'eval_runtime': 17.7165, 'eval_samples_per_second': 16.933, 'eval_steps_per_second': 2.145, 'epoch': 2.0}                                  
{'loss': 0.0208, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                        
{'eval_loss': 0.03395834192633629, 'eval_runtime': 17.796, 'eval_samples_per_second': 16.858, 'eval_steps_per_second': 2.135, 'epoch': 3.0}                                   
{'loss': 0.016, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                        
{'eval_loss': 0.03505074232816696, 'eval_runtime': 17.8044, 'eval_samples_per_second': 16.85, 'eval_steps_per_second': 2.134, 'epoch': 4.0}                                   
{'loss': 0.0137, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                       
{'eval_loss': 0.03606244921684265, 'eval_runtime': 17.7782, 'eval_samples_per_second': 16.875, 'eval_steps_per_second': 2.137, 'epoch': 5.0}                                  
{'train_runtime': 521.3034, 'train_samples_per_second': 47.957, 'train_steps_per_second': 47.957, 'train_loss': 0.1808611209869385, 'epoch': 5.0}                             
 10%|█████████████                                                                                                                     | 2500/25000 [08:41<1:18:11,  4.80it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_correct_max_seq_lenght_batch_size_4

Starting fine-tuning...
{'loss': 1.7824, 'learning_rate': 0.00013297872340425532, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.06758204847574234, 'eval_runtime': 17.87, 'eval_samples_per_second': 16.788, 'eval_steps_per_second': 2.126, 'epoch': 1.0}                                    
{'loss': 0.0443, 'learning_rate': 0.00019994838405294247, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.04257085546851158, 'eval_runtime': 17.9533, 'eval_samples_per_second': 16.71, 'eval_steps_per_second': 2.117, 'epoch': 2.0}                                   
{'loss': 0.0287, 'learning_rate': 0.00019953077497104603, 'epoch': 3.0}                                                                                                       
{'eval_loss': 0.035779789090156555, 'eval_runtime': 17.8917, 'eval_samples_per_second': 16.768, 'eval_steps_per_second': 2.124, 'epoch': 3.0}                                 
{'loss': 0.025, 'learning_rate': 0.00019869563038461664, 'epoch': 4.0}                                                                                                        
{'eval_loss': 0.03633948788046837, 'eval_runtime': 17.8036, 'eval_samples_per_second': 16.85, 'eval_steps_per_second': 2.134, 'epoch': 4.0}                                   
{'loss': 0.0199, 'learning_rate': 0.0001974464537579482, 'epoch': 5.0}                                                                                                        
{'eval_loss': 0.03632332757115364, 'eval_runtime': 17.9153, 'eval_samples_per_second': 16.745, 'eval_steps_per_second': 2.121, 'epoch': 5.0}                                  
{'train_runtime': 451.2306, 'train_samples_per_second': 55.404, 'train_steps_per_second': 13.851, 'train_loss': 0.38008057861328126, 'epoch': 5.0}                            
 10%|█████████████▏                                                                                                                      | 625/6250 [07:31<1:07:41,  1.39it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_correct_max_seq_lenght_MAX_SEQ_LENGTH_1800

Starting fine-tuning...
{'loss': 0.8241, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.046056557446718216, 'eval_runtime': 17.8785, 'eval_samples_per_second': 16.78, 'eval_steps_per_second': 2.125, 'epoch': 1.0}                                  
{'loss': 0.0296, 'learning_rate': 0.0001999479755739169, 'epoch': 2.0}                                                                                                        
{'eval_loss': 0.035456880927085876, 'eval_runtime': 17.7772, 'eval_samples_per_second': 16.876, 'eval_steps_per_second': 2.138, 'epoch': 2.0}                                 
{'loss': 0.0208, 'learning_rate': 0.00019952959804615835, 'epoch': 3.0}                                                                                                       
{'eval_loss': 0.033714160323143005, 'eval_runtime': 17.7823, 'eval_samples_per_second': 16.871, 'eval_steps_per_second': 2.137, 'epoch': 3.0}                                 
{'loss': 0.0168, 'learning_rate': 0.0001986937587952128, 'epoch': 4.0}                                                                                                        
{'eval_loss': 0.036432843655347824, 'eval_runtime': 17.7723, 'eval_samples_per_second': 16.88, 'eval_steps_per_second': 2.138, 'epoch': 4.0}                                  
{'loss': 0.0136, 'learning_rate': 0.0001974439636213734, 'epoch': 5.0}                                                                                                        
{'eval_loss': 0.03641720488667488, 'eval_runtime': 17.7822, 'eval_samples_per_second': 16.871, 'eval_steps_per_second': 2.137, 'epoch': 5.0}                                  
{'train_runtime': 543.0945, 'train_samples_per_second': 46.033, 'train_steps_per_second': 46.033, 'train_loss': 0.18096906909942628, 'epoch': 5.0}                            
 10%|█████████████                                                                                                                     | 2500/25000 [09:03<1:21:27,  4.60it/s]
Fine-tuning completed.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_efficient_loss_correct_only_padding_longest

Starting fine-tuning...
{'loss': 0.8245, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.04628570377826691, 'eval_runtime': 17.8477, 'eval_samples_per_second': 16.809, 'eval_steps_per_second': 2.129, 'epoch': 1.0}                                  
{'loss': 0.0293, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.0365125872194767, 'eval_runtime': 17.6957, 'eval_samples_per_second': 16.953, 'eval_steps_per_second': 2.147, 'epoch': 2.0}                                   
{'loss': 0.0208, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                        
{'eval_loss': 0.03344814479351044, 'eval_runtime': 17.7675, 'eval_samples_per_second': 16.885, 'eval_steps_per_second': 2.139, 'epoch': 3.0}                                  
{'loss': 0.0164, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                       
{'eval_loss': 0.035842228680849075, 'eval_runtime': 17.7729, 'eval_samples_per_second': 16.88, 'eval_steps_per_second': 2.138, 'epoch': 4.0}                                  
{'loss': 0.0137, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                       
{'eval_loss': 0.03619945049285889, 'eval_runtime': 17.7379, 'eval_samples_per_second': 16.913, 'eval_steps_per_second': 2.142, 'epoch': 5.0}                                  
{'train_runtime': 518.3999, 'train_samples_per_second': 48.225, 'train_steps_per_second': 48.225, 'train_loss': 0.180930580329895, 'epoch': 5.0}                              
 10%|█████████████                                                                                                                     | 2500/25000 [08:38<1:17:45,  4.82it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''

'''

Starting fine-tuning...
{'loss': 0.8245, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.046327728778123856, 'eval_runtime': 17.9439, 'eval_samples_per_second': 16.719, 'eval_steps_per_second': 2.118, 'epoch': 1.0}                                 
{'loss': 0.0296, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.036112286150455475, 'eval_runtime': 17.7785, 'eval_samples_per_second': 16.874, 'eval_steps_per_second': 2.137, 'epoch': 2.0}                                 
{'loss': 0.0213, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                        
{'eval_loss': 0.035305723547935486, 'eval_runtime': 17.7381, 'eval_samples_per_second': 16.913, 'eval_steps_per_second': 2.142, 'epoch': 3.0}                                 
{'loss': 0.0167, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                       
{'eval_loss': 0.03565148264169693, 'eval_runtime': 17.7568, 'eval_samples_per_second': 16.895, 'eval_steps_per_second': 2.14, 'epoch': 4.0}                                   
{'loss': 0.0134, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                       
{'eval_loss': 0.036192044615745544, 'eval_runtime': 17.7395, 'eval_samples_per_second': 16.911, 'eval_steps_per_second': 2.142, 'epoch': 5.0}                                 
{'train_runtime': 565.4676, 'train_samples_per_second': 44.211, 'train_steps_per_second': 44.211, 'train_loss': 0.18110931892395019, 'epoch': 5.0}                            
 10%|█████████████                                                                                                                     | 2500/25000 [09:25<1:24:49,  4.42it/s]
Fine-tuning completed.
Model and tokenizer saved successfully.
'''