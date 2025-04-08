#############################################################################################################################################
# - Script to fine-tune a generic LLM on a given dataset.
#############################################################################################################################################

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    EarlyStoppingCallback
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
        datetime_obj = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d %H:%M:%S") # convert the timestamp to a datetime object
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
#LOGGING_STEPS = int(os.getenv("LOGGING_STEPS"))

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
print("MAX_SEQ_LENGTH: ", MAX_SEQ_LENGTH)
print("PACKING: ", PACKING)
print("DEVICE_MAP: ", DEVICE_MAP)
print("")
print("")
##########################################################################################


##########################################################################################
# STEP 2 - SETTING paths needed for fine-tuning the model:

# I create the full path of the fine-tuning dataset:
path_dataset_training = os.path.join(BASE_PATH, PATH_DATASET_TRAINING)
path_dataset_validation = ""
if(PATH_DATASET_VALIDATION != ""):
    path_dataset_validation = os.path.join(BASE_PATH, PATH_DATASET_VALIDATION)
print("")
print("Full path to the FT training dataset:", path_dataset_training)
print("")
print("Full path to the FT validation dataset:", path_dataset_validation)
print("")
print("")
print("Complete path of the basic model:", PATH_BASE_MODEL)
print("")
print("")

# Directory where the new fine-tuned model will be saved:
new_model = BASE_PATH+"/venvText2SQL/models/FineTuning_models/Fine_tuned_models/Cache_dir/"+NAME_NEW_MODEL

# Directory offload of new model:
new_model_offload_dir = BASE_PATH+"/venvText2SQL/models/FineTuning_models/Fine_tuned_models/Offload_dir/"+NAME_OFFLOAD_CACHE_DIR_NEW_MODEL

# Memory Monitor Data Folder Path:
PATH_csv = BASE_PATH+"/venvText2SQL/models/FineTuning_models/Fine_tuned_models/Memory_occupation/"+NAME_NEW_MODEL+"/memory_usage.csv"

# Extract only the path part of the directory
PATH_csv_dir = os.path.dirname(PATH_csv)

# Create folders if they do not exist
os.makedirs(new_model, exist_ok=True)
os.makedirs(new_model_offload_dir, exist_ok=True)
os.makedirs(PATH_csv_dir, exist_ok=True)
##########################################################################################


##########################################################################################
# STEP 3 - SETTING output folder:

# cleaning old checkpoints before training
if os.path.exists("./results"):
    shutil.rmtree("./results")

# Output directory where the model predictions and checkpoints will be stored
output_dir = "./results"
##########################################################################################


##########################################################################################
# Step 4: Loading dataset for fine-tuning (and possibly validation):

dataset_training = load_dataset('json', data_files=path_dataset_training, split="train")
dataset_training = dataset_training.shuffle(seed=42) # shuffle the training dataset

dataset_validation = load_dataset('json', data_files=path_dataset_validation, split="train")
print(f"Total number of rows in the dataset_training: {len(dataset_training)}")
print(f"Total number of rows in the dataset_validation: {len(dataset_validation)}")
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

if(path_dataset_validation == ""):
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        optim=OPTIM,
        save_steps=SAVE_STEPS,
        logging_strategy="epoch", # (with 'epoch' will show the value of loss function, for the training set, after each epoch)
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
        train_dataset=dataset_training,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=PACKING,
    )
else:
    print("I will use the validation set!")

    # Set training parameters
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

    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
    model=base_model,
    train_dataset=dataset_training,
    eval_dataset=dataset_validation,  # Aggiungi il validation set al trainer
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=PACKING,
    callbacks=[early_stopping]  # Aggiungi il callback per l'early stopping
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
    print("Start fine-tuning model.")
    trainer.train()

    end_time = time.time()
    stop_event.set() # signal the thread to stop
    monitor_thread.join() # I'm waiting for the thread to end

    print("End fine-tuning model.")
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
print("- The new model obtained after the fine-tuning has this path: ", new_model)
print("")
print("- Full path to new model offloading folder after fine-tuning: ", new_model_offload_dir)
print("")
print("- The file containing the info on the memory occupation of the new model after the fine-tuning has this path: ", PATH_csv)
print("")
print("Fine-tuning completed successfully.")
print("")
##########################################################################################


'''
- Llama-3-8B-Instruct-base-finetune_BioInfoDataset_v3_plus_newQueries_it_en_500_bilanced_final_4Epochs_TRAIN_BATCH_SIZE_1

Start fine-tuning model.
{'loss': 2.7692, 'learning_rate': 8.333333333333334e-05, 'epoch': 0.05}                                                                                                 
{'loss': 1.9164, 'learning_rate': 0.0001666666666666667, 'epoch': 0.1}                                                                                                  
{'loss': 0.354, 'learning_rate': 0.0001999704996306308, 'epoch': 0.15}                                                                                                  
{'loss': 0.1227, 'learning_rate': 0.00019979028262377118, 'epoch': 0.2}                                                                                                 
{'loss': 0.1906, 'learning_rate': 0.00019944653267951504, 'epoch': 0.25}                                                                                                
{'loss': 0.131, 'learning_rate': 0.00019893981312363562, 'epoch': 0.3}                                                                                                  
{'loss': 0.1243, 'learning_rate': 0.00019827095435098925, 'epoch': 0.35}                                                                                                
{'loss': 0.0906, 'learning_rate': 0.00019744105246469263, 'epoch': 0.4}                                                                                                 
{'loss': 0.117, 'learning_rate': 0.0001964514674798659, 'epoch': 0.45}                                                                                                  
{'loss': 0.0568, 'learning_rate': 0.0001953038210948861, 'epoch': 0.5}                                                                                                  
{'loss': 0.1312, 'learning_rate': 0.00019399999403380266, 'epoch': 0.55}                                                                                                
{'loss': 0.0673, 'learning_rate': 0.00019254212296427044, 'epoch': 0.6}                                                                                                 
{'loss': 0.1133, 'learning_rate': 0.00019093259699605125, 'epoch': 0.65}                                                                                                
{'loss': 0.0945, 'learning_rate': 0.00018917405376582145, 'epoch': 0.7}                                                                                                 
{'loss': 0.0706, 'learning_rate': 0.00018726937511470246, 'epoch': 0.75}                                                                                                
{'loss': 0.0667, 'learning_rate': 0.00018522168236559695, 'epoch': 0.8}                                                                                                 
{'loss': 0.0945, 'learning_rate': 0.0001830343312080704, 'epoch': 0.85}                                                                                                 
{'loss': 0.0593, 'learning_rate': 0.00018071090619916093, 'epoch': 0.9}                                                                                                 
{'loss': 0.1464, 'learning_rate': 0.0001782552148891283, 'epoch': 0.95}                                                                                                 
{'loss': 0.0567, 'learning_rate': 0.00017567128158176953, 'epoch': 1.0}                                                                                                 
{'loss': 0.0841, 'learning_rate': 0.00017296334073952605, 'epoch': 1.05}                                                                                                
{'loss': 0.0537, 'learning_rate': 0.00017013583004418993, 'epoch': 1.1}                                                                                                 
{'loss': 0.0982, 'learning_rate': 0.00016719338312458124, 'epoch': 1.15}                                                                                                
{'loss': 0.0546, 'learning_rate': 0.000164140821963114, 'epoch': 1.2}                                                                                                   
{'loss': 0.0924, 'learning_rate': 0.00016111140969350503, 'epoch': 1.25}                                                                                                
{'loss': 0.0502, 'learning_rate': 0.0001578576956519654, 'epoch': 1.3}                                                                                                  
{'loss': 0.0996, 'learning_rate': 0.00015450916637719684, 'epoch': 1.35}                                                                                                
{'loss': 0.0667, 'learning_rate': 0.0001510713093256315, 'epoch': 1.4}                                                                                                  
{'loss': 0.0847, 'learning_rate': 0.00014754975834103877, 'epoch': 1.45}                                                                                                
{'loss': 0.0601, 'learning_rate': 0.0001439502844219723, 'epoch': 1.5}                                                                                                  
{'loss': 0.0744, 'learning_rate': 0.0001402787862644534, 'epoch': 1.55}                                                                                                 
{'loss': 0.052, 'learning_rate': 0.0001365412805953872, 'epoch': 1.6}                                                                                                   
{'loss': 0.0744, 'learning_rate': 0.00013274389231255466, 'epoch': 1.65}                                                                                                
{'loss': 0.0437, 'learning_rate': 0.00012889284444733722, 'epoch': 1.7}                                                                                                 
{'loss': 0.0721, 'learning_rate': 0.00012499444796662353, 'epoch': 1.75}                                                                                                
{'loss': 0.0496, 'learning_rate': 0.00012105509143061071, 'epoch': 1.8}                                                                                                 
{'loss': 0.0758, 'learning_rate': 0.00011708123052344804, 'epoch': 1.85}                                                                                                
{'loss': 0.049, 'learning_rate': 0.00011307937747388034, 'epoch': 1.9}                                                                                                  
{'loss': 0.0654, 'learning_rate': 0.00010905609038322779, 'epoch': 1.95}                                                                                                
{'loss': 0.0336, 'learning_rate': 0.00010501796247819175, 'epoch': 2.0}                                                                                                 
{'loss': 0.0508, 'learning_rate': 0.00010097161130609773, 'epoch': 2.05}                                                                                                
{'loss': 0.0377, 'learning_rate': 9.692366789028307e-05, 'epoch': 2.1}                                                                                                  
{'loss': 0.0549, 'learning_rate': 9.288076586340006e-05, 'epoch': 2.15}                                                                                                 
{'loss': 0.041, 'learning_rate': 8.88495305964436e-05, 'epoch': 2.2}                                                                                                    
{'loss': 0.0591, 'learning_rate': 8.48365683413172e-05, 'epoch': 2.25}                                                                                                  
{'loss': 0.0359, 'learning_rate': 8.084845540473126e-05, 'epoch': 2.3}                                                                                                  
{'loss': 0.0526, 'learning_rate': 7.689172737117389e-05, 'epoch': 2.35}                                                                                                 
{'loss': 0.0393, 'learning_rate': 7.297286839261658e-05, 'epoch': 2.4}                                                                                                  
{'loss': 0.0484, 'learning_rate': 6.909830056250527e-05, 'epoch': 2.45}                                                                                                 
{'loss': 0.0319, 'learning_rate': 6.527437339145098e-05, 'epoch': 2.5}                                                                                                  
{'loss': 0.0567, 'learning_rate': 6.15073534018669e-05, 'epoch': 2.55}                                                                                                  
{'loss': 0.0421, 'learning_rate': 5.780341385860333e-05, 'epoch': 2.6}                                                                                                  
{'loss': 0.0524, 'learning_rate': 5.416862465241033e-05, 'epoch': 2.65}                                                                                                 
{'loss': 0.0324, 'learning_rate': 5.0608942352806364e-05, 'epoch': 2.7}                                                                                                 
{'loss': 0.0457, 'learning_rate': 4.7130200446653475e-05, 'epoch': 2.75}                                                                                                
{'loss': 0.0351, 'learning_rate': 4.373809977843676e-05, 'epoch': 2.8}                                                                                                  
{'loss': 0.0482, 'learning_rate': 4.043819920791322e-05, 'epoch': 2.85}                                                                                                 
{'loss': 0.0349, 'learning_rate': 3.7235906500440574e-05, 'epoch': 2.9}                                                                                                 
{'loss': 0.053, 'learning_rate': 3.4136469464914575e-05, 'epoch': 2.95}                                                                                                 
{'loss': 0.0322, 'learning_rate': 3.1144967353837196e-05, 'epoch': 3.0}                                                                                                 
{'loss': 0.0438, 'learning_rate': 2.8266302539609745e-05, 'epoch': 3.05}                                                                                                
{'loss': 0.0251, 'learning_rate': 2.5505192480690865e-05, 'epoch': 3.1}                                                                                                 
{'loss': 0.033, 'learning_rate': 2.2866161990785228e-05, 'epoch': 3.15}                                                                                                 
{'loss': 0.0311, 'learning_rate': 2.035353582373205e-05, 'epoch': 3.2}                                                                                                  
{'loss': 0.0374, 'learning_rate': 1.7971431586244815e-05, 'epoch': 3.25}                                                                                                
{'loss': 0.0239, 'learning_rate': 1.5723752990116947e-05, 'epoch': 3.3}                                                                                                 
{'loss': 0.0338, 'learning_rate': 1.3614183454950824e-05, 'epoch': 3.35}                                                                                                
{'loss': 0.0292, 'learning_rate': 1.1646180071894607e-05, 'epoch': 3.4}                                                                                                 
{'loss': 0.0334, 'learning_rate': 9.822967938278171e-06, 'epoch': 3.45}                                                                                                 
{'loss': 0.0252, 'learning_rate': 8.14753487243276e-06, 'epoch': 3.5}                                                                                                   
{'loss': 0.0349, 'learning_rate': 6.622626517355557e-06, 'epoch': 3.55}                                                                                                 
{'loss': 0.0257, 'learning_rate': 5.250741841242734e-06, 'epoch': 3.6}                                                                                                  
{'loss': 0.03, 'learning_rate': 4.034129042265066e-06, 'epoch': 3.65}                                                                                                   
{'loss': 0.0247, 'learning_rate': 2.9747818642967827e-06, 'epoch': 3.7}                                                                                                 
{'loss': 0.0328, 'learning_rate': 2.074436329635687e-06, 'epoch': 3.75}                                                                                                 
{'loss': 0.0225, 'learning_rate': 1.3345678940684613e-06, 'epoch': 3.8}                                                                                                 
{'loss': 0.0366, 'learning_rate': 7.563890289437825e-07, 'epoch': 3.85}                                                                                                 
{'loss': 0.0272, 'learning_rate': 3.4084723421521356e-07, 'epoch': 3.9}                                                                                                 
{'loss': 0.0382, 'learning_rate': 8.862348571043733e-08, 'epoch': 3.95}                                                                                                 
{'loss': 0.0248, 'learning_rate': 1.311191710651194e-10, 'epoch': 4.0}                                                                                                  
{'train_runtime': 312.7968, 'train_samples_per_second': 6.394, 'train_steps_per_second': 6.394, 'train_loss': 0.12006199645996093, 'epoch': 4.0}
'''

'''
- Llama-3-70B-Instruct-base-finetune_BioInfoDataset_v3_plus_newQueries_it_en_500_bilanced_final_4Epochs_TRAIN_BATCH_SIZE_1_4bit 

{'loss': 2.6193, 'learning_rate': 7.333333333333333e-05, 'epoch': 0.05}                                                                                        
{'loss': 1.4919, 'learning_rate': 0.00015666666666666666, 'epoch': 0.1}                                                                                        
{'loss': 0.285, 'learning_rate': 0.0001999811194293973, 'epoch': 0.15}                                                                                         
{'loss': 0.1123, 'learning_rate': 0.0001998205515108853, 'epoch': 0.2}                                                                                         
{'loss': 0.1649, 'learning_rate': 0.00019949640105134918, 'epoch': 0.25}                                                                                       
{'loss': 0.1235, 'learning_rate': 0.00019900919925759002, 'epoch': 0.3}                                                                                        
{'loss': 0.1155, 'learning_rate': 0.0001983597445394162, 'epoch': 0.35}                                                                                        
{'loss': 0.0881, 'learning_rate': 0.00019754910120123675, 'epoch': 0.4}                                                                                        
{'loss': 0.1071, 'learning_rate': 0.00019657859769791505, 'epoch': 0.45}                                                                                       
{'loss': 0.0583, 'learning_rate': 0.00019544982445774217, 'epoch': 0.5}                                                                                        
{'loss': 0.117, 'learning_rate': 0.00019416463127609656, 'epoch': 0.55}                                                                                        
{'loss': 0.0636, 'learning_rate': 0.0001927251242840623, 'epoch': 0.6}                                                                                         
{'loss': 0.0941, 'learning_rate': 0.0001911336624969725, 'epoch': 0.65}                                                                                        
{'loss': 0.0851, 'learning_rate': 0.00018939285394853502, 'epoch': 0.7}                                                                                        
{'loss': 0.0595, 'learning_rate': 0.000187505551416875, 'epoch': 0.75}                                                                                         
{'loss': 0.0591, 'learning_rate': 0.00018547484774949867, 'epoch': 0.8}                                                                                        
{'loss': 0.0849, 'learning_rate': 0.00018330407079483952, 'epoch': 0.85}                                                                                       
{'loss': 0.0549, 'learning_rate': 0.00018099677794869296, 'epoch': 0.9}                                                                                        
{'loss': 0.1237, 'learning_rate': 0.00017855675032447648, 'epoch': 0.95}                                                                                       
{'loss': 0.0516, 'learning_rate': 0.000175987986556869, 'epoch': 1.0}                                                                                          
{'loss': 0.0698, 'learning_rate': 0.0001732946962489836, 'epoch': 1.05}                                                                                        
{'loss': 0.0456, 'learning_rate': 0.00017048129307381266, 'epoch': 1.1}                                                                                        
{'loss': 0.0827, 'learning_rate': 0.00016755238754124965, 'epoch': 1.15}                                                                                       
{'loss': 0.0464, 'learning_rate': 0.00016451277944254185, 'epoch': 1.2}                                                                                        
{'loss': 0.0749, 'learning_rate': 0.00016136744998455476, 'epoch': 1.25}                                                                                       
{'loss': 0.042, 'learning_rate': 0.00015812155362673896, 'epoch': 1.3}                                                                                         
{'loss': 0.0851, 'learning_rate': 0.0001547804096341763, 'epoch': 1.35}                                                                                        
{'loss': 0.0556, 'learning_rate': 0.00015134949336054865, 'epoch': 1.4}                                                                                        
{'loss': 0.0677, 'learning_rate': 0.00014783442727531328, 'epoch': 1.45}                                                                                       
{'loss': 0.048, 'learning_rate': 0.00014424097174979038, 'epoch': 1.5}                                                                                         
{'loss': 0.0611, 'learning_rate': 0.00014057501561726157, 'epoch': 1.55}                                                                                       
{'loss': 0.0445, 'learning_rate': 0.00013684256652254906, 'epoch': 1.6}                                                                                        
{'loss': 0.0649, 'learning_rate': 0.00013304974107689087, 'epoch': 1.65}                                                                                       
{'loss': 0.0353, 'learning_rate': 0.00012920275483424538, 'epoch': 1.7}                                                                                        
{'loss': 0.0584, 'learning_rate': 0.00012530791210545162, 'epoch': 1.75}                                                                                       
{'loss': 0.0413, 'learning_rate': 0.00012137159562693838, 'epoch': 1.8}                                                                                        
{'loss': 0.0581, 'learning_rate': 0.00011740025610091159, 'epoch': 1.85}                                                                                       
{'loss': 0.0436, 'learning_rate': 0.00011340040162416197, 'epoch': 1.9}                                                                                        
{'loss': 0.0539, 'learning_rate': 0.00010937858702281631, 'epoch': 1.95}                                                                                       
{'loss': 0.0312, 'learning_rate': 0.00010534140311051026, 'epoch': 2.0}                                                                                        
{'loss': 0.0396, 'learning_rate': 0.00010129546588758605, 'epoch': 2.05}                                                                                       
{'loss': 0.0321, 'learning_rate': 9.724740569901503e-05, 'epoch': 2.1}                                                                                         
{'loss': 0.0412, 'learning_rate': 9.320385636881283e-05, 'epoch': 2.15}                                                                                        
{'loss': 0.0328, 'learning_rate': 8.91714443287531e-05, 'epoch': 2.2}                                                                                          
{'loss': 0.0441, 'learning_rate': 8.515677775919527e-05, 'epoch': 2.25}                                                                                        
{'loss': 0.0305, 'learning_rate': 8.116643575982253e-05, 'epoch': 2.3}                                                                                         
{'loss': 0.0436, 'learning_rate': 7.72069575680357e-05, 'epoch': 2.35}                                                                                         
{'loss': 0.0326, 'learning_rate': 7.328483184267235e-05, 'epoch': 2.4}                                                                                         
{'loss': 0.0392, 'learning_rate': 6.940648603061263e-05, 'epoch': 2.45}                                                                                        
{'loss': 0.0277, 'learning_rate': 6.557827583369647e-05, 'epoch': 2.5}                                                                                         
{'loss': 0.0407, 'learning_rate': 6.180647479321485e-05, 'epoch': 2.55}                                                                                        
{'loss': 0.0353, 'learning_rate': 5.809726400904242e-05, 'epoch': 2.6}                                                                                         
{'loss': 0.0395, 'learning_rate': 5.445672201026054e-05, 'epoch': 2.65}                                                                                        
{'loss': 0.028, 'learning_rate': 5.089081479386928e-05, 'epoch': 2.7}                                                                                          
{'loss': 0.0339, 'learning_rate': 4.74053860479137e-05, 'epoch': 2.75}                                                                                         
{'loss': 0.0286, 'learning_rate': 4.400614757504564e-05, 'epoch': 2.8}                                                                                         
{'loss': 0.0365, 'learning_rate': 4.0698669932214727e-05, 'epoch': 2.85}                                                                                       
{'loss': 0.0275, 'learning_rate': 3.7488373301828296e-05, 'epoch': 2.9}                                                                                        
{'loss': 0.0335, 'learning_rate': 3.4380518609340076e-05, 'epoch': 2.95}                                                                                       
{'loss': 0.0272, 'learning_rate': 3.138019890182331e-05, 'epoch': 3.0}                                                                                         
{'loss': 0.0303, 'learning_rate': 2.8492331001657945e-05, 'epoch': 3.05}                                                                                       
{'loss': 0.022, 'learning_rate': 2.572164744900827e-05, 'epoch': 3.1}                                                                                          
{'loss': 0.0279, 'learning_rate': 2.307268874629649e-05, 'epoch': 3.15}                                                                                        
{'loss': 0.0234, 'learning_rate': 2.0549795917380864e-05, 'epoch': 3.2}                                                                                        
{'loss': 0.028, 'learning_rate': 1.8157103393632868e-05, 'epoch': 3.25}                                                                                        
{'loss': 0.0221, 'learning_rate': 1.589853223857103e-05, 'epoch': 3.3}                                                                                         
{'loss': 0.026, 'learning_rate': 1.3777783722154603e-05, 'epoch': 3.35}                                                                                        
{'loss': 0.0234, 'learning_rate': 1.1798333255267857e-05, 'epoch': 3.4}                                                                                        
{'loss': 0.0266, 'learning_rate': 9.963424694334122e-06, 'epoch': 3.45}                                                                                        
{'loss': 0.0222, 'learning_rate': 8.276065025393908e-06, 'epoch': 3.5}                                                                                         
{'loss': 0.0263, 'learning_rate': 6.739019436357774e-06, 'epoch': 3.55}                                                                                        
{'loss': 0.0228, 'learning_rate': 5.354806785510113e-06, 'epoch': 3.6}                                                                                         
{'loss': 0.0252, 'learning_rate': 4.125695473689406e-06, 'epoch': 3.65}                                                                                        
{'loss': 0.0209, 'learning_rate': 3.053699726909676e-06, 'epoch': 3.7}                                                                                         
{'loss': 0.0253, 'learning_rate': 2.1405762955151176e-06, 'epoch': 3.75}                                                                                       
{'loss': 0.0212, 'learning_rate': 1.3878215752771262e-06, 'epoch': 3.8}                                                                                        
{'loss': 0.0264, 'learning_rate': 7.966691551514527e-07, 'epoch': 3.85}                                                                                        
{'loss': 0.0232, 'learning_rate': 3.6808779571451126e-07, 'epoch': 3.9}                                                                                        
{'loss': 0.025, 'learning_rate': 1.0277984159122733e-07, 'epoch': 3.95}                                                                                        
{'loss': 0.0231, 'learning_rate': 1.1800704765030368e-09, 'epoch': 4.0}                                                                                        
{'train_runtime': 4254.8186, 'train_samples_per_second': 0.47, 'train_steps_per_second': 0.47, 'train_loss': 0.103180978089571, 'epoch': 4.0}
'''

'''
- Llama-3-8B-Instruct-base-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_MAX_20Epochs_BatchSize_8

{'loss': 1.6227, 'learning_rate': 0.0001997935295756461, 'epoch': 1.0}                                                                                               
{'eval_loss': 0.1650983989238739, 'eval_runtime': 6.2746, 'eval_samples_per_second': 47.812, 'eval_steps_per_second': 6.056, 'epoch': 1.0}                           
{'loss': 0.1124, 'learning_rate': 0.00019745176468573893, 'epoch': 2.0}                                                                                              
{'eval_loss': 0.10983411222696304, 'eval_runtime': 6.2769, 'eval_samples_per_second': 47.794, 'eval_steps_per_second': 6.054, 'epoch': 2.0}                          
{'loss': 0.0754, 'learning_rate': 0.00019255918721902235, 'epoch': 3.0}                                                                                              
{'eval_loss': 0.10027991235256195, 'eval_runtime': 6.2713, 'eval_samples_per_second': 47.837, 'eval_steps_per_second': 6.059, 'epoch': 3.0}                          
{'loss': 0.0613, 'learning_rate': 0.00018524386102508755, 'epoch': 4.0}                                                                                              
{'eval_loss': 0.10031630843877792, 'eval_runtime': 6.2881, 'eval_samples_per_second': 47.709, 'eval_steps_per_second': 6.043, 'epoch': 4.0}                          
{'loss': 0.0521, 'learning_rate': 0.00017569726571560837, 'epoch': 5.0}                                                                                              
{'eval_loss': 0.10702593624591827, 'eval_runtime': 6.2923, 'eval_samples_per_second': 47.677, 'eval_steps_per_second': 6.039, 'epoch': 5.0}                          
{'loss': 0.0438, 'learning_rate': 0.00016416928466067723, 'epoch': 6.0}                                                                                              
{'eval_loss': 0.10828951746225357, 'eval_runtime': 6.3036, 'eval_samples_per_second': 47.592, 'eval_steps_per_second': 6.028, 'epoch': 6.0}                          
{'train_runtime': 178.4175, 'train_samples_per_second': 56.048, 'train_steps_per_second': 7.062, 'train_loss': 0.3279497787435219, 'epoch': 6.0}                     
 30%|█████████████████████████████████████▌                                                                                       | 378/1260 [02:58<06:56,  2.12it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- Llama-3-8B-Instruct-base-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_MAX_20Epochs_BatchSize_8

{'loss': 1.6174, 'learning_rate': 0.0001997935295756461, 'epoch': 1.0}                                                                                       
{'eval_loss': 0.1678396314382553, 'eval_runtime': 6.2704, 'eval_samples_per_second': 47.844, 'eval_steps_per_second': 6.06, 'epoch': 1.0}                    
{'loss': 0.1136, 'learning_rate': 0.00019745176468573893, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.10989481955766678, 'eval_runtime': 6.2712, 'eval_samples_per_second': 47.838, 'eval_steps_per_second': 6.059, 'epoch': 2.0}                  
{'loss': 0.0759, 'learning_rate': 0.00019255918721902235, 'epoch': 3.0}                                                                                      
{'eval_loss': 0.09919679909944534, 'eval_runtime': 6.2737, 'eval_samples_per_second': 47.819, 'eval_steps_per_second': 6.057, 'epoch': 3.0}                  
{'loss': 0.061, 'learning_rate': 0.00018524386102508755, 'epoch': 4.0}                                                                                       
{'eval_loss': 0.09967923164367676, 'eval_runtime': 6.3074, 'eval_samples_per_second': 47.563, 'eval_steps_per_second': 6.025, 'epoch': 4.0}                  
{'loss': 0.0517, 'learning_rate': 0.00017569726571560837, 'epoch': 5.0}                                                                                      
{'eval_loss': 0.10595410317182541, 'eval_runtime': 6.3357, 'eval_samples_per_second': 47.351, 'eval_steps_per_second': 5.998, 'epoch': 5.0}                  
{'loss': 0.0442, 'learning_rate': 0.00016416928466067723, 'epoch': 6.0}                                                                                      
{'eval_loss': 0.10761833190917969, 'eval_runtime': 6.3538, 'eval_samples_per_second': 47.216, 'eval_steps_per_second': 5.981, 'epoch': 6.0}                  
{'train_runtime': 178.8727, 'train_samples_per_second': 55.906, 'train_steps_per_second': 7.044, 'train_loss': 0.32731113862739036, 'epoch': 6.0} 
'''

'''

{'loss': 1.6226, 'learning_rate': 0.0001997935295756461, 'epoch': 1.0}                                                                                       
{'eval_loss': 0.16677436232566833, 'eval_runtime': 6.2718, 'eval_samples_per_second': 47.833, 'eval_steps_per_second': 6.059, 'epoch': 1.0}                  
{'loss': 0.1132, 'learning_rate': 0.00019745176468573893, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.10975358635187149, 'eval_runtime': 6.2729, 'eval_samples_per_second': 47.825, 'eval_steps_per_second': 6.058, 'epoch': 2.0}                  
{'loss': 0.0756, 'learning_rate': 0.00019255918721902235, 'epoch': 3.0}                                                                                      
{'eval_loss': 0.10025402903556824, 'eval_runtime': 6.2715, 'eval_samples_per_second': 47.835, 'eval_steps_per_second': 6.059, 'epoch': 3.0}                  
{'loss': 0.0614, 'learning_rate': 0.00018524386102508755, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.09969419240951538, 'eval_runtime': 6.2978, 'eval_samples_per_second': 47.636, 'eval_steps_per_second': 6.034, 'epoch': 4.0}                  
{'loss': 0.0521, 'learning_rate': 0.00017569726571560837, 'epoch': 5.0}                                                                                      
{'eval_loss': 0.10892874747514725, 'eval_runtime': 6.3268, 'eval_samples_per_second': 47.417, 'eval_steps_per_second': 6.006, 'epoch': 5.0}                  
{'loss': 0.0441, 'learning_rate': 0.00016416928466067723, 'epoch': 6.0}                                                                                      
{'eval_loss': 0.10846225917339325, 'eval_runtime': 6.3366, 'eval_samples_per_second': 47.344, 'eval_steps_per_second': 5.997, 'epoch': 6.0}                  
{'loss': 0.0376, 'learning_rate': 0.00015096166425919175, 'epoch': 7.0}                                                                                      
{'eval_loss': 0.11936645209789276, 'eval_runtime': 6.3385, 'eval_samples_per_second': 47.33, 'eval_steps_per_second': 5.995, 'epoch': 7.0}                   
{'loss': 0.0326, 'learning_rate': 0.00013642011568773744, 'epoch': 8.0}                                                                                      
{'eval_loss': 0.11636809259653091, 'eval_runtime': 6.328, 'eval_samples_per_second': 47.408, 'eval_steps_per_second': 6.005, 'epoch': 8.0}                   
{'loss': 0.0277, 'learning_rate': 0.0001209252658657134, 'epoch': 9.0}                                                                                       
{'eval_loss': 0.13250987231731415, 'eval_runtime': 6.3479, 'eval_samples_per_second': 47.26, 'eval_steps_per_second': 5.986, 'epoch': 9.0}                   
{'train_runtime': 268.6767, 'train_samples_per_second': 37.219, 'train_steps_per_second': 4.69, 'train_loss': 0.229654337573514, 'epoch': 9.0}               
 45%|████████████████████████████████████████████████████▋                                                                | 567/1260 [04:28<05:28,  2.11it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''


'''
1 epoch:

Start fine-tuning model.
{'loss': 0.5596, 'learning_rate': 0.00019979028262377118, 'epoch': 1.0}                                                                                      
{'eval_loss': 0.10747231543064117, 'eval_runtime': 6.2822, 'eval_samples_per_second': 47.754, 'eval_steps_per_second': 6.049, 'epoch': 1.0}                  
{'loss': 0.067, 'learning_rate': 0.00019744105246469263, 'epoch': 2.0}                                                                                       
{'eval_loss': 0.0956832766532898, 'eval_runtime': 6.2713, 'eval_samples_per_second': 47.837, 'eval_steps_per_second': 6.059, 'epoch': 2.0}                   
{'loss': 0.0488, 'learning_rate': 0.0001925543910508503, 'epoch': 3.0}                                                                                       
{'eval_loss': 0.09987049549818039, 'eval_runtime': 6.2681, 'eval_samples_per_second': 47.862, 'eval_steps_per_second': 6.062, 'epoch': 3.0}                  
{'loss': 0.0383, 'learning_rate': 0.00018523862269959583, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.105838343501091, 'eval_runtime': 6.2687, 'eval_samples_per_second': 47.857, 'eval_steps_per_second': 6.062, 'epoch': 4.0}                    
{'loss': 0.0314, 'learning_rate': 0.00017571361226352306, 'epoch': 5.0}                                                                                      
{'eval_loss': 0.11119303107261658, 'eval_runtime': 6.2612, 'eval_samples_per_second': 47.914, 'eval_steps_per_second': 6.069, 'epoch': 5.0}                  
{'train_runtime': 426.9107, 'train_samples_per_second': 23.424, 'train_steps_per_second': 23.424, 'train_loss': 0.1490085174560547, 'epoch': 5.0}            
 25%|████████████████████████████▊                                                                                      | 2500/10000 [07:06<21:20,  5.86it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- Llama-3-8B-Instruct-base-FT-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_1_p2

{'loss': 0.5592, 'learning_rate': 0.00019979237383039745, 'epoch': 1.0}                                                                                      
{'eval_loss': 0.10720962285995483, 'eval_runtime': 6.2912, 'eval_samples_per_second': 47.686, 'eval_steps_per_second': 6.04, 'epoch': 1.0}                   
{'loss': 0.0668, 'learning_rate': 0.0001974483272899535, 'epoch': 2.0}                                                                                       
{'eval_loss': 0.0973922535777092, 'eval_runtime': 6.2607, 'eval_samples_per_second': 47.918, 'eval_steps_per_second': 6.07, 'epoch': 2.0}                    
{'loss': 0.0485, 'learning_rate': 0.0001925543910508503, 'epoch': 3.0}                                                                                       
{'eval_loss': 0.10129699856042862, 'eval_runtime': 6.2771, 'eval_samples_per_second': 47.793, 'eval_steps_per_second': 6.054, 'epoch': 3.0}                  
{'loss': 0.0386, 'learning_rate': 0.00018523862269959583, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.11065905541181564, 'eval_runtime': 6.2745, 'eval_samples_per_second': 47.813, 'eval_steps_per_second': 6.056, 'epoch': 4.0}                  
{'train_runtime': 325.9861, 'train_samples_per_second': 30.676, 'train_steps_per_second': 30.676, 'train_loss': 0.17826729583740233, 'epoch': 4.0}           
 20%|███████████████████████                                                                                            | 2000/10000 [05:25<21:43,  6.14it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- Llama-3-8B-Instruct-base-FT-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_8_p2

Start fine-tuning model.
{'loss': 1.6215, 'learning_rate': 0.0001997935295756461, 'epoch': 1.0}                                                                                       
{'eval_loss': 0.16601517796516418, 'eval_runtime': 6.2731, 'eval_samples_per_second': 47.823, 'eval_steps_per_second': 6.058, 'epoch': 1.0}                  
{'loss': 0.1131, 'learning_rate': 0.00019745176468573893, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.10954185575246811, 'eval_runtime': 6.2778, 'eval_samples_per_second': 47.787, 'eval_steps_per_second': 6.053, 'epoch': 2.0}                  
{'loss': 0.0761, 'learning_rate': 0.00019255918721902235, 'epoch': 3.0}                                                                                      
{'eval_loss': 0.10079669207334518, 'eval_runtime': 6.2846, 'eval_samples_per_second': 47.736, 'eval_steps_per_second': 6.047, 'epoch': 3.0}                  
{'loss': 0.0617, 'learning_rate': 0.00018524386102508755, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.10033874213695526, 'eval_runtime': 6.2873, 'eval_samples_per_second': 47.715, 'eval_steps_per_second': 6.044, 'epoch': 4.0}                  
{'loss': 0.0518, 'learning_rate': 0.00017569726571560837, 'epoch': 5.0}                                                                                      
{'eval_loss': 0.10838324576616287, 'eval_runtime': 6.308, 'eval_samples_per_second': 47.558, 'eval_steps_per_second': 6.024, 'epoch': 5.0}                   
{'loss': 0.0445, 'learning_rate': 0.00016416928466067723, 'epoch': 6.0}                                                                                      
{'eval_loss': 0.10646001994609833, 'eval_runtime': 6.3312, 'eval_samples_per_second': 47.385, 'eval_steps_per_second': 6.002, 'epoch': 6.0}                  
{'train_runtime': 179.1143, 'train_samples_per_second': 55.83, 'train_steps_per_second': 7.035, 'train_loss': 0.3281048069555293, 'epoch': 6.0}              
 30%|███████████████████████████████████                                                                                  | 378/1260 [02:59<06:57,  2.11it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''


'''
- Llama-3-8B-Instruct-base-FT-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_4_p2

Start fine-tuning model.
{'loss': 1.1107, 'learning_rate': 0.00019979028262377118, 'epoch': 1.0}                                                                                      
{'eval_loss': 0.13560564815998077, 'eval_runtime': 6.2649, 'eval_samples_per_second': 47.886, 'eval_steps_per_second': 6.066, 'epoch': 1.0}                  
{'loss': 0.0898, 'learning_rate': 0.00019744105246469263, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.10130855441093445, 'eval_runtime': 6.2604, 'eval_samples_per_second': 47.92, 'eval_steps_per_second': 6.07, 'epoch': 2.0}                    
{'loss': 0.0645, 'learning_rate': 0.00019254212296427044, 'epoch': 3.0}                                                                                      
{'eval_loss': 0.09778562933206558, 'eval_runtime': 6.268, 'eval_samples_per_second': 47.862, 'eval_steps_per_second': 6.063, 'epoch': 3.0}                   
{'loss': 0.0517, 'learning_rate': 0.00018522168236559695, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.10443221032619476, 'eval_runtime': 6.2667, 'eval_samples_per_second': 47.872, 'eval_steps_per_second': 6.064, 'epoch': 4.0}                  
{'loss': 0.0423, 'learning_rate': 0.00017567128158176953, 'epoch': 5.0}                                                                                      
{'eval_loss': 0.10773957520723343, 'eval_runtime': 6.2696, 'eval_samples_per_second': 47.85, 'eval_steps_per_second': 6.061, 'epoch': 5.0}                   
{'train_runtime': 159.9988, 'train_samples_per_second': 62.5, 'train_steps_per_second': 15.625, 'train_loss': 0.27181015853881835, 'epoch': 5.0}             
 25%|█████████████████████████████▎                                                                                       | 625/2500 [02:39<07:59,  3.91it/s]
End fine-tuning model.
'''


'''
- Llama-3-70B-Instruct-base-FT-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_4_p2

Start fine-tuning model.
  0%|                                                                                                                               | 0/2500 [00:00<?, ?it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.9423, 'learning_rate': 0.00019980671916744352, 'epoch': 1.0}                                                                                      
{'eval_loss': 0.11718600988388062, 'eval_runtime': 69.4238, 'eval_samples_per_second': 4.321, 'eval_steps_per_second': 0.547, 'epoch': 1.0}                  
  5%|█████▊                                                                                                             | 125/2500 [07:43<2:06:42,  3.20s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0812, 'learning_rate': 0.00019755622261881427, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.08926526457071304, 'eval_runtime': 69.2855, 'eval_samples_per_second': 4.33, 'eval_steps_per_second': 0.548, 'epoch': 2.0}                   
 10%|███████████▌                                                                                                       | 250/2500 [15:27<1:51:48,  2.98s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0548, 'learning_rate': 0.00019273724660820088, 'epoch': 3.0}                                                                                      
{'eval_loss': 0.09061967581510544, 'eval_runtime': 69.4121, 'eval_samples_per_second': 4.322, 'eval_steps_per_second': 0.547, 'epoch': 3.0}                  
 15%|█████████████████▎                                                                                                 | 375/2500 [23:12<1:48:46,  3.07s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0408, 'learning_rate': 0.00018549165378040327, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.0952005609869957, 'eval_runtime': 69.2973, 'eval_samples_per_second': 4.329, 'eval_steps_per_second': 0.548, 'epoch': 4.0}                   
{'train_runtime': 1862.0536, 'train_samples_per_second': 5.37, 'train_steps_per_second': 1.343, 'train_loss': 0.2798012819290161, 'epoch': 4.0}              
 20%|███████████████████████                                                                                            | 500/2500 [31:00<2:04:03,  3.72s/it]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- Llama-3-70B-Instruct-base-FT-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_1_p2

Start fine-tuning model.
  0%|                                                                                                                              | 0/10000 [00:00<?, ?it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.4801, 'learning_rate': 0.00019980063397715683, 'epoch': 1.0}                                                                                      
{'eval_loss': 0.09647130966186523, 'eval_runtime': 69.5003, 'eval_samples_per_second': 4.317, 'eval_steps_per_second': 0.547, 'epoch': 1.0}                  
  5%|█████▋                                                                                                            | 500/10000 [16:10<4:34:34,  1.73s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0581, 'learning_rate': 0.00019747732436460952, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.08800161629915237, 'eval_runtime': 69.3487, 'eval_samples_per_second': 4.326, 'eval_steps_per_second': 0.548, 'epoch': 2.0}                  
 10%|███████████▎                                                                                                     | 1000/10000 [32:23<4:22:08,  1.75s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0412, 'learning_rate': 0.0001926033662990558, 'epoch': 3.0}                                                                                       
{'eval_loss': 0.0933597981929779, 'eval_runtime': 69.3143, 'eval_samples_per_second': 4.328, 'eval_steps_per_second': 0.548, 'epoch': 3.0}                   
 15%|████████████████▉                                                                                                | 1500/10000 [49:36<4:10:24,  1.77s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0332, 'learning_rate': 0.00018530629460647657, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.10056784749031067, 'eval_runtime': 69.3111, 'eval_samples_per_second': 4.328, 'eval_steps_per_second': 0.548, 'epoch': 4.0}                  
{'train_runtime': 3953.6406, 'train_samples_per_second': 2.529, 'train_steps_per_second': 2.529, 'train_loss': 0.15314015007019044, 'epoch': 4.0}            
 20%|██████████████████████▏                                                                                        | 2000/10000 [1:05:53<4:23:34,  1.98s/it]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- Llama-2-13B-hf-base-FT-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_8_p2

Start fine-tuning model.
{'loss': 1.2973, 'learning_rate': 0.0001997935295756461, 'epoch': 1.0}                                                                                       
{'eval_loss': 0.144582137465477, 'eval_runtime': 12.0245, 'eval_samples_per_second': 24.949, 'eval_steps_per_second': 3.16, 'epoch': 1.0}                    
{'loss': 0.0948, 'learning_rate': 0.00019745176468573893, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.09552514553070068, 'eval_runtime': 12.1133, 'eval_samples_per_second': 24.766, 'eval_steps_per_second': 3.137, 'epoch': 2.0}                 
{'loss': 0.0657, 'learning_rate': 0.00019255918721902235, 'epoch': 3.0}                                                                                      
{'eval_loss': 0.08796025812625885, 'eval_runtime': 12.194, 'eval_samples_per_second': 24.602, 'eval_steps_per_second': 3.116, 'epoch': 3.0}                  
{'loss': 0.0536, 'learning_rate': 0.00018524386102508755, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.08726830035448074, 'eval_runtime': 12.235, 'eval_samples_per_second': 24.52, 'eval_steps_per_second': 3.106, 'epoch': 4.0}                   
{'loss': 0.0434, 'learning_rate': 0.00017569726571560837, 'epoch': 5.0}                                                                                      
{'eval_loss': 0.0915578231215477, 'eval_runtime': 12.2935, 'eval_samples_per_second': 24.403, 'eval_steps_per_second': 3.091, 'epoch': 5.0}                  
{'loss': 0.0365, 'learning_rate': 0.00016416928466067723, 'epoch': 6.0}                                                                                      
{'eval_loss': 0.09381135553121567, 'eval_runtime': 12.2304, 'eval_samples_per_second': 24.529, 'eval_steps_per_second': 3.107, 'epoch': 6.0}                 
{'train_runtime': 324.1994, 'train_samples_per_second': 30.845, 'train_steps_per_second': 3.886, 'train_loss': 0.2651964167438487, 'epoch': 6.0}             
 30%|███████████████████████████████████                                                                                  | 378/1260 [05:24<12:36,  1.17it/s]
End fine-tuning model.
'''

'''
- Llama-2-13B-hf-base-FT-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_4_p2

Start fine-tuning model.
{'loss': 0.8904, 'learning_rate': 0.00019979028262377118, 'epoch': 1.0}                                                                                      
{'eval_loss': 0.11278064548969269, 'eval_runtime': 12.342, 'eval_samples_per_second': 24.307, 'eval_steps_per_second': 3.079, 'epoch': 1.0}                  
{'loss': 0.0769, 'learning_rate': 0.00019744105246469263, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.08916482329368591, 'eval_runtime': 12.3546, 'eval_samples_per_second': 24.282, 'eval_steps_per_second': 3.076, 'epoch': 2.0}                 
{'loss': 0.0561, 'learning_rate': 0.00019254212296427044, 'epoch': 3.0}                                                                                      
{'eval_loss': 0.08579087257385254, 'eval_runtime': 12.4836, 'eval_samples_per_second': 24.032, 'eval_steps_per_second': 3.044, 'epoch': 3.0}                 
{'loss': 0.0435, 'learning_rate': 0.00018522168236559695, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.09174492955207825, 'eval_runtime': 12.4151, 'eval_samples_per_second': 24.164, 'eval_steps_per_second': 3.061, 'epoch': 4.0}                 
{'loss': 0.0347, 'learning_rate': 0.00017567128158176953, 'epoch': 5.0}                                                                                      
{'eval_loss': 0.09652598202228546, 'eval_runtime': 12.4909, 'eval_samples_per_second': 24.018, 'eval_steps_per_second': 3.042, 'epoch': 5.0}                 
{'train_runtime': 285.7735, 'train_samples_per_second': 34.993, 'train_steps_per_second': 8.748, 'train_loss': 0.22033503341674804, 'epoch': 5.0}            
 25%|█████████████████████████████▎                                                                                       | 625/2500 [04:45<14:17,  2.19it/s]
End fine-tuning model.
'''

'''
- Llama-2-13B-hf-base-FT-BioInfoDataset_training_it_en_500_bilanced_final_with_index_template_and_sqlVariants_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_1_p2

Start fine-tuning model.
{'loss': 0.8916, 'learning_rate': 0.00019979028262377118, 'epoch': 1.0}                                                                                      
{'eval_loss': 0.11197379976511002, 'eval_runtime': 12.3706, 'eval_samples_per_second': 24.251, 'eval_steps_per_second': 3.072, 'epoch': 1.0}                 
{'loss': 0.0768, 'learning_rate': 0.00019744105246469263, 'epoch': 2.0}                                                                                      
{'eval_loss': 0.0892435684800148, 'eval_runtime': 12.4384, 'eval_samples_per_second': 24.119, 'eval_steps_per_second': 3.055, 'epoch': 2.0}                  
{'loss': 0.0556, 'learning_rate': 0.00019254212296427044, 'epoch': 3.0}                                                                                      
{'eval_loss': 0.08747170120477676, 'eval_runtime': 12.4136, 'eval_samples_per_second': 24.167, 'eval_steps_per_second': 3.061, 'epoch': 3.0}                 
{'loss': 0.0435, 'learning_rate': 0.00018522168236559695, 'epoch': 4.0}                                                                                      
{'eval_loss': 0.09188021719455719, 'eval_runtime': 12.3826, 'eval_samples_per_second': 24.227, 'eval_steps_per_second': 3.069, 'epoch': 4.0}                 
{'loss': 0.0344, 'learning_rate': 0.00017567128158176953, 'epoch': 5.0}                                                                                      
{'eval_loss': 0.09635037928819656, 'eval_runtime': 12.4335, 'eval_samples_per_second': 24.128, 'eval_steps_per_second': 3.056, 'epoch': 5.0}                 
{'train_runtime': 286.6036, 'train_samples_per_second': 34.891, 'train_steps_per_second': 8.723, 'train_loss': 0.22038063354492188, 'epoch': 5.0}            
 25%|█████████████████████████████▎                                                                                       | 625/2500 [04:46<14:19,  2.18it/s]
End fine-tuning model.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-Text2SQLDataset_1Epoch_TRAIN_BATCH_SIZE_1_FT-BioInfoDataset_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_1_p2

Start fine-tuning model.
{'loss': 0.5592, 'learning_rate': 0.00019979237383039745, 'epoch': 1.0}                                                                                                
{'eval_loss': 0.10906223207712173, 'eval_runtime': 6.2964, 'eval_samples_per_second': 47.646, 'eval_steps_per_second': 6.035, 'epoch': 1.0}                            
{'loss': 0.0674, 'learning_rate': 0.0001974483272899535, 'epoch': 2.0}                                                                                                 
{'eval_loss': 0.09429027885198593, 'eval_runtime': 6.2684, 'eval_samples_per_second': 47.859, 'eval_steps_per_second': 6.062, 'epoch': 2.0}                            
{'loss': 0.0491, 'learning_rate': 0.0001925543910508503, 'epoch': 3.0}                                                                                                 
{'eval_loss': 0.10057611763477325, 'eval_runtime': 6.2632, 'eval_samples_per_second': 47.899, 'eval_steps_per_second': 6.067, 'epoch': 3.0}                            
{'loss': 0.0383, 'learning_rate': 0.00018523862269959583, 'epoch': 4.0}                                                                                                
{'eval_loss': 0.10334677994251251, 'eval_runtime': 6.269, 'eval_samples_per_second': 47.854, 'eval_steps_per_second': 6.062, 'epoch': 4.0}                             
{'train_runtime': 345.1743, 'train_samples_per_second': 28.971, 'train_steps_per_second': 28.971, 'train_loss': 0.17849202823638916, 'epoch': 4.0}                     
 20%|█████████████████████████                                                                                                    | 2000/10000 [05:45<23:00,  5.79it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- secondTuning_4bit/Llama-3-70B-Instruct-base-finetune_Text2SQLDataset_1Epoch_TRAIN_BATCH_SIZE_4_4bit_FT-BioInfoDataset_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_1_p2

Start fine-tuning model.
  0%|                                                                                                                                        | 0/10000 [00:00<?, ?it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
  0%|▎                                                                                    | 30/10000 [00:55<4:48:57,  1.74s/it]  0%|▌                                                                                                   | 50/10000 [01:29<4:48:21,  1.74s/it]{'loss': 0.4788, 'learning_rate': 0.00019979858464286317, 'epoch': 1.0}                                                                       
{'eval_loss': 0.09496777504682541, 'eval_runtime': 71.0532, 'eval_samples_per_second': 4.222, 'eval_steps_per_second': 0.535, 'epoch': 1.0}   
  5%|████▉                                                                                              | 500/10000 [16:10<4:34:14,  1.73s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.058, 'learning_rate': 0.00019747732436460952, 'epoch': 2.0}                                                                        
{'eval_loss': 0.0922861099243164, 'eval_runtime': 70.7641, 'eval_samples_per_second': 4.239, 'eval_steps_per_second': 0.537, 'epoch': 2.0}    
 10%|█████████▊                                                                                        | 1000/10000 [32:21<4:21:51,  1.75s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0416, 'learning_rate': 0.00019261558583015707, 'epoch': 3.0}                                                                       
{'eval_loss': 0.09172787517309189, 'eval_runtime': 71.095, 'eval_samples_per_second': 4.22, 'eval_steps_per_second': 0.534, 'epoch': 3.0}     
 15%|██████████████▋                                                                                   | 1500/10000 [48:34<4:06:24,  1.74s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0335, 'learning_rate': 0.000185323190217045, 'epoch': 4.0}                                                                         
{'eval_loss': 0.09938681870698929, 'eval_runtime': 70.8347, 'eval_samples_per_second': 4.235, 'eval_steps_per_second': 0.536, 'epoch': 4.0}   
 20%|███████████████████▏                                                                            | 2000/10000 [1:04:45<3:50:40,  1.73s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0291, 'learning_rate': 0.00017579817830520644, 'epoch': 5.0}                                                                       
{'eval_loss': 0.09951154887676239, 'eval_runtime': 70.8238, 'eval_samples_per_second': 4.236, 'eval_steps_per_second': 0.537, 'epoch': 5.0}   
{'train_runtime': 4900.9321, 'train_samples_per_second': 2.04, 'train_steps_per_second': 2.04, 'train_loss': 0.12820840301513672, 'epoch': 5.0}                                                                                                                                             
 25%|████████████████████████                                                                        | 2500/10000 [1:21:40<4:05:02,  1.96s/it]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- secondTuning/Llama-2-13B-hf-base-Text2SQLDataset_1Epoch_TRAIN_BATCH_SIZE_1_FT-BioInfoDataset_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_1_p2

Start fine-tuning model.
{'loss': 0.4462, 'learning_rate': 0.00019979028262377118, 'epoch': 1.0}                                                                                                
{'eval_loss': 0.0956571102142334, 'eval_runtime': 11.7681, 'eval_samples_per_second': 25.493, 'eval_steps_per_second': 3.229, 'epoch': 1.0}                            
{'loss': 0.0567, 'learning_rate': 0.00019744105246469263, 'epoch': 2.0}                                                                                                
{'eval_loss': 0.08923602849245071, 'eval_runtime': 11.7711, 'eval_samples_per_second': 25.486, 'eval_steps_per_second': 3.228, 'epoch': 2.0}                           
{'loss': 0.0405, 'learning_rate': 0.00019254212296427044, 'epoch': 3.0}                                                                                                
{'eval_loss': 0.09075725823640823, 'eval_runtime': 11.7568, 'eval_samples_per_second': 25.517, 'eval_steps_per_second': 3.232, 'epoch': 3.0}                           
{'loss': 0.0309, 'learning_rate': 0.00018522168236559695, 'epoch': 4.0}                                                                                                
{'eval_loss': 0.09565151482820511, 'eval_runtime': 11.7817, 'eval_samples_per_second': 25.463, 'eval_steps_per_second': 3.225, 'epoch': 4.0}                           
{'train_runtime': 424.7127, 'train_samples_per_second': 23.545, 'train_steps_per_second': 23.545, 'train_loss': 0.1435529227256775, 'epoch': 4.0}                      
 20%|█████████████████████████                                                                                                    | 2000/10000 [07:04<28:18,  4.71it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- Llama-2-13B-hf-base-Text2SQLDataset_1Epoch_TRAIN_BATCH_SIZE_1_FT-BioInfoDataset_V4_WITH_VALIDATION_BioInfoDataset_validation_it_en_300_bilanced_final_with_index_template_and_sqlVariants_V4_MAX_20Epochs_BatchSize_4_p2

Start fine-tuning model.
{'loss': 0.8895, 'learning_rate': 0.00019979028262377118, 'epoch': 1.0}                                                                                                
{'eval_loss': 0.11133908480405807, 'eval_runtime': 11.7568, 'eval_samples_per_second': 25.517, 'eval_steps_per_second': 3.232, 'epoch': 1.0}                           
{'loss': 0.0766, 'learning_rate': 0.00019744105246469263, 'epoch': 2.0}                                                                                                
{'eval_loss': 0.08890051394701004, 'eval_runtime': 11.8112, 'eval_samples_per_second': 25.4, 'eval_steps_per_second': 3.217, 'epoch': 2.0}                             
{'loss': 0.0561, 'learning_rate': 0.00019254212296427044, 'epoch': 3.0}                                                                                                
{'eval_loss': 0.08593139797449112, 'eval_runtime': 11.9115, 'eval_samples_per_second': 25.186, 'eval_steps_per_second': 3.19, 'epoch': 3.0}                            
{'loss': 0.0435, 'learning_rate': 0.00018522168236559695, 'epoch': 4.0}                                                                                                
{'eval_loss': 0.09224473685026169, 'eval_runtime': 11.9171, 'eval_samples_per_second': 25.174, 'eval_steps_per_second': 3.189, 'epoch': 4.0}                           
{'loss': 0.0352, 'learning_rate': 0.00017567128158176953, 'epoch': 5.0}                                                                                                
{'eval_loss': 0.09631644189357758, 'eval_runtime': 11.9354, 'eval_samples_per_second': 25.135, 'eval_steps_per_second': 3.184, 'epoch': 5.0}                           
{'train_runtime': 280.472, 'train_samples_per_second': 35.654, 'train_steps_per_second': 8.914, 'train_loss': 0.2201583770751953, 'epoch': 5.0}                        
 25%|███████████████████████████████▊                                                                                               | 625/2500 [04:40<14:01,  2.23it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- Llama-2-13B-hf:

Total Parameters: 13,068,293,120
Trainable Parameters (LoRA): 52,428,800
Percentage of Trainable Parameters (LoRA): 0.4012%
'''

'''
- Llama-3-8B-Instruct:

Total Parameters: 8,057,524,224
Trainable Parameters (LoRA): 27,262,976
Percentage of Trainable Parameters (LoRA): 0.3384%
'''

'''
- Llama-3-70B-Instruct:

Total Parameters: 36,459,257,856
Trainable Parameters (LoRA): 131,072,000
Percentage of Trainable Parameters (LoRA): 0.3595%

Total Parameters: 70,684,778,496
Trainable Parameters (LoRA): 131,072,000
Percentage of Trainable Parameters (LoRA): 0.1854%
'''

'''
- firstTuning/Llama-3-8B-Instruct-base-Text2SQLDataset_splitted_Max20Epoch_TRAIN_BATCH_SIZE_1_FT_p2

Start fine-tuning model.
{'loss': 0.5757, 'learning_rate': 0.0001974431149363042, 'epoch': 1.0}                                                                                                 
{'eval_loss': 0.46209433674812317, 'eval_runtime': 135.092, 'eval_samples_per_second': 116.328, 'eval_steps_per_second': 14.546, 'epoch': 1.0}                         
{'loss': 0.4333, 'learning_rate': 0.0001852259501452907, 'epoch': 2.0}                                                                                                 
{'eval_loss': 0.43738502264022827, 'eval_runtime': 137.6951, 'eval_samples_per_second': 114.129, 'eval_steps_per_second': 14.271, 'epoch': 2.0}                        
{'loss': 0.4045, 'learning_rate': 0.0001641547290278784, 'epoch': 3.0}                                                                                                 
{'eval_loss': 0.4247114658355713, 'eval_runtime': 139.3828, 'eval_samples_per_second': 112.747, 'eval_steps_per_second': 14.098, 'epoch': 3.0}                         
{'loss': 0.3798, 'learning_rate': 0.00013642690019600347, 'epoch': 4.0}                                                                                                
{'eval_loss': 0.41724056005477905, 'eval_runtime': 137.5983, 'eval_samples_per_second': 114.209, 'eval_steps_per_second': 14.281, 'epoch': 4.0}                        
{'loss': 0.3565, 'learning_rate': 0.00010490522377528769, 'epoch': 5.0}                                                                                                
{'eval_loss': 0.4151022136211395, 'eval_runtime': 139.0294, 'eval_samples_per_second': 113.034, 'eval_steps_per_second': 14.134, 'epoch': 5.0}                         
{'loss': 0.3344, 'learning_rate': 7.287380538884602e-05, 'epoch': 6.0}                                                                                                 
{'eval_loss': 0.4171023368835449, 'eval_runtime': 139.6069, 'eval_samples_per_second': 112.566, 'eval_steps_per_second': 14.075, 'epoch': 6.0}                         
{'loss': 0.3144, 'learning_rate': 4.3661291989657815e-05, 'epoch': 7.0}                                                                                                
{'eval_loss': 0.42147526144981384, 'eval_runtime': 137.6921, 'eval_samples_per_second': 114.131, 'eval_steps_per_second': 14.271, 'epoch': 7.0}                        
{'train_runtime': 5252.1933, 'train_samples_per_second': 89.764, 'train_steps_per_second': 5.611, 'train_loss': 0.39980907940391863, 'epoch': 7.0}                     
 70%|█████████████████████████████████████████████████████████████████████████████████████▍                                    | 20629/29470 [1:27:32<37:30,  3.93it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- secondTuning_4bit/Llama-3-70B-Instruct-base-best_double_finetune_PROMPT_CHANGED

Start fine-tuning model.
  0%|                                                                                                                                    | 0/5000 [00:00<?, ?it/s]/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.3386, 'learning_rate': 0.00019747009043142555, 'epoch': 1.0}                                                                                           
{'eval_loss': 3.641910734586418e-05, 'eval_runtime': 115.2034, 'eval_samples_per_second': 2.604, 'eval_steps_per_second': 0.33, 'epoch': 1.0}                     
 10%|████████████                                                                                                            | 500/5000 [20:28<2:46:41,  2.22s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 0.00018528939004767376, 'epoch': 2.0}                                                                                              
{'eval_loss': 6.1970067690708674e-06, 'eval_runtime': 116.0173, 'eval_samples_per_second': 2.586, 'eval_steps_per_second': 0.328, 'epoch': 2.0}                   
 20%|███████████████████████▊                                                                                               | 1000/5000 [40:56<2:28:03,  2.22s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 0.00016424015900470587, 'epoch': 3.0}                                                                                              
{'eval_loss': 2.6967138637701282e-06, 'eval_runtime': 115.1754, 'eval_samples_per_second': 2.605, 'eval_steps_per_second': 0.33, 'epoch': 3.0}                    
 30%|███████████████████████████████████                                                                                  | 1500/5000 [1:01:23<2:09:26,  2.22s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 0.00013651113086894952, 'epoch': 4.0}                                                                                              
{'eval_loss': 1.6137294096552068e-06, 'eval_runtime': 115.5697, 'eval_samples_per_second': 2.596, 'eval_steps_per_second': 0.329, 'epoch': 4.0}                   
 40%|██████████████████████████████████████████████▊                                                                      | 2000/5000 [1:22:28<1:50:55,  2.22s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 0.00010498561546399193, 'epoch': 5.0}                                                                                              
{'eval_loss': 1.1139328535136883e-06, 'eval_runtime': 114.6718, 'eval_samples_per_second': 2.616, 'eval_steps_per_second': 0.331, 'epoch': 5.0}                   
 50%|██████████████████████████████████████████████████████████▌                                                          | 2500/5000 [1:42:54<1:32:28,  2.22s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 7.294168758624307e-05, 'epoch': 6.0}                                                                                               
{'eval_loss': 8.637840096525906e-07, 'eval_runtime': 114.9923, 'eval_samples_per_second': 2.609, 'eval_steps_per_second': 0.33, 'epoch': 6.0}                     
 60%|██████████████████████████████████████████████████████████████████████▏                                              | 3000/5000 [2:03:28<1:13:56,  2.22s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 4.371132741127553e-05, 'epoch': 7.0}                                                                                               
{'eval_loss': 7.258282721522846e-07, 'eval_runtime': 115.0833, 'eval_samples_per_second': 2.607, 'eval_steps_per_second': 0.33, 'epoch': 7.0}                     
 70%|███████████████████████████████████████████████████████████████████████████████████▎                                   | 3500/5000 [2:24:31<55:51,  2.23s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 2.033395573042952e-05, 'epoch': 8.0}                                                                                               
{'eval_loss': 6.521889872601605e-07, 'eval_runtime': 110.5545, 'eval_samples_per_second': 2.714, 'eval_steps_per_second': 0.344, 'epoch': 8.0}                    
 80%|███████████████████████████████████████████████████████████████████████████████████████████████▏                       | 4000/5000 [2:44:53<36:57,  2.22s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 5.240389992013606e-06, 'epoch': 9.0}                                                                                               
{'eval_loss': 6.19097249909828e-07, 'eval_runtime': 109.9135, 'eval_samples_per_second': 2.729, 'eval_steps_per_second': 0.346, 'epoch': 9.0}                     
 90%|███████████████████████████████████████████████████████████████████████████████████████████████████████████            | 4500/5000 [3:05:14<18:29,  2.22s/it/beegfs/home/mmetta/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/torch/utils/checkpoint.py:464: UserWarning: torch.utils.checkpoint: the use_reentrant parameter should be passed explicitly. In version 2.4 we will raise an exception if use_reentrant is not passed. use_reentrant=False is recommended, but if you need to preserve the current default behavior, you can pass use_reentrant=True. Refer to docs for more details on the differences between the two variants.
  warnings.warn(
{'loss': 0.0, 'learning_rate': 8.391627608350661e-11, 'epoch': 10.0}                                                                                              
{'eval_loss': 6.151358320494182e-07, 'eval_runtime': 110.1987, 'eval_samples_per_second': 2.722, 'eval_steps_per_second': 0.345, 'epoch': 10.0}                   
{'train_runtime': 12338.6103, 'train_samples_per_second': 0.405, 'train_steps_per_second': 0.405, 'train_loss': 0.03386713106308598, 'epoch': 10.0}               
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 5000/5000 [3:25:38<00:00,  2.47s/it]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED

Start fine-tuning model.
{'loss': 0.5407, 'learning_rate': 0.00019979028262377118, 'epoch': 1.0}                                                                                                           
{'eval_loss': 5.3807805670658126e-05, 'eval_runtime': 10.9751, 'eval_samples_per_second': 27.335, 'eval_steps_per_second': 3.462, 'epoch': 1.0}                                   
  8%|███████████▍                                                                                                                             | 834/10000 [02:32<24:07,  6.33it/s]
  8%|███████████▍                                                                                                                             | 835/10000 [02:32<23:41,  6.45  8%|███████████▎                                                                                                                           | 836/10000 [02:32<23:40,  6.45it  8%|███████████▏                                                                                                                         | 837/10000 [02:33<23:43,  6.44it/s  8%|███████████                                                                                                                         | 839/10000 [02:33<23:56,  6.38it/s]  9%|█████████████▏                                                                                                                                      | 890/10000 [02:41<23:59,  6.  9%|████████████▋                                                                                                                                  | 891/10000 [02:41<23:47,  6.38it/  9%|████████████▋                                                                                                                                 | 892/10000 [02:41<23:46,  6.39it/s  9%|████████████▋                                                                                                                                 | 893/10000 [02:42<23:48,  6.38it/s  9%|████████████▌                                                                                                                                | 894/10000 [02:42<24:19,  6.24it/s]{'loss': 0.0, 'learning_rate': 0.00019744105246469263, 'epoch': 2.0}                                                                                                                  
{'eval_loss': 1.5557119695586152e-05, 'eval_runtime': 10.8722, 'eval_samples_per_second': 27.593, 'eval_steps_per_second': 3.495, 'epoch': 2.0}                                       
{'loss': 0.0, 'learning_rate': 0.00019254212296427044, 'epoch': 3.0}                                                                                                                  
{'eval_loss': 7.82908591645537e-06, 'eval_runtime': 10.7243, 'eval_samples_per_second': 27.974, 'eval_steps_per_second': 3.543, 'epoch': 3.0}                                         
{'loss': 0.0, 'learning_rate': 0.00018522168236559695, 'epoch': 4.0}                                                                                                                  
{'eval_loss': 4.733726655103965e-06, 'eval_runtime': 10.711, 'eval_samples_per_second': 28.009, 'eval_steps_per_second': 3.548, 'epoch': 4.0}                                         
{'loss': 0.0, 'learning_rate': 0.00017567128158176953, 'epoch': 5.0}                                                                                                                  
{'eval_loss': 3.1747761113365414e-06, 'eval_runtime': 10.7176, 'eval_samples_per_second': 27.991, 'eval_steps_per_second': 3.546, 'epoch': 5.0}                                       
{'loss': 0.0, 'learning_rate': 0.000164140821963114, 'epoch': 6.0}                                                                                                                    
{'eval_loss': 2.273920699735754e-06, 'eval_runtime': 10.6822, 'eval_samples_per_second': 28.084, 'eval_steps_per_second': 3.557, 'epoch': 6.0}                                        
{'loss': 0.0, 'learning_rate': 0.00015093201623287631, 'epoch': 7.0}                                                                                                                  
{'eval_loss': 1.702709141682135e-06, 'eval_runtime': 10.692, 'eval_samples_per_second': 28.058, 'eval_steps_per_second': 3.554, 'epoch': 7.0}                                         
{'loss': 0.0, 'learning_rate': 0.00013639049369634876, 'epoch': 8.0}                                                                                                                  
{'eval_loss': 1.3150233826308977e-06, 'eval_runtime': 10.6654, 'eval_samples_per_second': 28.128, 'eval_steps_per_second': 3.563, 'epoch': 8.0}                                       
{'loss': 0.0, 'learning_rate': 0.00012089675630312754, 'epoch': 9.0}                                                                                                                  
{'eval_loss': 1.0482914376552799e-06, 'eval_runtime': 10.6668, 'eval_samples_per_second': 28.125, 'eval_steps_per_second': 3.562, 'epoch': 9.0}                                       
{'loss': 0.0, 'learning_rate': 0.00010485622221144484, 'epoch': 10.0}                                                                                                                 
{'eval_loss': 8.535730557923671e-07, 'eval_runtime': 10.6796, 'eval_samples_per_second': 28.091, 'eval_steps_per_second': 3.558, 'epoch': 10.0}                                       
{'loss': 0.0, 'learning_rate': 8.868861738047158e-05, 'epoch': 11.0}                                                                                                                  
{'eval_loss': 7.087282369866443e-07, 'eval_runtime': 10.6928, 'eval_samples_per_second': 28.056, 'eval_steps_per_second': 3.554, 'epoch': 11.0}                                       
{'loss': 0.0, 'learning_rate': 7.281699277636572e-05, 'epoch': 12.0}                                                                                                                  
{'eval_loss': 6.007065280755342e-07, 'eval_runtime': 10.6467, 'eval_samples_per_second': 28.178, 'eval_steps_per_second': 3.569, 'epoch': 12.0}                                       
{'loss': 0.0, 'learning_rate': 5.765665457425102e-05, 'epoch': 13.0}                                                                                                                  
{'eval_loss': 5.235644380263693e-07, 'eval_runtime': 10.6588, 'eval_samples_per_second': 28.146, 'eval_steps_per_second': 3.565, 'epoch': 13.0}                                       
{'loss': 0.0, 'learning_rate': 4.360429701490934e-05, 'epoch': 14.0}                                                                                                                  
{'eval_loss': 4.6238696427280956e-07, 'eval_runtime': 10.654, 'eval_samples_per_second': 28.158, 'eval_steps_per_second': 3.567, 'epoch': 14.0}                                       
{'loss': 0.0, 'learning_rate': 3.102762227218957e-05, 'epoch': 15.0}                                                                                                                  
{'eval_loss': 4.189216440408927e-07, 'eval_runtime': 10.6533, 'eval_samples_per_second': 28.16, 'eval_steps_per_second': 3.567, 'epoch': 15.0}                                        
{'loss': 0.0, 'learning_rate': 2.025571894372794e-05, 'epoch': 16.0}                                                                                                                  
{'eval_loss': 3.822150631549448e-07, 'eval_runtime': 10.6201, 'eval_samples_per_second': 28.248, 'eval_steps_per_second': 3.578, 'epoch': 16.0}                                       
{'loss': 0.0, 'learning_rate': 1.1570450926997655e-05, 'epoch': 17.0}                                                                                                                 
{'eval_loss': 3.618226571688865e-07, 'eval_runtime': 10.6759, 'eval_samples_per_second': 28.101, 'eval_steps_per_second': 3.559, 'epoch': 17.0}                                       
{'loss': 0.0, 'learning_rate': 5.199082004372957e-06, 'epoch': 18.0}                                                                                                                  
{'eval_loss': 3.48771408198445e-07, 'eval_runtime': 10.6604, 'eval_samples_per_second': 28.141, 'eval_steps_per_second': 3.565, 'epoch': 18.0}                                        
{'loss': 0.0, 'learning_rate': 1.30832912661093e-06, 'epoch': 19.0}                                                                                                                   
{'eval_loss': 3.4294495776521217e-07, 'eval_runtime': 10.6529, 'eval_samples_per_second': 28.161, 'eval_steps_per_second': 3.567, 'epoch': 19.0}                                      
{'loss': 0.0, 'learning_rate': 0.0, 'epoch': 20.0}                                                                                                                                    
{'eval_loss': 3.401482615572604e-07, 'eval_runtime': 10.6491, 'eval_samples_per_second': 28.172, 'eval_steps_per_second': 3.568, 'epoch': 20.0}                                       
{'train_runtime': 1812.5735, 'train_samples_per_second': 5.517, 'train_steps_per_second': 5.517, 'train_loss': 0.027035921400744702, 'epoch': 20.0}                                   
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 10000/10000 [30:12<00:00,  5.52it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED

Start fine-tuning model.
  0%|▍                                                                                                                                           | 72/2500  0%|▎                                                                                                               | 73/25000 [00:12<1:04:50,  6.41it/s]{'loss': 0.7923, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                               
{'eval_loss': 0.00018938357243314385, 'eval_runtime': 11.0122, 'eval_samples_per_second': 27.242, 'eval_steps_per_second': 3.451, 'epoch': 1.0}                                       
{'loss': 0.0001, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                               
{'eval_loss': 2.1330795789253898e-05, 'eval_runtime': 11.0721, 'eval_samples_per_second': 27.095, 'eval_steps_per_second': 3.432, 'epoch': 2.0}                                       
{'loss': 0.0, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                                   
{'eval_loss': 9.316127943748143e-06, 'eval_runtime': 10.7936, 'eval_samples_per_second': 27.794, 'eval_steps_per_second': 3.521, 'epoch': 3.0}                                        
{'loss': 0.0, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                                  
{'eval_loss': 5.3175826906226575e-06, 'eval_runtime': 10.753, 'eval_samples_per_second': 27.899, 'eval_steps_per_second': 3.534, 'epoch': 4.0}                                        
{'loss': 0.0, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                                  
{'eval_loss': 3.4135262012569e-06, 'eval_runtime': 10.7444, 'eval_samples_per_second': 27.922, 'eval_steps_per_second': 3.537, 'epoch': 5.0}                                          
{'loss': 0.0, 'learning_rate': 0.00019578173241879872, 'epoch': 6.0}                                                                                                                  
{'eval_loss': 2.3448817501048325e-06, 'eval_runtime': 10.7213, 'eval_samples_per_second': 27.982, 'eval_steps_per_second': 3.544, 'epoch': 6.0}                                       
{'loss': 0.0, 'learning_rate': 0.00019372067050063438, 'epoch': 7.0}                                                                                                                  
{'eval_loss': 1.6783534420028445e-06, 'eval_runtime': 10.7459, 'eval_samples_per_second': 27.918, 'eval_steps_per_second': 3.536, 'epoch': 7.0}                                       
{'loss': 0.0, 'learning_rate': 0.00019126651152015403, 'epoch': 8.0}                                                                                                                  
{'eval_loss': 1.231704914061993e-06, 'eval_runtime': 10.7394, 'eval_samples_per_second': 27.935, 'eval_steps_per_second': 3.538, 'epoch': 8.0}                                        
{'loss': 0.0, 'learning_rate': 0.00018842954907300236, 'epoch': 9.0}                                                                                                                  
{'eval_loss': 9.254705446437583e-07, 'eval_runtime': 10.7518, 'eval_samples_per_second': 27.902, 'eval_steps_per_second': 3.534, 'epoch': 9.0}                                        
{'loss': 0.0, 'learning_rate': 0.00018522168236559695, 'epoch': 10.0}                                                                                                                 
{'eval_loss': 7.077959480739082e-07, 'eval_runtime': 10.7674, 'eval_samples_per_second': 27.862, 'eval_steps_per_second': 3.529, 'epoch': 10.0}                                       
{'loss': 0.0, 'learning_rate': 0.0001816563663057211, 'epoch': 11.0}                                                                                                                  
{'eval_loss': 5.439569008558465e-07, 'eval_runtime': 10.7616, 'eval_samples_per_second': 27.877, 'eval_steps_per_second': 3.531, 'epoch': 11.0}                                       
{'loss': 0.0, 'learning_rate': 0.00017774855506796496, 'epoch': 12.0}                                                                                                                 
{'eval_loss': 4.2486450979595247e-07, 'eval_runtime': 10.7612, 'eval_samples_per_second': 27.878, 'eval_steps_per_second': 3.531, 'epoch': 12.0}                                      
{'loss': 0.0, 'learning_rate': 0.00017351463937072004, 'epoch': 13.0}                                                                                                                 
{'eval_loss': 3.3280684874625877e-07, 'eval_runtime': 10.752, 'eval_samples_per_second': 27.902, 'eval_steps_per_second': 3.534, 'epoch': 13.0}                                       
{'loss': 0.0, 'learning_rate': 0.00016897237772781044, 'epoch': 14.0}                                                                                                                 
{'eval_loss': 2.640547620558209e-07, 'eval_runtime': 10.7691, 'eval_samples_per_second': 27.857, 'eval_steps_per_second': 3.529, 'epoch': 14.0}                                       
{'loss': 0.0, 'learning_rate': 0.000164140821963114, 'epoch': 15.0}                                                                                                                   
{'eval_loss': 2.1371427294525347e-07, 'eval_runtime': 10.7405, 'eval_samples_per_second': 27.932, 'eval_steps_per_second': 3.538, 'epoch': 15.0}                                      
{'loss': 0.0, 'learning_rate': 0.00015904023730059228, 'epoch': 16.0}                                                                                                                 
{'eval_loss': 1.742109958513538e-07, 'eval_runtime': 10.7603, 'eval_samples_per_second': 27.88, 'eval_steps_per_second': 3.531, 'epoch': 16.0}                                        
{'loss': 0.0, 'learning_rate': 0.0001536920173648984, 'epoch': 17.0}                                                                                                                  
{'eval_loss': 1.4705973683248885e-07, 'eval_runtime': 10.7625, 'eval_samples_per_second': 27.875, 'eval_steps_per_second': 3.531, 'epoch': 17.0}                                      
{'loss': 0.0, 'learning_rate': 0.00014811859444908052, 'epoch': 18.0}                                                                                                                 
{'eval_loss': 1.246861387471654e-07, 'eval_runtime': 10.7793, 'eval_samples_per_second': 27.831, 'eval_steps_per_second': 3.525, 'epoch': 18.0}                                       
{'loss': 0.0, 'learning_rate': 0.00014234334542574906, 'epoch': 19.0}                                                                                                                 
{'eval_loss': 1.076729105875529e-07, 'eval_runtime': 10.766, 'eval_samples_per_second': 27.866, 'eval_steps_per_second': 3.53, 'epoch': 19.0}                                         
{'loss': 0.0, 'learning_rate': 0.00013639049369634876, 'epoch': 20.0}                                                                                                                 
{'eval_loss': 9.60200026156599e-08, 'eval_runtime': 10.7286, 'eval_samples_per_second': 27.963, 'eval_steps_per_second': 3.542, 'epoch': 20.0}                                        
{'loss': 0.0, 'learning_rate': 0.00013028500758979506, 'epoch': 21.0}                                                                                                                 
{'eval_loss': 8.658111738668595e-08, 'eval_runtime': 10.7646, 'eval_samples_per_second': 27.869, 'eval_steps_per_second': 3.53, 'epoch': 21.0}                                        
{'loss': 0.0, 'learning_rate': 0.00012405249563662537, 'epoch': 22.0}                                                                                                                 
{'eval_loss': 8.180342803143503e-08, 'eval_runtime': 10.7525, 'eval_samples_per_second': 27.9, 'eval_steps_per_second': 3.534, 'epoch': 22.0}                                         
{'loss': 0.0, 'learning_rate': 0.0001177190991579223, 'epoch': 23.0}                                                                                                                  
{'eval_loss': 7.609350660686687e-08, 'eval_runtime': 10.7257, 'eval_samples_per_second': 27.97, 'eval_steps_per_second': 3.543, 'epoch': 23.0}                                        
{'loss': 0.0257, 'learning_rate': 0.00011138861045295201, 'epoch': 24.0}                                                                                                              
{'eval_loss': 2.4607294108136557e-05, 'eval_runtime': 10.7417, 'eval_samples_per_second': 27.929, 'eval_steps_per_second': 3.538, 'epoch': 24.0}                                      
{'loss': 0.0, 'learning_rate': 0.00010493385915491052, 'epoch': 25.0}                                                                                                                 
{'eval_loss': 1.028189763019327e-05, 'eval_runtime': 10.748, 'eval_samples_per_second': 27.912, 'eval_steps_per_second': 3.536, 'epoch': 25.0}                                        
{'train_runtime': 2280.4802, 'train_samples_per_second': 10.963, 'train_steps_per_second': 10.963, 'train_loss': 0.03272304169671726, 'epoch': 25.0}                                  
 50%|█████████████████████████████████████████████████████████████████████▌                                                                     | 12500/25000 [38:00<38:00,  5.48it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''

'''
- secondTuning/Llama-3-8B-Instruct-base-best_double_finetune_PROMPT_CHANGED_loss_final

Start fine-tuning model.
{'loss': 0.8242, 'learning_rate': 0.00013333333333333334, 'epoch': 1.0}                                                                                                       
{'eval_loss': 0.04477979987859726, 'eval_runtime': 20.0647, 'eval_samples_per_second': 14.952, 'eval_steps_per_second': 1.894, 'epoch': 1.0}                                  
{'loss': 0.0292, 'learning_rate': 0.00019994755690455152, 'epoch': 2.0}                                                                                                       
{'eval_loss': 0.03531605005264282, 'eval_runtime': 19.5879, 'eval_samples_per_second': 15.316, 'eval_steps_per_second': 1.94, 'epoch': 2.0}                                   
{'loss': 0.0209, 'learning_rate': 0.0001995283421166614, 'epoch': 3.0}                                                                                                        
{'eval_loss': 0.03455667942762375, 'eval_runtime': 19.3072, 'eval_samples_per_second': 15.538, 'eval_steps_per_second': 1.968, 'epoch': 3.0}                                  
{'loss': 0.016, 'learning_rate': 0.00019869167087338907, 'epoch': 4.0}                                                                                                        
{'eval_loss': 0.03707951679825783, 'eval_runtime': 19.4116, 'eval_samples_per_second': 15.455, 'eval_steps_per_second': 1.958, 'epoch': 4.0}                                  
{'loss': 0.0135, 'learning_rate': 0.00019744105246469263, 'epoch': 5.0}                                                                                                       
{'eval_loss': 0.03818492963910103, 'eval_runtime': 19.5583, 'eval_samples_per_second': 15.339, 'eval_steps_per_second': 1.943, 'epoch': 5.0}                                  
{'train_runtime': 533.3061, 'train_samples_per_second': 46.877, 'train_steps_per_second': 46.877, 'train_loss': 0.18075579471588135, 'epoch': 5.0}                            
 10%|█████████████                                                                                                                     | 2500/25000 [08:53<1:19:59,  4.69it/s]
End fine-tuning model.
Saving new model completed.
Saving new model tokenizer completed.
'''