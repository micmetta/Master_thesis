########################################################################################################################################################
# Script to test the performance of a given LLM on a given dataset.
########################################################################################################################################################

import os
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    BitsAndBytesConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
import re
import time
import threading
import csv
from datetime import datetime
from dotenv import load_dotenv
import json


# - Find the generated sql query inside the complete model response:
def extract_sql_query(generated_output):
    match = re.search(r'\b(SELECT)\b.*?(```|\n|$)', generated_output, re.IGNORECASE)
    if match:
        query_sql = match.group()
        print("")
        print("First occurrence of the SQL query found:", query_sql)
    else:
        query_sql = "No SQL queries found."
        print("No SQL queries found.")
    return query_sql

# Custom class to stop generation when stop token is generated
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id

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
PATH_TEST_SET = os.getenv("PATH_TEST_SET")
PATH_BASE_MODEL_TEST = os.getenv("PATH_BASE_MODEL_TEST")
PATH_OFFLOAD_BASE_MODEL_TEST = os.getenv("PATH_OFFLOAD_BASE_MODEL_TEST")
PATH_FT_MODEL_TEST = os.getenv("PATH_FT_MODEL_TEST")
PATH_OFFLOAD_MODEL_TEST = os.getenv("PATH_OFFLOAD_MODEL_TEST")
NAME_DIR_QUERY_RESULTS = os.getenv("NAME_DIR_QUERY_RESULTS")

# Retrieve database structure variable:
TABLE = os.getenv("TABLE")
print("")
print("TABLE: ", TABLE)
print("")

# Get all variables for the transformers library:
QUANTIZATION_BASE_MODEL = bool(os.getenv("QUANTIZATION_BASE_MODEL", "").lower() == "true")
STOPPING_CRITERIA = bool(os.getenv("STOPPING_CRITERIA", "").lower() == "true")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS"))
TEMPERATURE = float(os.getenv("TEMPERATURE"))
TOP_K = int(os.getenv("TOP_K"))
TOP_P = float(os.getenv("TOP_P"))

# Get token ID as termination criteria
STOP_TOKEN_ID = int(os.getenv("STOP_TOKEN_ID"))

# Loading dataset
PATH_TEST_SET = BASE_PATH+"/"+PATH_TEST_SET
with open(PATH_TEST_SET, "r") as file:
    dict_categories_questions = json.load(file)


# Base model:
path_base_model_test = BASE_PATH+"/"+PATH_BASE_MODEL_TEST
path_offload_base_model_test = BASE_PATH+"/"+PATH_OFFLOAD_BASE_MODEL_TEST

# Fine-tuned model to test:
path_model_FT = BASE_PATH+"/"+PATH_FT_MODEL_TEST
path_offload_model_FT = BASE_PATH+"/"+PATH_OFFLOAD_MODEL_TEST

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
        path_base_model_test,
        device_map = {"": 0},
        offload_folder=path_offload_base_model_test,
        cache_dir=path_base_model_test,
        torch_dtype=torch.float16,
        quantization_config=bnb_config,
    )
else:
    print("- I DO NOT PERFORM 4-BIT QUANTIZATION.")
    print("")
    base_model = AutoModelForCausalLM.from_pretrained(
        path_base_model_test,
        device_map = {"": 0},
        offload_folder=path_offload_base_model_test,
        cache_dir=path_base_model_test,
        torch_dtype=torch.float16,
)

# Check if the user wants to test only the base model or not:
model = None
tokenizer = None
if(PATH_FT_MODEL_TEST != ""):
    model = PeftModel.from_pretrained(base_model, path_model_FT)
    tokenizer = AutoTokenizer.from_pretrained(path_model_FT, trust_remote_code=True)
else:
    model = base_model
    tokenizer = AutoTokenizer.from_pretrained(path_base_model_test, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

stopping_criteria = None
if(STOPPING_CRITERIA):
    # convert the id to token to print
    token = tokenizer.convert_ids_to_tokens(STOP_TOKEN_ID)
    print("")
    print("")
    print(f"The token associated with the ID {STOP_TOKEN_ID} is: '{token}'")
    print("")
    print("")

    # definition of StoppingCriteria
    stopping_criteria = StoppingCriteriaList([StopOnToken(STOP_TOKEN_ID)])

# create the pipeline
pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

######################################################################################################################################
PATH_csv = BASE_PATH+"/venvText2SQL/models/Testing/on_dataset_without_postProcessing/Queries_results/"+NAME_DIR_QUERY_RESULTS+"/memory_usage.csv"
PATH_txt = BASE_PATH+"/venvText2SQL/models/Testing/on_dataset_without_postProcessing/Queries_results/"+NAME_DIR_QUERY_RESULTS+"/query_results.txt"
PATH_wrong_queries = BASE_PATH+"/venvText2SQL/models/Testing/on_dataset_without_postProcessing/Queries_results/"+NAME_DIR_QUERY_RESULTS+"/accuracy_results.txt"
PATH_category_results_json = BASE_PATH+"/venvText2SQL/models/Testing/on_dataset_without_postProcessing/Queries_results/"+NAME_DIR_QUERY_RESULTS+"/category_results.json"

# Extract only the directory path part:
PATH_csv_dir = os.path.dirname(PATH_csv)
PATH_txt_dir = os.path.dirname(PATH_txt)
PATH_wrong_queries_dir = os.path.dirname(PATH_wrong_queries)

# Create folders if they don't exist:
os.makedirs(PATH_csv_dir, exist_ok=True)
os.makedirs(PATH_txt_dir, exist_ok=True)
os.makedirs(PATH_wrong_queries_dir, exist_ok=True)
os.makedirs(os.path.dirname(PATH_category_results_json), exist_ok=True)

query_results = [] # will contain info about the answers provided by the model and the inference times for each query of each category
accuracy_results = [] # will contain all the info about the queries that the model failed and the overall accuracy result
current_question = 0
correct_answers = 0
total_test_queries = sum(len(questions) for questions in dict_categories_questions.values())
category_results = {category: {"incorrect": 0, "total": 0} for category in dict_categories_questions.keys()}
accuracy_results.append(f"\nSUMMARY OF INCORRECT QUERIES\n\n")
# I open the .csv file to record the monitoring data
with open(PATH_csv, mode="w", newline="") as file:
    csv_writer = csv.writer(file)

    # Iterate over the categories and questions in the test set:
    for current_category, questions in dict_categories_questions.items():
        current_question = 0
        query_results.append(f"Current category: {current_category}\n")
        csv_writer.writerow([])
        csv_writer.writerow(["Current category: " + str(current_category)])

        # Iterate over questions within the current category:
        for question_data in questions:
            current_question += 1
            question = question_data["question"]
            expected_sql_query = question_data["sql_query"]
            
            csv_writer.writerow([])
            csv_writer.writerow(["Current question: Q" + str(current_question)])
            csv_writer.writerow([question])
            csv_writer.writerow(["timestamp", "allocated_memory_GB", "reserved_memory_GB"])
            
            # Building the prompt:
            prompt = f"[INST] Write SQLite query to answer the following question given the database schema. Please wrap your code answer using ```: Schema: {TABLE} Question: {question} [/INST] Here is the SQLite query to answer to the question: {question}: ``` "
            
            print(f"\nCurrent question:\n{question}\n")
            print("Start generating query.")
            
            # Memory and inference time monitoring
            stop_event = threading.Event() # create an event to stop the monitoring thread
            monitor_thread = threading.Thread(target=monitor_memory, args=(1, stop_event, csv_writer))
            monitor_thread.start()
            
            start_time = time.time()
            if(STOPPING_CRITERIA):
                ans = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, stopping_criteria=stopping_criteria, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P)
            else:
                ans = pipe(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_k=TOP_K, top_p=TOP_P)
            end_time = time.time()
            
            stop_event.set() # signals the thread to stop
            monitor_thread.join() # waiting for the thread to finish
            print("End generating query.\n")

            # Extracting the generated SQL query
            print("full text (prompt + generated): ", ans)
            query = extract_sql_query(ans[0]['generated_text'].split('```')[2])
            elapsed_time = end_time - start_time
            print("")
            print("Query after regex expression:", query)
            print("Elapsed time:", elapsed_time)

            # Update query_results list:
            query_results.append(f"\tcurrent question: Q{current_question}\n")
            query_results.append(f"\tquestion: {question}\n")
            query_results.append(f"\tquery: {query}\n")
            query_results.append(f"\tTime taken: {elapsed_time} seconds\n\n")

            # Remove spaces and make both strings lowercase
            query_normalized = query.replace(" ", "").lower()
            expected_sql_query_normalized = expected_sql_query.replace(" ", "").lower()
            print("expected_sql_query_normalized: ", expected_sql_query_normalized)
            print("query_normalized: ", query_normalized)
            print("")
            print("")

            # check if the query generated by the model is the same as expected:
            if(query_normalized == expected_sql_query_normalized):
                correct_answers+=1
            else:
                # Update accuracy results list:
                accuracy_results.append(f"ID-query: {current_category}-Q{current_question}\n")
                accuracy_results.append(f"question: {question}\n")
                accuracy_results.append(f"real query: {expected_sql_query}\n")
                accuracy_results.append(f"generated query: {query}\n\n")
                category_results[current_category]["incorrect"] += 1

            category_results[current_category]["total"] += 1

accuracy_results.append(f"\n\nCorrect answers: {correct_answers}\n")
accuracy_results.append(f"Total questions: {total_test_queries}\n")
accuracy_model = correct_answers / total_test_queries if total_test_queries > 0 else 0
accuracy_results.append(f"Model accuracy: {accuracy_model * 100:.2f}%\n")
accuracy_results.append(f"\n######################################################################################\n\n")

# Saving results to file with path "PATH_txt":
with open(PATH_txt, "w") as file:
    file.writelines(query_results)

# Saving results to file with path "PATH_wrong_queries":
with open(PATH_wrong_queries, "w") as file:
    file.writelines(accuracy_results)

# Saving dictionary category results with path "PATH_category_results_json":
with open(PATH_category_results_json, "w") as json_file:
    json.dump(category_results, json_file, indent=4)

print("")
print("")
print("SAVING RESULTS COMPLETED SUCCESSFULLY (Queries Results folder updated).")
print("")
