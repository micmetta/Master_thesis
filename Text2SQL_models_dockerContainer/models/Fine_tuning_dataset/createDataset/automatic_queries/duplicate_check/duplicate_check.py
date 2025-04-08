import os
import json
from dotenv import load_dotenv

# Load environment variables from configuration file
load_dotenv("configuration.env")

BASE_PATH = os.getenv("BASE_PATH")
PATH_DATASET_TRAINING = os.getenv("PATH_DATASET_TRAINING")
PATH_DATASET_VALIDATION = os.getenv("PATH_DATASET_VALIDATION")
PATH_DATASET_TEST = os.getenv("PATH_DATASET_TEST")

# Full dataset paths
training_dataset_path = os.path.join(BASE_PATH, PATH_DATASET_TRAINING) if PATH_DATASET_TRAINING else ""
validation_dataset_path = os.path.join(BASE_PATH, PATH_DATASET_VALIDATION) if PATH_DATASET_VALIDATION else ""
test_dataset_path = os.path.join(BASE_PATH, PATH_DATASET_TEST) if PATH_DATASET_TEST else ""


# Function to check for duplicates in training set or validation set
def check_duplicates_training_set_or_validation_set(dataset):
    seen = set()
    duplicates = []
    
    for example in dataset:
        # Extract the question and query from the text
        question = extract_between(example["text"], "Question:", "[/INST]").strip()
        sql_query = extract_between(example["text"], ": ```", "```").strip()
        
        key = (question, sql_query)
        if key in seen:
            duplicates.append(example)
        else:
            seen.add(key)
    return duplicates

# Function to check for duplicates in the test set
def check_duplicates_test_set(dataset):
    seen = set()
    duplicates = []
    
    for category, queries in dataset.items():
        for query in queries:
            key = (query["question"], query["sql_query_variants"][0])
            #print("key: ", key)
            if key in seen:
                duplicates.append(query)
            else:
                seen.add(key)
    return duplicates

# Function to check for duplicates between training set and validation set
def check_duplicates_btw_train_and_validation(training_dataset, validation_dataset):
    # create a set of tuples (question, sql_query) from the training set
    training_queries = {
        (extract_between(example["text"], "Question:", "[/INST]").strip(),
         extract_between(example["text"], ": ```", "```").strip())
        for example in training_dataset
    }
    # create a set of tuples (question, sql_query) from the validation set
    validation_queries = {
        (extract_between(example["text"], "Question:", "[/INST]").strip(),
         extract_between(example["text"], ": ```", "```").strip())
        for example in validation_dataset
    }
    
    # identify duplicates by finding the intersection of the two sets
    duplicates = training_queries.intersection(validation_queries)
    
    return duplicates

# Helper function to extract a part of text
def extract_between(text, start_marker, end_marker):
    start = text.find(start_marker) + len(start_marker)
    end = text.find(end_marker, start)
    return text[start:end].strip()

# Function to check for duplicates between training set and test set
def check_duplicates_btw_train_and_test(training_dataset, test_dataset):
    # create a set of tuples (question, sql_query) from the training set
    training_queries = {
        (extract_between(example["text"], "Question:", "[/INST]").strip(),
         extract_between(example["text"], ": ```", "```").strip())
        for example in training_dataset
    }
    
    duplicates = []
    for category, queries in test_dataset.items():
        for query in queries:
            key = (query["question"], query["sql_query_variants"][0])
            if (key in training_queries):
                duplicates.append(query)
    return duplicates

# Function to check for duplicates between validation set and test set
def check_duplicates_btw_validation_and_test(validation_dataset, test_dataset):
    validation_queries = {
        (extract_between(example["text"], "Question:", "[/INST]").strip(),
         extract_between(example["text"], ": ```", "```").strip())
        for example in validation_dataset
    }
    
    duplicates = []
    for category, queries in test_dataset.items():
        for query in queries:
            key = (query["question"], query["sql_query_variants"][0])
            if (key in validation_queries):
                duplicates.append(query)
    return duplicates



# Main logic
if training_dataset_path == "" and validation_dataset_path == "" and test_dataset_path == "":
    raise ValueError("Both PATH_DATASET_TRAINING and PATH_DATASET_TEST are empty. Please provide at least one dataset path.")

# if there is only the test set:
elif training_dataset_path == "" and validation_dataset_path == "":
    with open(test_dataset_path, "r") as file:
        test_dataset = json.load(file)
    duplicates = check_duplicates_test_set(test_dataset)
    if duplicates:
        print(f"Found {len(duplicates)} duplicate entries in the test set:")
        for duplicate in duplicates:
            print(json.dumps(duplicate, indent=4))
    else:
        print("No duplicates found in the test set.")

# if there is only the training set:
elif test_dataset_path == "" and validation_dataset_path == "":
    with open(training_dataset_path, "r") as file:
        training_dataset = json.load(file)
    duplicates = check_duplicates_training_set_or_validation_set(training_dataset)
    if duplicates:
        print(f"Found {len(duplicates)} duplicate entries in the training set:")
        for duplicate in duplicates:
            print(json.dumps(duplicate, indent=4))
    else:
        print("No duplicates found in the training set.")

# if there is only the validation set:
elif training_dataset_path == "" and test_dataset_path == "":
    with open(validation_dataset_path, "r") as file:
        validation_dataset = json.load(file)
    duplicates = check_duplicates_training_set_or_validation_set(validation_dataset)
    if duplicates:
        print(f"Found {len(duplicates)} duplicate entries in the validation set:")
        for duplicate in duplicates:
            print(json.dumps(duplicate, indent=4))
    else:
        print("No duplicates found in the validation set.")

# if all 3 datasets are present:
else:
    with open(training_dataset_path, "r") as file:
        training_dataset = json.load(file)
    with open(validation_dataset_path, "r") as file:
        validation_dataset = json.load(file)
    with open(test_dataset_path, "r") as file:
        test_dataset = json.load(file)

    # check for duplicates within individual datasets
    duplicates_in_training = check_duplicates_training_set_or_validation_set(training_dataset)
    duplicates_in_validation = check_duplicates_training_set_or_validation_set(validation_dataset)
    duplicates_in_test = check_duplicates_test_set(test_dataset)
    
    # check eventually duplicates between training set and validation set
    duplicates_btw_train_val = check_duplicates_btw_train_and_validation(training_dataset, validation_dataset)

    # check for cross-duplicates between training set (or validation set) and test set
    duplicates_btw_train_test = check_duplicates_btw_train_and_test(training_dataset, test_dataset)
    duplicates_btw_val_test = check_duplicates_btw_validation_and_test(validation_dataset, test_dataset)

    # print results
    if duplicates_in_training:
        print(f"Found {len(duplicates_in_training)} duplicate entries in the training set:")
        for duplicate in duplicates_in_training:
            print(json.dumps(duplicate, indent=4))
    else:
        print("\nNo duplicates found in the training set.")
    
    if duplicates_in_validation:
        print(f"Found {len(duplicates_in_validation)} duplicate entries in the validation set:")
        for duplicate in duplicates_in_validation:
            print(json.dumps(duplicate, indent=4))
    else:
        print("\nNo duplicates found in the validation set.")

    if duplicates_in_test:
        print(f"Found {len(duplicates_in_test)} duplicate entries in the test set:")
        for duplicate in duplicates_in_test:
            print(json.dumps(duplicate, indent=4))
    else:
        print("\nNo duplicates found in the test set.\n")
    
    if duplicates_btw_train_val:
        print(f"Found {len(duplicates_btw_train_val)} duplicate entries between the training set and validation set:")
        for duplicate in duplicates_btw_train_val:
            print(json.dumps(duplicate, indent=4))
    else:
        print("\nNo duplicates found between the training set and validation set.")

    if duplicates_btw_train_test:
        print(f"Found {len(duplicates_btw_train_test)} duplicate entries between the training set and test set:")
        for duplicate in duplicates_btw_train_test:
            print(json.dumps(duplicate, indent=4))
    else:
        print("\nNo duplicates found between the training set and test set.")

    if duplicates_btw_val_test:
        print(f"Found {len(duplicates_btw_val_test)} duplicate entries between the validation set and test set:")
        for duplicate in duplicates_btw_val_test:
            print(json.dumps(duplicate, indent=4))
    else:
        print("\nNo duplicates found between the validation set and test set.")
    print("\n")