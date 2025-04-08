import os
import json
from dotenv import load_dotenv
from query_generators.C1_generator import generate_c1_queries
from query_generators.C2_generator import generate_c2_queries
from query_generators.C3_generator import generate_c3_queries
from query_generators.C4_generator import generate_c4_queries
from query_generators.C5_generator import generate_c5_queries
from shared.functions import load_column_value_dict, add_new_queries_to_training_set, add_new_queries_to_training_set_in_specific_positions, remove_placeholder_examples
import re

load_dotenv("configuration.env")

BASE_PATH = os.getenv("BASE_PATH")
LANGUAGE_OF_QUERIES_TO_GENERATE = os.getenv("LANGUAGE_OF_QUERIES_TO_GENERATE")
PATH_LAST_TRAINING_SET = BASE_PATH + "/" + "venvText2SQL/models/Fine_tuning_dataset/dataset/training/" + os.getenv("NAME_LAST_TRAINING_SET")
PATH_NAME_LAST_VALIDATION_SET = BASE_PATH + "/" + "venvText2SQL/models/Fine_tuning_dataset/dataset/validation/" + os.getenv("NAME_LAST_VALIDATION_SET")
PATH_LAST_TEST_SET_EN = BASE_PATH + "/venvText2SQL/models/Fine_tuning_dataset/dataset/test/en/" + os.getenv("NAME_LAST_TEST_SET_EN")
PATH_LAST_TEST_SET_IT = BASE_PATH + "/venvText2SQL/models/Fine_tuning_dataset/dataset/test/it/" + os.getenv("NAME_LAST_TEST_SET_IT")
PATH_COLUMNS_VALUES_DB = BASE_PATH + "/" + os.getenv("PATH_COLUMN_VALUES_DB")

DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES = os.getenv("DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES")
if((DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES != "training") and 
   (DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES != "validation") and 
   (DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES != "test")):
    raise ValueError(f"Invalid dataset: '{DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES}'. Allowed values are 'training' or 'validation' or 'test'.")

elif(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="training"):
    PATH_FILE_NAME_OUTPUT_TRAINING_SET = BASE_PATH + "/venvText2SQL/models/Fine_tuning_dataset/dataset/training/" + os.getenv("NAME_FILE_NAME_OUTPUT_TRAINING_SET")

elif(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="validation"):
    PATH_FILE_NAME_OUTPUT_VALIDATION_SET = BASE_PATH + "/venvText2SQL/models/Fine_tuning_dataset/dataset/validation/" + os.getenv("NAME_FILE_NAME_OUTPUT_VALIDATION_SET")

elif(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="test"):
    PATH_FILE_NAME_OUTPUT_TEST_SET_EN = BASE_PATH + "/venvText2SQL/models/Fine_tuning_dataset/dataset/test/en/" + os.getenv("NAME_FILE_NAME_OUTPUT_TEST_SET_EN")
    PATH_FILE_NAME_OUTPUT_TEST_SET_IT = BASE_PATH + "/venvText2SQL/models/Fine_tuning_dataset/dataset/test/it/" + os.getenv("NAME_FILE_NAME_OUTPUT_TEST_SET_IT")

N_QUERIES_TO_GENERATE_C1 = int(os.getenv("N_QUERIES_TO_GENERATE_C1"))
N_QUERIES_TO_GENERATE_C2 = int(os.getenv("N_QUERIES_TO_GENERATE_C2"))
N_QUERIES_TO_GENERATE_C3 = int(os.getenv("N_QUERIES_TO_GENERATE_C3"))
N_QUERIES_TO_GENERATE_C4 = int(os.getenv("N_QUERIES_TO_GENERATE_C4"))
N_QUERIES_TO_GENERATE_C5 = int(os.getenv("N_QUERIES_TO_GENERATE_C5"))

# Reading the dictionary of columns and possible values
column_value_dict = load_column_value_dict(PATH_COLUMNS_VALUES_DB)

with open(PATH_LAST_TRAINING_SET, "r") as file:
    data_previous_training_set = json.load(file)

with open(PATH_NAME_LAST_VALIDATION_SET, "r") as file:
    data_previous_validation_set = json.load(file)

with open(PATH_LAST_TEST_SET_EN, "r") as file:
    data_previous_test_set_en = json.load(file)

with open(PATH_LAST_TEST_SET_IT, "r") as file:
    data_previous_test_set_it = json.load(file)

# Pattern to extract the question and SQL query from training set
question_pattern = r"Question:\s*(.*?)\s*\[/INST\]"
query_pattern = r": ```\s*(.*?)\s*```"
existing_questions_training_set = { # extraction of questions
    re.search(question_pattern, example["text"]).group(1)
    for example in data_previous_training_set
    if re.search(question_pattern, example["text"])
}
existing_queries_training_set = { # extraction of queries
    re.search(query_pattern, example["text"]).group(1)
    for example in data_previous_training_set
    if re.search(query_pattern, example["text"])
}

# Pattern to extract the question and SQL query from validation set
question_pattern = r"Question:\s*(.*?)\s*\[/INST\]"
query_pattern = r": ```\s*(.*?)\s*```"
existing_questions_validation_set = {
    re.search(question_pattern, example["text"]).group(1)
    for example in data_previous_validation_set
    if re.search(question_pattern, example["text"])
}
existing_queries_validation_set = {
    re.search(query_pattern, example["text"]).group(1)
    for example in data_previous_validation_set
    if re.search(query_pattern, example["text"])
}

# Exatction questions and queries from test set
existing_questions_test_set_en = {query["question"] for category in data_previous_test_set_en.values() for query in category}
existing_queries_test_set_en = {query["sql_query"] for category in data_previous_test_set_en.values() for query in category}
existing_questions_test_set_it = {query["question"] for category in data_previous_test_set_it.values() for query in category}
existing_queries_test_set_it = {query["sql_query"] for category in data_previous_test_set_it.values() for query in category}

# Generate new queries
new_c1_queries_en, new_c1_queries_it = generate_c1_queries(N_QUERIES_TO_GENERATE_C1, column_value_dict, existing_questions_training_set, existing_queries_training_set, existing_questions_validation_set, existing_queries_validation_set, existing_questions_test_set_en, existing_queries_test_set_en, existing_questions_test_set_it, existing_queries_test_set_it, DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES)
new_c2_queries_en, new_c2_queries_it = generate_c2_queries(N_QUERIES_TO_GENERATE_C2, column_value_dict, existing_questions_training_set, existing_queries_training_set, existing_questions_validation_set, existing_queries_validation_set, existing_questions_test_set_en, existing_queries_test_set_en, existing_questions_test_set_it, existing_queries_test_set_it, DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES)
new_c3_queries_en, new_c3_queries_it = generate_c3_queries(N_QUERIES_TO_GENERATE_C3, column_value_dict, existing_questions_training_set, existing_queries_training_set, existing_questions_validation_set, existing_queries_validation_set, existing_questions_test_set_en, existing_queries_test_set_en, existing_questions_test_set_it, existing_queries_test_set_it, DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES)
new_c4_queries_en, new_c4_queries_it = generate_c4_queries(N_QUERIES_TO_GENERATE_C4, column_value_dict, existing_questions_training_set, existing_queries_training_set, existing_questions_validation_set, existing_queries_validation_set, existing_questions_test_set_en, existing_queries_test_set_en, existing_questions_test_set_it, existing_queries_test_set_it, DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES)
new_c5_queries_en, new_c5_queries_it = generate_c5_queries(N_QUERIES_TO_GENERATE_C5, column_value_dict, existing_questions_training_set, existing_queries_training_set, existing_questions_validation_set, existing_queries_validation_set, existing_questions_test_set_en, existing_queries_test_set_en, existing_questions_test_set_it, existing_queries_test_set_it, DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES)

print("")
print("len(new_c1_queries_en):", len(new_c1_queries_en))
print("len(new_c2_queries_en):", len(new_c2_queries_en))
print("len(new_c3_queries_en):", len(new_c3_queries_en))
print("len(new_c4_queries_en):", len(new_c4_queries_en))
print("len(new_c5_queries_en):", len(new_c5_queries_en))
print("")
print("len(new_c1_queries_it):", len(new_c1_queries_it))
print("len(new_c2_queries_it):", len(new_c2_queries_it))
print("len(new_c3_queries_it):", len(new_c3_queries_it))
print("len(new_c4_queries_it):", len(new_c4_queries_it))
print("len(new_c5_queries_it):", len(new_c5_queries_it))
print("")

# Adds new queries to existing categories
if(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="training"):

    # Added new queries to the training set respecting the prompt
    data = data_previous_training_set
    add_new_queries_to_training_set(data, new_c1_queries_en)
    add_new_queries_to_training_set(data, new_c1_queries_it)

    add_new_queries_to_training_set(data, new_c2_queries_en)
    add_new_queries_to_training_set(data, new_c2_queries_it)

    add_new_queries_to_training_set(data, new_c3_queries_en)
    add_new_queries_to_training_set(data, new_c3_queries_it)

    add_new_queries_to_training_set(data, new_c4_queries_en)
    add_new_queries_to_training_set(data, new_c4_queries_it)
    
    add_new_queries_to_training_set(data, new_c5_queries_en)
    add_new_queries_to_training_set(data, new_c5_queries_it)

    # Add queries to the specified locations
    #add_new_queries_to_training_set_in_specific_positions(data, new_c1_queries_it, 10)  # After the first 10 examples
    #add_new_queries_to_training_set_in_specific_positions(data, new_c2_queries_it, 30)  # After the first 30 examples
    #add_new_queries_to_training_set_in_specific_positions(data, new_c3_queries_it, 50)  # After the first 50 examples
    #add_new_queries_to_training_set_in_specific_positions(data, new_c4_queries_it, 100)  # After the first 70 examples
    #add_new_queries_to_training_set_in_specific_positions(data, new_c5_queries_it, len(data))  # At the bottom of the dataset

    data = remove_placeholder_examples(data)
    print("Number of examples in the training set:", len(data))
    with open(PATH_FILE_NAME_OUTPUT_TRAINING_SET, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"""\n\nDATASET CREATION COMPLETED:\nThe new queries have been added and saved in: {PATH_FILE_NAME_OUTPUT_TRAINING_SET}\n""")

elif(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="validation"):
    # Added new queries to the validation set respecting the prompt
    data = data_previous_validation_set
    add_new_queries_to_training_set(data, new_c1_queries_en)
    add_new_queries_to_training_set(data, new_c1_queries_it)

    add_new_queries_to_training_set(data, new_c2_queries_en)
    add_new_queries_to_training_set(data, new_c2_queries_it)

    add_new_queries_to_training_set(data, new_c3_queries_en)
    add_new_queries_to_training_set(data, new_c3_queries_it)

    add_new_queries_to_training_set(data, new_c4_queries_en)
    add_new_queries_to_training_set(data, new_c4_queries_it)
    
    add_new_queries_to_training_set(data, new_c5_queries_en)
    add_new_queries_to_training_set(data, new_c5_queries_it)

    data = remove_placeholder_examples(data)
    print("Number of examples in the validation set:", len(data))
    with open(PATH_FILE_NAME_OUTPUT_VALIDATION_SET, "w") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

    print(f"""\n\nDATASET CREATION COMPLETED:\nThe new queries have been added and saved in: {PATH_FILE_NAME_OUTPUT_VALIDATION_SET}\n""")


else:
    data_en = data_previous_test_set_en
    data_en["C1"].extend(new_c1_queries_en)
    data_en["C2"].extend(new_c2_queries_en)
    data_en["C3"].extend(new_c3_queries_en)
    data_en["C4"].extend(new_c4_queries_en)
    data_en["C5"].extend(new_c5_queries_en)

    data_it = data_previous_test_set_it
    data_it["C1"].extend(new_c1_queries_it)
    data_it["C2"].extend(new_c2_queries_it)
    data_it["C3"].extend(new_c3_queries_it)
    data_it["C4"].extend(new_c4_queries_it)
    data_it["C5"].extend(new_c5_queries_it)

    data_en = remove_placeholder_examples(data_en)
    data_it = remove_placeholder_examples(data_it)

    print(f"Number of examples in english validation set: {sum(len(queries) for queries in data_en.values())}")
    print(f"Number of examples in italian validation set: {sum(len(queries) for queries in data_it.values())}")

    # Save all queries (existing + new) to a new JSON file
    with open(PATH_FILE_NAME_OUTPUT_TEST_SET_EN, "w") as file:
        json.dump(data_en, file, ensure_ascii=False, indent=4)
    with open(PATH_FILE_NAME_OUTPUT_TEST_SET_IT, "w") as file:
        json.dump(data_it, file, ensure_ascii=False, indent=4)

    print(f"""\n\nDATASET CREATION COMPLETED:\nThe new queries have been added and saved in:
        - english: {PATH_FILE_NAME_OUTPUT_TEST_SET_EN}
        - italian: {PATH_FILE_NAME_OUTPUT_TEST_SET_IT}\n""")