import random
import json
import os
from shared.functions import format_value, get_template_by_index
from dotenv import load_dotenv

load_dotenv("configuration.env")

BASE_PATH = os.getenv("BASE_PATH")
PATH_AUTOMATIC_QUERIES_FOLDER = BASE_PATH + "/" + os.getenv("PATH_AUTOMATIC_QUERIES_FOLDER")

def load_c1_templates_test(language="en"):
    template_path = os.path.join(PATH_AUTOMATIC_QUERIES_FOLDER, f"templates/{language}/test/C1_templates.json")
    with open(template_path, "r") as file:
        return json.load(file)
    
def load_c1_templates_training(language="en"):
    template_path = os.path.join(PATH_AUTOMATIC_QUERIES_FOLDER, f"templates/{language}/training/C1_templates.json")
    with open(template_path, "r") as file:
        return json.load(file)


def generate_c1_queries(n, column_value_dict, existing_questions_training_set, existing_queries_training_set, 
                        existing_questions_validation_set, existing_queries_validation_set,
                        existing_questions_test_set_en, existing_queries_test_set_en, 
                        existing_questions_test_set_it, existing_queries_test_set_it,
                        DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES):
    
    # upload templates in both languages
    if(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="test"):
        c1_templates_en = load_c1_templates_test(language="en")
        c1_templates_it = load_c1_templates_test(language="it")
    else:
        # for validation set We will use the same templates used for build the training set
        c1_templates_en = load_c1_templates_training(language="en")
        c1_templates_it = load_c1_templates_training(language="it")

    queries_en = []
    queries_it = []
    generated_sql_queries = set()
    generated_questions_en = set()
    generated_questions_it = set()

    while len(queries_en) < n:
        template_en = random.choice(c1_templates_en) # select a random template for English
        template_it = get_template_by_index(template_en["index_template"], c1_templates_it)
        index_template_en = template_en["index_template"]
        index_template_it = template_it["index_template"]

        column = random.choice(list(column_value_dict.keys()))
        value = random.choice(column_value_dict[column])

        # generate the question and SQL query in both languages
        question_en = template_en["question_template"].format(column=column, value=value)
        question_it = template_it["question_template"].format(column=column, value=value)
        sql_query = template_en["sql_template"].format(
            column=column,
            value=format_value(value)
        )

        sql_query_variants_en = []
        sql_query_variants_it = []
        # control to avoid duplicates
        if ((question_en not in existing_questions_training_set) and
            (question_en not in existing_questions_validation_set) and
            (question_en not in existing_questions_test_set_en) and 
            (question_en not in generated_questions_en) and 
            (question_it not in existing_questions_training_set) and
            (question_it not in existing_questions_validation_set) and
            (question_it not in existing_questions_test_set_it) and
            (question_it not in generated_questions_it) and
            (sql_query not in existing_queries_training_set) and
            (sql_query not in existing_queries_validation_set) and
            (sql_query not in existing_queries_test_set_en and sql_query not in existing_queries_test_set_it) and 
            (sql_query not in generated_sql_queries)
            ):

            # adding generated questions and queries to corresponding sets
            generated_questions_en.add(question_en)
            generated_questions_it.add(question_it)

            # adding the current 'sql_query' to the generated_sql_queries only one time
            generated_sql_queries.add(sql_query)

            # in the case of category C1 there is only one variant for each query, so I add only this one
            sql_query_variants_en.append(sql_query)
            sql_query_variants_it.append(sql_query)

            if((DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="training") or 
               (DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="validation")):
                    sql_query_variants_final_en = sql_query_variants_en[0] # saving only the first variant
                    sql_query_variants_final_it = sql_query_variants_it[0]
            else:
                    sql_query_variants_final_en = sql_query_variants_en # saving all variants
                    sql_query_variants_final_it = sql_query_variants_it

            # adding generated queries to their respective lists
            queries_en.append({
                "index_template": index_template_en,
                "question": question_en,
                "sql_query_variants": sql_query_variants_final_en # saving the only variant (because we are in category C1) both training set and validation set
            })
            queries_it.append({
                "index_template": index_template_it,
                "question": question_it,
                "sql_query_variants": sql_query_variants_final_it
            })

    return queries_en, queries_it