import random
import json
import os
from shared.functions import format_value, get_template_by_index, generate_sql_variants
from dotenv import load_dotenv

load_dotenv("configuration.env")

BASE_PATH = os.getenv("BASE_PATH")
PATH_AUTOMATIC_QUERIES_FOLDER = BASE_PATH + "/" + os.getenv("PATH_AUTOMATIC_QUERIES_FOLDER")


def load_c3_templates_test(language="en"):
    template_path = os.path.join(PATH_AUTOMATIC_QUERIES_FOLDER, f"templates/{language}/test/C3_templates.json")
    with open(template_path, "r") as file:
        return json.load(file)

def load_c3_templates_training(language="en"):
    template_path = os.path.join(PATH_AUTOMATIC_QUERIES_FOLDER, f"templates/{language}/training/C3_templates.json")
    with open(template_path, "r") as file:
        return json.load(file)


def generate_c3_queries(n, column_value_dict, existing_questions_training_set, existing_queries_training_set, 
                        existing_questions_validation_set, existing_queries_validation_set,
                        existing_questions_test_set_en, existing_queries_test_set_en, 
                        existing_questions_test_set_it, existing_queries_test_set_it, 
                        DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES):
    
    if(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="test"):
        c3_templates_en = load_c3_templates_test(language="en")
        c3_templates_it = load_c3_templates_test(language="it")
    else:
        c3_templates_en = load_c3_templates_training(language="en")
        c3_templates_it = load_c3_templates_training(language="it")

    queries_en = []
    queries_it = []
    generated_sql_queries = set()
    generated_questions_en = set()
    generated_questions_it = set()

    while len(queries_en) < n:
        template_en = random.choice(c3_templates_en)
        template_it = get_template_by_index(template_en["index_template"], c3_templates_it)
        index_template_en = template_en["index_template"]
        index_template_it = template_it["index_template"]
        sql_format_dict = {}

        # special case management
        if template_en["question_template"] == "Select '{column1}' and '{column2}' from strain.":
            question_en, typo_columns, sql_format_dict = generate_template_C3_with_2_columns(template_en, column_value_dict)
            question_it = template_it["question_template"].format(
                column1=typo_columns[0],
                column2=typo_columns[1]
            )
        elif template_en["question_template"] in ["Count the number of strains where {column1} is {value1} and {column2} is {value2}.",
                                                  "Select all strains where {column1} is {value1} or {column2} is {value2}."]:
            question_en, typo_columns, typo_values, sql_format_dict = generate_template_C3_with_2_columns_and_2_values(template_en, column_value_dict)
            question_it = template_it["question_template"].format(
                column1=typo_columns[0],
                value1=typo_values[0],
                column2=typo_columns[1],
                value2=typo_values[1],
            )
        else:
            # random generation of errors in column and value
            column = random.choice(list(column_value_dict.keys()))
            value = random.choice(column_value_dict[column])
            typo_column = introduce_typo(column)
            typo_value = introduce_typo(str(value))

            formatted_value = format_value(value)
            question_format_dict = {
                "column": typo_column,
                "value": typo_value
            }
            sql_format_dict = {
                "column": column,
                "value": formatted_value
            }
            question_en = template_en["question_template"].format(**question_format_dict)
            question_it = template_it["question_template"].format(**question_format_dict)

        # generate all query variants using separate function
        sql_variants = generate_sql_variants(template_en, sql_format_dict)
        sql_variants_list = list(sql_variants)

        sql_query_variants_en = []
        sql_query_variants_it = []
        for sql_variant in sql_variants_list:
            if ((question_en not in existing_questions_training_set) and
                (question_en not in existing_questions_validation_set) and
                (question_en not in existing_questions_test_set_en) and 
                (question_en not in generated_questions_en) and 
                (question_it not in existing_questions_training_set) and
                (question_it not in existing_questions_validation_set) and
                (question_it not in existing_questions_test_set_it) and
                (question_it not in generated_questions_it) and
                (sql_variant not in existing_queries_training_set) and
                (sql_variant not in existing_queries_validation_set) and
                (sql_variant not in existing_queries_test_set_en and sql_variant not in existing_queries_test_set_it) and 
                (sql_variant not in generated_sql_queries)
                ):

                # adding the current variant to the variant list for each language
                sql_query_variants_en.append(sql_variant)
                sql_query_variants_it.append(sql_variant)

                # adding the current 'sql_variant' to the generated_sql_queries only one time
                generated_sql_queries.add(sql_variant)
                
                # I will add "question_en" and "question_it" respectively in "generated_questions_en" and "generated_questions_it"
                # only if the current sql_variant is the last, otherwise I could have problem with 
                # "(question_en not in generated_questions_en)" and "(question_it not in generated_questions_it)" in the previous if(..)
                if sql_variant == sql_variants_list[-1]: 
                    generated_questions_en.add(question_en)
                    generated_questions_it.add(question_it)

        if sql_query_variants_en:
            
            if((DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="training") or 
               (DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="validation")):
                sql_query_variants_final_en = sql_query_variants_en[0] # saving only the first variant
                sql_query_variants_final_it = sql_query_variants_it[0]
            else:
                sql_query_variants_final_en = sql_query_variants_en # saving all variants
                sql_query_variants_final_it = sql_query_variants_it

            queries_en.append({
                "index_template": index_template_en,
                "question": question_en,
                "sql_query_variants": sql_query_variants_final_en
            })
            queries_it.append({
                "index_template": index_template_it,
                "question": question_it,
                "sql_query_variants": sql_query_variants_final_it
            })

    return queries_en, queries_it



def generate_template_C3_with_2_columns(template, column_value_dict):
    columns = random.sample(list(column_value_dict.keys()), 2)

    typo_column1 = introduce_typo(columns[0])
    typo_column2 = introduce_typo(columns[1])

    question_format_dict = {
        "column1": typo_column1,
        "column2": typo_column2
    }

    sql_format_dict = {
        "column1": columns[0],
        "column2": columns[1]
    }

    question = template["question_template"].format(**question_format_dict)

    return question, (typo_column1, typo_column2), sql_format_dict


def generate_template_C3_with_2_columns_and_2_values(template, column_value_dict):
    columns = random.sample(list(column_value_dict.keys()), 2)
    value1 = random.choice(column_value_dict[columns[0]])
    value2 = random.choice(column_value_dict[columns[1]])

    typo_column1 = introduce_typo(columns[0])
    typo_value1 = introduce_typo(str(value1))

    typo_column2 = introduce_typo(columns[1])
    typo_value2 = introduce_typo(str(value2))

    question_format_dict = {
        "column1": typo_column1,
        "value1": typo_value1,
        "column2": typo_column2,
        "value2": typo_value2,
    }

    sql_format_dict = {
        "column1": columns[0],
        "value1": format_value(value1),
        "column2": columns[1],
        "value2": format_value(value2),
    }

    question = template["question_template"].format(**question_format_dict)

    return question, (typo_column1, typo_column2), (typo_value1, typo_value2), sql_format_dict


def introduce_typo(word):
    """Introduce un errore casuale nel nome di una colonna, tabella o valore."""
    if len(word) > 3:
        pos = random.randint(1, len(word) - 2)
        typo = word[:pos] + random.choice('abcdefghijklmnopqrstuvwxyz') + word[pos + 1:]
        return typo
    return word