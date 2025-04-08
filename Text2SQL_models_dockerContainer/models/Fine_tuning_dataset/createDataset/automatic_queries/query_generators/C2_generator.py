import random
import json
import os
from shared.functions import format_value, get_template_by_index, generate_sql_variants
from dotenv import load_dotenv

load_dotenv("configuration.env")

BASE_PATH = os.getenv("BASE_PATH")
PATH_AUTOMATIC_QUERIES_FOLDER = BASE_PATH + "/" + os.getenv("PATH_AUTOMATIC_QUERIES_FOLDER")

def load_c2_templates_test(language="en"):
    template_path = os.path.join(PATH_AUTOMATIC_QUERIES_FOLDER, f"templates/{language}/test/C2_templates.json")
    with open(template_path, "r") as file:
        return json.load(file)
    
def load_c2_templates_training(language="en"):
    template_path = os.path.join(PATH_AUTOMATIC_QUERIES_FOLDER, f"templates/{language}/training/C2_templates.json")
    with open(template_path, "r") as file:
        return json.load(file)


def generate_c2_queries(n, column_value_dict, existing_questions_training_set, existing_queries_training_set,
                        existing_questions_validation_set, existing_queries_validation_set,
                        existing_questions_test_set_en, existing_queries_test_set_en,
                        existing_questions_test_set_it, existing_queries_test_set_it,
                        DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES):

    if(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="test"):
        c2_templates_en = load_c2_templates_test(language="en")
        c2_templates_it = load_c2_templates_test(language="it")
    else:
        c2_templates_en = load_c2_templates_training(language="en")
        c2_templates_it = load_c2_templates_training(language="it")

    queries_en = []
    queries_it = []
    generated_sql_queries = set()
    generated_questions_en = set()
    generated_questions_it = set()

    while len(queries_en) < n:
        template_en = random.choice(c2_templates_en)
        template_it = get_template_by_index(template_en["index_template"], c2_templates_it)
        index_template_en = template_en["index_template"]
        index_template_it = template_it["index_template"]
        sql_format_dict = {}

        # special case management
        if template_en["question_template"] in ["Give me all strains that have {column1} equal to {value1} or {value2} or {value3}."]:
            question_en, column, values, sql_format_dict = generate_template_C2_1_column_on_3_values(template_en, column_value_dict)
            question_it = template_it["question_template"].format( # use the column values ​​and selected values ​selected in english ​to generate the question in Italian
                column1=column,
                value1=values[0],
                value2=values[1],
                value3=values[2]
            )

        elif template_en["question_template"] in ["Give me all strains that have {column1} equal to {value1} and the {column2} equal to {value2} and {column3} is {value3} or {value4}."]:
            question_en, columns, values, sql_format_dict = generate_template_C2_3_columns_and_4_values(template_en, column_value_dict)
            question_it = template_it["question_template"].format(
                column1=columns[0], column2=columns[1], column3=columns[2],
                value1=values[0], value2=values[1], value3=values[2], value4=values[3]
            )

        else:
            # generic case
            num_columns_needed = template_en["sql_template"].count("{column")
            columns = random.sample(list(column_value_dict.keys()), num_columns_needed)
            values = [random.choice(column_value_dict[column]) for column in columns]

            question_values = values[:]
            formatted_values = [format_value(value) for value in values]

            question_format_dict = {}
            sql_format_dict = {}
            for i in range(num_columns_needed):
                question_format_dict[f"column{i+1}"] = columns[i]
                question_format_dict[f"value{i+1}"] = question_values[i]
                sql_format_dict[f"column{i+1}"] = columns[i]
                sql_format_dict[f"value{i+1}"] = formatted_values[i]

            question_en = template_en["question_template"].format(**question_format_dict)
            question_it = template_it["question_template"].format(**question_format_dict)

        # generate all query variants using separate function
        sql_variants = generate_sql_variants(template_en, sql_format_dict)
        sql_variants_list = list(sql_variants)
        
        sql_query_variants_en = []
        sql_query_variants_it = []
        for sql_variant in sql_variants_list:
            # - Duplicate check, as we have only one training set which is in English language and two test sets in "it" and "en", so: 
            #   1) We don't add a "question_en" or a "question_it" that already exist in training set or test_set_en or test_set_it.
            #   2) Also, we don't add a "sql_variant" that already exist in:
            #       - existing_queries_training_set or 
            #       - existing_queries_test_set_en or 
            #       - existing_queries_test_set_it or
            #       - generated_sql_queries
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
                "sql_query_variants": sql_query_variants_final_en # saving the first variant
            })
            queries_it.append({
                "index_template": index_template_it,
                "question": question_it,
                "sql_query_variants": sql_query_variants_final_it # saving the first variant
            })

    return queries_en, queries_it


def generate_template_C2_1_column_on_3_values(template, column_value_dict):
    while True:
        column = random.choice(list(column_value_dict.keys()))
        if len(set(column_value_dict[column])) >= 3:
            break

    values = random.sample(set(column_value_dict[column]), 3)

    question_format_dict = {
        "column1": column,
        "value1": values[0],
        "value2": values[1],
        "value3": values[2]
    }

    sql_format_dict = {
        "column1": column,
        "value1": format_value(values[0]),
        "value2": format_value(values[1]),
        "value3": format_value(values[2])
    }

    question = template["question_template"].format(**question_format_dict)

    return question, column, values, sql_format_dict


def generate_template_C2_3_columns_and_4_values(template, column_value_dict):
    columns = random.sample(list(column_value_dict.keys()), 3)
    value1 = random.choice(column_value_dict[columns[0]])
    value2 = random.choice(column_value_dict[columns[1]])
    value3, value4 = random.sample(column_value_dict[columns[2]], 2) # I make sure these two values ​​are different

    question_format_dict = {
        "column1": columns[0], "value1": value1,
        "column2": columns[1], "value2": value2,
        "column3": columns[2], "value3": value3, "value4": value4
    }

    sql_format_dict = {
        "column1": columns[0], "value1": format_value(value1),
        "column2": columns[1], "value2": format_value(value2),
        "column3": columns[2], "value3": format_value(value3), "value4": format_value(value4)
    }

    question = template["question_template"].format(**question_format_dict)

    return question, columns, [value1, value2, value3, value4], sql_format_dict