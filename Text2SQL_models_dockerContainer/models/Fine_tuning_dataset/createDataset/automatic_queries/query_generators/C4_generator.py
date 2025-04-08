import random
import json
import os
from shared.functions import format_value, get_template_by_index, generate_sql_variants
from dotenv import load_dotenv

load_dotenv("configuration.env")

BASE_PATH = os.getenv("BASE_PATH")
PATH_AUTOMATIC_QUERIES_FOLDER = BASE_PATH + "/" + os.getenv("PATH_AUTOMATIC_QUERIES_FOLDER")
PATH_COLUMNS_DATE_VALUES_DB = BASE_PATH + "/" + os.getenv("PATH_COLUMNS_DATE_VALUES_DB")

with open(PATH_COLUMNS_DATE_VALUES_DB, "r") as file:
    column_data = json.load(file)
column_data_value_list = column_data["date_columns"]

def load_c4_templates_test(language="en"):
    template_path = os.path.join(PATH_AUTOMATIC_QUERIES_FOLDER, f"templates/{language}/test/C4_templates.json")
    with open(template_path, "r") as file:
        return json.load(file)
    
def load_c4_templates_training(language="en"):
    template_path = os.path.join(PATH_AUTOMATIC_QUERIES_FOLDER, f"templates/{language}/training/C4_templates.json")
    with open(template_path, "r") as file:
        return json.load(file)


def generate_c4_queries(n, column_value_dict, existing_questions_training_set, existing_queries_training_set, 
                        existing_questions_validation_set, existing_queries_validation_set,
                        existing_questions_test_set_en, existing_queries_test_set_en, 
                        existing_questions_test_set_it, existing_queries_test_set_it, 
                        DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES):
    
    if(DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES=="test"):
        c4_templates_en = load_c4_templates_test(language="en")
        c4_templates_it = load_c4_templates_test(language="it")
    else:
        c4_templates_en = load_c4_templates_training(language="en")
        c4_templates_it = load_c4_templates_training(language="it")

    queries_en = []
    queries_it = []
    generated_sql_queries = set()
    generated_questions_en = set()
    generated_questions_it = set()

    while len(queries_en) < n:
        template_en = random.choice(c4_templates_en)
        template_it = get_template_by_index(template_en["index_template"], c4_templates_it)
        index_template_en = template_en["index_template"]
        index_template_it = template_it["index_template"]
        sql_format_dict = {}

        # special case management
        if "I also want the" in template_en["question_template"]:
            question_en, question_it, sql_format_dict = generate_c4_between_dates_two_columns(template_en, template_it, column_value_dict)
        elif "between" in template_en["question_template"]:
            question_en, question_it, sql_format_dict = generate_c4_between_dates_single_column(template_en, template_it, column_value_dict)
        else:
            question_en, question_it, sql_format_dict = generate_c4_single_column_date(template_en, template_it, column_value_dict)

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


def generate_c4_single_column_date(template_en, template_it, column_value_dict):
    column_data = random.choice(list(column_data_value_list))
    column = random.choice(list(column_value_dict.keys()))
    value = random.choice(column_value_dict[column])

    year = random.randint(1990, 2015)
    month = random.randint(1, 12)
    if month in [1, 3, 5, 7, 8, 10, 12]:
        day = random.randint(1, 31)
    else:
        day = random.randint(1, 30)

    date = f"{year:04d}-{month:02d}-{day:02d}"

    question_format_dict_en = {
        "column_data": column_data,
        "column": column,
        "value": value,
        "year": year,
        "date": date
    }

    question_format_dict_it = question_format_dict_en  # same mapping for italian

    sql_format_dict = {
        "column_data": column_data,
        "column": column,
        "value": format_value(value),
        "year": year,
        "date": date
    }

    question_en = template_en["question_template"].format(**question_format_dict_en)
    question_it = template_it["question_template"].format(**question_format_dict_it)

    return question_en, question_it, sql_format_dict


def generate_c4_between_dates_single_column(template_en, template_it, column_value_dict):
    column_data = random.choice(list(column_data_value_list))
    column = random.choice(list(column_value_dict.keys()))
    value = random.choice(column_value_dict[column])

    start_year = random.randint(1990, 2005)
    end_year = start_year + random.randint(5, 15)

    question_format_dict_en = {
        "column_data": column_data,
        "column": column,
        "value": value,
        "start_year": start_year,
        "end_year": end_year
    }

    question_format_dict_it = question_format_dict_en

    sql_format_dict = {
        "column_data": column_data,
        "column": column,
        "value": format_value(value),
        "start_year": start_year,
        "end_year": end_year
    }

    question_en = template_en["question_template"].format(**question_format_dict_en)
    question_it = template_it["question_template"].format(**question_format_dict_it)

    return question_en, question_it, sql_format_dict


def generate_c4_between_dates_two_columns(template_en, template_it, column_value_dict):
    column_data = random.choice(list(column_data_value_list))
    column1, column2 = random.sample(list(column_value_dict.keys()), 2)
    value1 = random.choice(column_value_dict[column1])
    value2, value3 = random.sample(column_value_dict[column2], 2)

    start_year = random.randint(1990, 2005)
    end_year = start_year + random.randint(5, 15)

    question_format_dict_en = {
        "column_data": column_data,
        "colum1": column1,
        "value1": value1,
        "column2": column2,
        "value2": value2,
        "value3": value3,
        "start_year": start_year,
        "end_year": end_year
    }

    question_format_dict_it = question_format_dict_en

    sql_format_dict = {
        "column_data": column_data,
        "colum1": column1,
        "value1": format_value(value1),
        "column2": column2,
        "value2": format_value(value2),
        "value3": format_value(value3),
        "start_year": start_year,
        "end_year": end_year
    }

    question_en = template_en["question_template"].format(**question_format_dict_en)
    question_it = template_it["question_template"].format(**question_format_dict_it)

    return question_en, question_it, sql_format_dict