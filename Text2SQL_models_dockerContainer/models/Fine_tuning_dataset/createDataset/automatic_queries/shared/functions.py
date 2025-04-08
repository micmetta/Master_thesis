import json

def load_column_value_dict(path):
    with open(path, "r") as file:
        return json.load(file)

def format_value(value):
    return f"'{value}'" if isinstance(value, str) else value

# Function to convert query to training set format
def format_training_example(question, sql_query):
    return (
        "[INST] Write SQLite query to answer the following question given the database schema. "
        "Please wrap your code answer using ```: Schema: CREATE TABLE detailedstrain ( strainid number PRIMARY KEY, "
        "collection number, accessionnumber text, gmo boolean, gmoconstructioninfo text, registeredcollection boolean, "
        "collectiondate text, isolationdate text, identificationtechnique text, depositdate text, otherdenominations text, "
        "typestrain boolean, inclusiondate text, remarks text, pathogenicity text, beforenagoya boolean, availablefordis boolean, "
        "otherccnumbers text, commentontaxonomy text, mutantinformation text, userestrictions number, nagoyaconditions number, "
        "geoorigin number, cacronym text, confirmed boolean, organismtype text, riskgroup number, euquarantine boolean, "
        "strain number, recommendedtemperature number, dualuse boolean, substrate text, testedtgmin number, testedtgmax number, "
        "morganismtype text, sexualstate text, mtafile text, prodofmetabolites text, genus text, species text, meuquarantine boolean, "
        "isolationhabitat text, status text, infrasubspecificnames text, qps boolean, axenic boolean, genotype text, mriskgroup number, "
        "historyofdeposit text, enzymeproduction text, applications text, plasmids text, plasmidscollectionsfields text, ploidy text, "
        "interspecifichybrid boolean, scientificname text, ineva boolean, inmirri boolean, othernames text, isolationhost text, name text, "
        "abbreviation text, propagationhost text, cultivar text, lyticcicle text, pathotypeserotypetype text, storageconditions text, "
        "symptomatologyonisolationhost text, transmissionby text, contamination boolean, infectivitytested boolean, restrictions text, "
        "nagoyaprotocol text ); Question: {question} [/INST] "
        "Here is the SQLite query to answer to the question: {question}: ``` {sql_query} ```"
    ).format(question=question, sql_query=sql_query)

# Function to add new queries to the training set
def add_new_queries_to_training_set(data, new_queries):
    for query in new_queries:
        formatted_example = format_training_example(query["question"], query["sql_query_variants"])
        data.append({"text": formatted_example})

def add_new_queries_to_training_set_in_specific_positions(data, new_queries, position):
    for query in new_queries:
        formatted_example = format_training_example(query["question"], query["sql_query_variants"])
        data.insert(position, {"text": formatted_example})
        position += 1  # increases position by 1 for each query entered

# Function to remove placeholder examples
def remove_placeholder_examples(data):
    if isinstance(data, list):
        return data[1:] if len(data) > 0 else data  # Remove first element if list is not empty
    elif isinstance(data, dict):
        return {cat: [example for example in queries if example.get("question") != ";;"] for cat, queries in data.items()}
    return data

def find_corresponding_template(template_en, templates_it):
    for template_it in templates_it:
        if template_it["sql_template"] == template_en["sql_template"]:
            return template_it
    raise ValueError("No corresponding Italian template found.")

def get_template_by_index(index, templates_it):
    for template_it in templates_it:
        if template_it['index_template'] == index:
            return template_it
    return ValueError("No corresponding Italian template found.")

def generate_sql_variants(template_en, sql_format_dict):
    generated_sql_variants = set()
    
    # iterates through all possible variants defined in the template
    for possible_query_template in template_en["possible_sql_queries"]:
        formatted_query = possible_query_template.format(**sql_format_dict)
        generated_sql_variants.add(formatted_query)
    
    return generated_sql_variants