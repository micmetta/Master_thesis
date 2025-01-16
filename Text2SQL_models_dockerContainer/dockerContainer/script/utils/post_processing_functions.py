import re
from rapidfuzz.process import extractOne
from rapidfuzz.distance import Levenshtein


def ensure_select_clause(generated_output):
    match = re.match(r'^\bSELECT\b', generated_output.strip(), re.IGNORECASE)  # check if the query starts with 'SELECT'
    if not match:
        generated_output = re.sub(r'^\w+', 'SELECT', generated_output.strip(),
                                  flags=re.IGNORECASE)  # replace the first word with 'SELECT'
        print("Adjusted query to start with SELECT:", generated_output)

    return generated_output


# Remove any incorrect tail characters (such as '[/')
def remove_trailing_characters(generated_output):
    generated_output = generated_output.strip()  # remove leading/trailing spaces
    if re.search(r'\[/\s*$', generated_output):
        generated_output = re.sub(r'\[/\s*$', '', generated_output)
        print("Removed trailing '[/' from the query:", generated_output)

    return generated_output


# 1) Force the SELECT clause includes only 'strainid' field.
# 2) Force the FROM clause includes only 'detailedstrain' table.
def control_SELECT_FROM(query_sql):
    query_sql = re.sub(r'\bSELECT\b.*?\bFROM\b', 'SELECT strainid FROM', query_sql, flags=re.IGNORECASE | re.DOTALL)
    print("The SELECT clause is been forced to includes only 'strainid' field: ", query_sql)
    if not re.search(r'\bFROM\s+detailedstrain\b', query_sql, re.IGNORECASE):
        query_sql = re.sub(r'\bFROM\s+\w+\b', 'FROM detailedstrain', query_sql, flags=re.IGNORECASE)
        print("The clause 'FROM detailedstrain' is been forced into query: ", query_sql)

    return query_sql


def handle_collection_column(query_sql):
    where_match = re.search(r'\bWHERE\b(.*)', query_sql, re.IGNORECASE | re.DOTALL)
    print("")
    print("where_match: ", where_match)
    if where_match:
        where_clause = where_match.group(1)
        print("where_clause: ", where_clause)
        collection_match = re.search(r'\bcollection\s*(=|!=|<>|LIKE|ILIKE)\s*(\'[^\']*\'|[^\s)]+)', where_clause,
                                     re.IGNORECASE)
        print("collection_match: ", collection_match)

        if collection_match:  # check if in the WHERE clause there is 'collection' column:
            value = collection_match.group(2)
            print("value: ", value)
            if not value.isdigit():  # if the value isn't a number
                print(f"Replacing of the column 'collection' with 'cacronym' (value: {value})")
                query_sql = re.sub(r'\bcollection\b', 'cacronym', query_sql, flags=re.IGNORECASE)
                print("query_sql after updating 'collection' column: ", query_sql)
    print("")

    return query_sql


# Fix any incorrect column names for 'detailededstrain' table.
def fix_column_names(query_sql):
    columns_detailedstrain = [
        "strainid", "collection", "accessionnumber", "gmo", "gmoconstructioninfo",
        "registeredcollection", "collectiondate", "isolationdate", "identificationtechnique",
        "depositdate", "otherdenominations", "typestrain", "inclusiondate", "remarks",
        "pathogenicity", "beforenagoya", "availablefordis", "otherccnumbers", "commentontaxonomy",
        "mutantinformation", "userestrictions", "nagoyaconditions", "geoorigin", "cacronym",
        "confirmed", "organismtype", "riskgroup", "euquarantine", "strain", "recommendedtemperature",
        "dualuse", "substrate", "testedtgmin", "testedtgmax", "morganismtype", "sexualstate",
        "mtafile", "prodofmetabolites", "genus", "species", "meuquarantine", "isolationhabitat",
        "status", "infrasubspecificnames", "qps", "axenic", "genotype", "mriskgroup", "historyofdeposit",
        "enzymeproduction", "applications", "plasmids", "plasmidscollectionsfields", "ploidy", "interspecifichybrid",
        "scientificname", "ineva", "inmirri", "othernames", "isolationhost", "name", "abbreviation", "propagationhost",
        "cultivar", "lyticcicle", "pathotypeserotypetype", "storageconditions", "symptomatologyonisolationhost",
        "transmissionby", "contamination", "infectivitytested", "restrictions", "nagoyaprotocol"
    ]
    normalized_columns = {col.lower() for col in columns_detailedstrain}

    column_regex = r"WHERE\s+([\s\S]+)"  # regex to catch everithing after 'WHERE'
    where_match = re.search(column_regex, query_sql,
                            re.IGNORECASE)  # search in 'query_sql' the first occurence of 'column_regex' pattern
    print("query_sql before of the check on the column names: ", query_sql)
    if where_match:
        where_clause = where_match.group(1)  # I catched everything after WHERE (not included)
        where_start = where_match.start(1)  # offset: start index of the WHERE clause in the query_sql
        column_names_where_clause = []  # to memorize the columns with start and end indexes

        # regex to find pair: (name_column, operator)
        individual_condition_regex = r"\b([a-zA-Z0-9_]+)\b\s*(=|<>|!=|>|<|>=|<=|IS\s+NULL|IS\s+NOT\s+NULL|LIKE|NOT\s+LIKE|BETWEEN|IN|NOT\s+IN|ILIKE|SIMILAR\s+TO)"

        for match in re.finditer(individual_condition_regex, where_clause, re.IGNORECASE):
            column_name = match.group(1)  # incorrect column name
            start_index = where_start + match.start(1)  # start index of the column found in the query_sql
            end_index = where_start + match.end(1)  # end index of the column found in the query_sql
            column_names_where_clause.append((column_name, start_index, end_index))

        print("Columns found in the query_sql with their indexes:", column_names_where_clause)

        # I search all columns that are in 'column_names_where_clause' but there aren't in 'normalized_columns'
        not_columns_of_detailedstrain = [
            (col, start, end) for col, start, end in column_names_where_clause if col.lower() not in normalized_columns
        ]
        print("Invalid columns with their indexes:", not_columns_of_detailedstrain)

        # I replace the wrong columns with the correct ones using the indexes
        if not_columns_of_detailedstrain:
            corrected_query_parts = []
            last_index = 0

            for col, start, end in not_columns_of_detailedstrain:
                similar_col, similarity = extractOne(col.lower(), normalized_columns,
                                                     scorer=Levenshtein.normalized_similarity)[:2]
                print(f"I replace '{col}' with '{similar_col}' (similarity: {similarity:.2f})")
                corrected_query_parts.append(query_sql[last_index:start])
                corrected_query_parts.append(similar_col)
                last_index = end

            corrected_query_parts.append(query_sql[last_index:])

            print("corrected_query_parts: ", corrected_query_parts)
            query_sql = "".join(corrected_query_parts)
            print("Updeted query with correct columns:", query_sql)

    return query_sql


# Transforms a SQL query by replacing each: [column = 'value'] with [column ILIKE '%value%'] only if 'value' contains no numbers.
def create_ilike_query(query_sql):
    condition_regex = re.compile(
        r"(\b\w+\b)\s*=\s*'([^']*)'",  # group 1: column, group 2: value
        re.IGNORECASE
    )

    def replace_condition(match):  # callback to replace each condition found
        column = match.group(1)
        value = match.group(2)
        if any(char.isdigit() for char in value):  # check if the value contains numbers
            print(f"The value '{value}' contains a number. Skipping replacement.")
            return match.group(0)  # returns the original condition without modifying it
        return f"{column} ILIKE '%{value}%'"  # replace column = 'value' with ILIKE column '%value%'

    # apply the replacement to all found occurrences
    updated_query_sql = condition_regex.sub(replace_condition, query_sql)

    return updated_query_sql


# - Simpler post-processing function because it only allows the use of the WHERE clause.
# - It locates the generated sql query within the complete response of the model and creates the subquery.
def post_processing_query(generated_output):
    generated_output = ensure_select_clause(generated_output)  # I make sure the query starts with SELECT
    match = re.search(r'\bSELECT\b(.*?)\bFROM\b', generated_output, re.IGNORECASE | re.DOTALL)

    if not match:
        query_sql = "No SQL query found."
        print("No SQL query found.")
        return query_sql

    query_sql = remove_trailing_characters(generated_output)

    # remove illegal clauses after WHERE or FROM (the WHERE clause might not even be there)
    clauses_from_WHERE_or_FROM_onwards = re.compile(r'(.*?\b(WHERE|FROM)\b.*?)(GROUP BY|HAVING|ORDER BY|LIMIT|OFFSET|UNION|JOIN)', re.IGNORECASE | re.DOTALL)
    match_clauses = clauses_from_WHERE_or_FROM_onwards.search(query_sql)
    if match_clauses:
        print(f"I found disallowed clauses after WHERE in the query: {query_sql}")
        query_sql = match_clauses.group(1)  # It only takes the part up to the WHERE clause
        print(f"I removed all the disallowed clauses, the new query is: {query_sql}")

    # deletion of any final ';' in the query
    if query_sql.strip().endswith(";"):
        query_sql = query_sql.strip()[:-1]
        print("I removed the final ';' from the query:", query_sql)

    query_sql = control_SELECT_FROM(query_sql)
    query_sql = handle_collection_column(query_sql)
    original_query_sql = fix_column_names(query_sql)
    ilike_query_sql = create_ilike_query(original_query_sql)

    print("final original_query_sql that I will insert as a subquery: ", original_query_sql)
    print("final ilike_query_sql that I will insert as a subquery: ", ilike_query_sql)

    original_subquery = f"({original_query_sql})"
    ilike_subquery = f"({ilike_query_sql})"

    # Creating the final queries that use the subquery
    original_final_query = (
        "SELECT syntheticstrain.* "
        f"FROM syntheticstrain "
        f"WHERE strainid IN {original_subquery};"
    )
    ilike_final_query = (
        "SELECT syntheticstrain.* "
        f"FROM syntheticstrain "
        f"WHERE strainid IN {ilike_subquery};"
    )

    print("Original final query with subquery:", original_final_query)
    print("Ilike final query with subquery:", ilike_final_query)

    return original_final_query, ilike_final_query


def save_query_to_file(original_query, ilike_query, session_id):
    separator = "\n---END-OF-QUERY---\n"  # separator between the two queries
    filename = f"output_querySQL_{session_id}.txt"  # unique identifier for the name of the file
    with open(filename, 'w') as file:
        file.write(original_query)
        file.write(separator)
        file.write(ilike_query)