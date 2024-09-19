from flask import Flask, request, jsonify, make_response
from flask_cors import CORS
from huggingface_hub import login
import torch
from transformers import (  # pip transformers==4.31.0
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteria,
    StoppingCriteriaList
)
from peft import PeftModel
import re
import time
import os
import sys
import psycopg2


# - Configurazione della connessione al database PostgreSQL chiamato 'bioInfoDB':
def connect_to_bioInfoDB():
    return psycopg2.connect(
        host="db",
        database="bioInfoDB",
        user="postgres",
        password="root",
        port="5432"
    )

# - Configurazione della connessione al database PostgreSQL chiamato 'experimentalDB':
def connect_to_experimentalDB():
    return psycopg2.connect(
        host="experimental_db",
        database="experimentalDB",
        user="postgres",
        password="root",
        port="5432" # c'era 5433
    )


# - Individua la query sql generata all'interno della risposta completa del modello:
def extract_sql_query(generated_output):
    #match = re.search(r'\b(SELECT)\b.*?(```|\n)', generated_output) # c'era prima
    match = re.search(r'\b(SELECT)\b.*?(```|\n|$)', generated_output, re.IGNORECASE)
    if match:
        query_sql = match.group()
        query_sql = query_sql.replace("`", "")  # rimuovo i backtick dalla query
        print("Prima occorrenza della query SQL trovata:", query_sql, file=sys.stderr)
    else:
        query_sql = "Nessuna query SQL trovata."
        print("Nessuna query SQL trovata.", file=sys.stderr)
    return query_sql


# - Classe custom per fermare la generazione quando viene generato il token di stop
class StopOnToken(StoppingCriteria):
    def __init__(self, stop_token_id):
        self.stop_token_id = stop_token_id

    def __call__(self, input_ids, scores, **kwargs):
        return input_ids[0, -1] == self.stop_token_id


# - Esecuzione della query nel DB
def execute_query_to_bioInfoDB(query):
    cursor = None
    connection = None
    try:
        connection = connect_to_bioInfoDB()
        print("", file=sys.stderr)
        print("", file=sys.stderr)
        print("connection eseguita con successo al database bioInfoDB.", file=sys.stderr)
        print("", file=sys.stderr)
        cursor = connection.cursor()
        print("cursor creato con successo per il database bioInfoDB.", file=sys.stderr)
        print("", file=sys.stderr)
        cursor.execute(query)
        print("query eseguita con successo sul database bioInfoDB.", file=sys.stderr)
        print("", file=sys.stderr)
        resultset = cursor.fetchall()
        print("resultset ottenuto con successo dal database bioInfoDB.", file=sys.stderr)
        print("", file=sys.stderr)
        # prendo i nomi delle colonne del resultset:
        column_names = [desc[0] for desc in cursor.description]
        print("nomi colonne del resultset ottenute con successo dal database bioInfoDB.", file=sys.stderr)
        print("", file=sys.stderr)

        return column_names, resultset

    except psycopg2.OperationalError as conn_error:
        # Errore specifico legato alla connessione
        print("Errore di connessione al database bioInfoDB.", file=sys.stderr)
        print(f"Errore: {conn_error}", file=sys.stderr)
        return CONNECTION_ERROR, CONNECTION_ERROR

    except psycopg2.DatabaseError as query_error:
        # Errore specifico legato all'esecuzione della query
        print("Errore durante l'esecuzione della query nel database bioInfoDB.", file=sys.stderr)
        print(f"Errore: {query_error}", file=sys.stderr)
        return QUERY_ERROR, QUERY_ERROR

    except Exception as e:
        # Qualsiasi altro errore non previsto
        print("Si è verificata un'eccezione imprevista per il database bioInfoDB.", file=sys.stderr)
        print(f"Errore: {e}", file=sys.stderr)
        return GENERIC_ERROR, GENERIC_ERROR

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()


# - Inserimento di una nuova riga nel DB 'experimentalDB'.
def update_experimentalDB(question, querySQL, response_execute_query):
    # - In base alla risposta ottenuta dal modello aggiorno la tabella 'querytest' del database 'experimentalDB':
    if ((response_execute_query == QUERY_ERROR) or (response_execute_query == GENERIC_ERROR) or (response_execute_query == CONNECTION_ERROR)):
        status = response_execute_query
    else:
        status = "query_executed_succesfully"

    # - Costruisco la query di inserimento
    query_insert = """
    INSERT INTO querytest (question, querySQL, response_execute_query)
    VALUES (%s, %s, %s)
    """

    # - Apro la connessione con il DB per l'aggiornamento:
    cursor = None
    connection = None
    try:
        connection = connect_to_experimentalDB()
        print("", file=sys.stderr)
        print("", file=sys.stderr)
        print("connection eseguita con successo al database experimental_db.", file=sys.stderr)
        print("", file=sys.stderr)
        cursor = connection.cursor()
        print("cursor creato con successo per il database experimental_db.", file=sys.stderr)

        print("", file=sys.stderr)
        cursor.execute(query_insert, (question, querySQL, status))
        connection.commit()  # confermo l'inserimento
        print("Query di inserimento eseguita con successo sul database experimental_db.", file=sys.stderr)
        print("", file=sys.stderr)
        return "insert_success"

    except psycopg2.OperationalError as conn_error:
        # Errore specifico legato alla connessione
        print("Errore di connessione al database experimental_db.", file=sys.stderr)
        print(f"Errore: {conn_error}", file=sys.stderr)
        return "insert_connection_error"

    except psycopg2.DatabaseError as query_error:
        # Errore specifico legato all'esecuzione della query
        print("Errore durante l'esecuzione della query nel database experimental_db.", file=sys.stderr)
        print(f"Errore: {query_error}", file=sys.stderr)
        return "insert_query_error"

    except Exception as e:
        # Qualsiasi altro errore non previsto
        print("Si è verificata un'eccezione imprevista per il database experimental_db.", file=sys.stderr)
        print(f"Errore: {e}", file=sys.stderr)
        return "insert_generic_error"

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()




############################################################################################################################
# 1) Caricamento token di HuggingFace e controllo dell'effettiva disponibilità del supporto CUDA per la libreria torch:
login(token="YOUR_HUGGINGFACE_TOKEN")  # my token huggingface
print("")
print("Supporto cuda per la libreria torch:", torch.cuda.is_available())  # verifico che torch sia stata installata con il supporto CUDA
############################################################################################################################

#############################################################################################
# 2) Istanzio la classe Flask per poter creare e gestire gli endpoints dell'applicazione web:
app = Flask(__name__)
CORS(app) # poichè il frontend e il backend sono separati bisogna gestire le richiste tramite cross-origin(CORS) in modo tale da permettere
# al backend di accettare le richiesta da altri domini
#############################################################################################

###########################################################################################################################################################
# 3) Istanzio il modello Llama3-8B usando il percorso relativo (portabilità):
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"  # modello base

# 4) Ottengo la directory corrente del file Python:
current_dir = os.path.dirname(os.path.abspath(__file__))

# 5) Costruisco i percorsi relativi sia per il modello base che per il modello finetune (portabilità).
#    I percorsi saranno costruiti in modo automaticamente corretto in base al sistema operativo
#    della macchina sulla quale verrà eseguito questo script:
base_model_path = os.path.join(current_dir, 'Llama3_8B_Instruct')
new_model_path = os.path.join(current_dir, 'modelli_riaddestrati', 'meta-llama', 'Meta-Llama-3-8B-Instruct-base-finetune_TRAINING_2_16bit_fp16_10Epochs')

# Verifica dei percorsi completo ottenuti:
print("")
print("Percorso completo modello base utilizzato:")
print(base_model_path)
print("")
print("Percorso completo modello finetune utilizzato:")
print(new_model_path)
print("")
print("")
###########################################################################################################################################################

###########################################################################################################################################################
# 6) Carico il modello d'interesse sulla GPU:
#    (tempi ovviamente minori rispetto a non usare la quantizzazione)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map={"": 0}, # GPU
    offload_folder=base_model_path,
    cache_dir=base_model_path,
    torch_dtype=torch.float16,
    load_in_8bit=True  # Opzionale: carico il modello in modalità 8-bit per ridurre ulteriormente l'uso della memoria.
)

model = PeftModel.from_pretrained(base_model, new_model_path)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Utilizzo del token ID 55375 come criterio di terminazione
stop_token_id = 55375 # è l'id del simbolo ```

# Converto l'ID al token associato per vedere se è il token che mi interessa (ovvero proprio: ```)
token = tokenizer.convert_ids_to_tokens(stop_token_id)
print("")
print("")
print(f"Il token associato all'ID {stop_token_id} è: '{token}'", file=sys.stderr)
print("")
print("")

# Definisco lo StoppingCriteria
stopping_criteria = StoppingCriteriaList([StopOnToken(stop_token_id)])

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)
###########################################################################################################################################################


###########################################################################################################################################################
# 7) Creazione tabella complessiva e prompt da fornire in input al modello per fare l'inferenza:
# - tabelle: 'virus' e 'strain':
table = "CREATE TABLE virus ( strain number PRIMARY KEY, vorganismtype text, isolationhost text, ineva boolean, inmirri boolean, name text, abbreviation text, othernames text, propagationhost text, scientificname text, veuquarantine boolean, vriskgroup integer, cultivar text, lyticcicle text, pathotypeserotypetype text, storageconditions text, symptomatologyisolationhost text, transmissionby text, contamination boolean, infectivitytested boolean, FOREIGN KEY (strain) REFERENCES strain(strainid)); CREATE TABLE strain ( strainid number PRIMARY KEY, collection number, accessionnumber text, gmo boolean, gmoconstructioninfo text, registeredcollection boolean, collectiondate text, isolationdate text, identificationtechnique text, depositdate text, otherdenominations text, typestrain boolean, inclusiondate text, remarks text, pathogenicity text, beforenagoya boolean, availablefordis boolean, otherccnumbers text, commentontaxonomy text, mutantinformation text, userestrictions number, nagoyaconditions number, geoorigin number );"
###########################################################################################################################################################

#####################################################
# - COSTANTI per lo stato di esecuzione della query
QUERY_ERROR = "query_error"
GENERIC_ERROR = "generic_error"
CONNECTION_ERROR = "connection_error"
#####################################################

###########################################################################################################################################################
# 8) ENDPOINTS:

# creazione di una route di test (GET - TEST)
@app.route('/test', methods=['GET'])
def test():
    print("", file=sys.stderr)
    print('Endpoint test funzionante!', file=sys.stderr)
    print("", file=sys.stderr)
    return make_response(jsonify({'message': 'test route'}), 200)


# Route per tradurre una query da NL a SQL
@app.route('/queryNL', methods=['GET'])
def text2sql():
    # Recupero la query scritta in NL:
    question = request.args.get('text', default='', type=str)
    print("Query in NL ricevuta: ", question, file=sys.stderr)  # file=sys.stderr (sys.stderr = flusso di messaggi di errore o log di Python)
    # serve per permettere a Docker di catturare la stampa nei logs del container.

    # - creazione prompt:
    prompt = f"[INST] Write SQLite query to answer the following question given the database schema. Please wrap your code answer using ```: Schema: {table} Question: {question} [/INST] Here is the SQLite query to answer to the question: {question}: ``` "

    print("Start generating query.", file=sys.stderr)
    start_time = time.time()
    #ans = pipe(prompt, max_new_tokens=200) # no determinismo e no tempo costante
    ans = pipe(prompt, max_new_tokens=200, stopping_criteria=stopping_criteria, temperature=1, top_k=1, top_p=0.0) # determinismo e tempo costante
    end_time = time.time()
    print("End generating query.", file=sys.stderr)
    print("Complete answer:", file=sys.stderr)
    print(ans, "\n\n", file=sys.stderr)
    print("ans[0]['generated_text'].split('```')[2]:", file=sys.stderr)
    print(ans[0]['generated_text'].split('```')[2], file=sys.stderr)

    list_answer = ans[0]['generated_text'].split('```')
    if(len(list_answer) > 2):
        # - Estraggo la query SQL dalla risposta del modello
        #  (anche per assicurarmi che all'inizio ci sia la SELECT in modo tale da evitare operazioni dannose nel DB)
        query = extract_sql_query(list_answer[2])
        # - Eseguo la query nel DB
        columns_names, response_execute_query = execute_query_to_bioInfoDB(query)
    else:
        # - Non estraggo la query SQL e non eseguo la query nel DB perchè è inutile
        query = "Nessuna query SQL trovata."
        columns_names = QUERY_ERROR
        response_execute_query = QUERY_ERROR

    ##############################################################################################################################################
    # - Aggiorno la tabella 'querytest' del database 'experimentalDB':
    new_test_added = update_experimentalDB(question, query, response_execute_query)
    ##############################################################################################################################################

    print("", file=sys.stderr)
    print("Query after regex expression:", query, file=sys.stderr)
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, file=sys.stderr)
    print("query: " + query)
    print("Time taken: " + str(elapsed_time) + "seconds", file=sys.stderr)
    print("", file=sys.stderr)
    print("", file=sys.stderr)


    # - Restituisco la risposta come JSON:
    return make_response(jsonify({'input_text': question, 'SQL_query_generated': query, 'columns_names':columns_names, 'response_execution_query': response_execute_query, 'new_test_added': new_test_added}), 200)


print("", file=sys.stderr)
print("", file=sys.stderr)
print("Listening for Natural Language queries..", file=sys.stderr)
###########################################################################################################################################################

# C:\Users\miky9\OneDrive\Desktop\PycharmProjects\pythonProject\Tesi magistrale\RestService