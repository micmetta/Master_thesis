import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
    StoppingCriteriaList
)
from peft import PeftModel
import time
import argparse
from utils.post_processing_functions import post_processing_query, save_query_to_file
from utils.stopping_criteria import StopOnTokens


def main():
    parser = argparse.ArgumentParser(description='Process natural language query.')
    parser.add_argument('--question', type=str, help='The natural language query to process')
    parser.add_argument('--session_id', type=str, help='Unique session ID for this job')
    parser.add_argument('--PATH_BASE_MODEL_REMOTE_FOR_INFERENCE', type=str, help='PATH_BASE_MODEL_REMOTE_FOR_INFERENCE')
    parser.add_argument('--PATH_MODEL_USE_REMOTE_FOR_INFERENCE', type=str, help='PATH_MODEL_USE_REMOTE_FOR_INFERENCE')

    args = parser.parse_args()
    question_NL = args.question
    session_id = args.session_id
    PATH_BASE_MODEL_REMOTE_FOR_INFERENCE = args.PATH_BASE_MODEL_REMOTE_FOR_INFERENCE
    PATH_MODEL_USE_REMOTE_FOR_INFERENCE = args.PATH_MODEL_USE_REMOTE_FOR_INFERENCE

    # print the parameters obtained from the next.js application
    print(f"\nProcessing question: {question_NL}")
    print(f"Session ID: {session_id}")
    print(f"PATH_BASE_MODEL_REMOTE_FOR_INFERENCE: {PATH_BASE_MODEL_REMOTE_FOR_INFERENCE}")
    print(f"PATH_MODEL_USE_REMOTE_FOR_INFERENCE: {PATH_MODEL_USE_REMOTE_FOR_INFERENCE}\n\n")

    # base model:
    model_name = PATH_BASE_MODEL_REMOTE_FOR_INFERENCE

    # fine-tuned model:
    new_model = PATH_MODEL_USE_REMOTE_FOR_INFERENCE

    # load model
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map = {"": 0}, # load model on GPU
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(base_model, new_model)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    stop_tokens = ["Ġ```", "Ġ[/"]  # list of termination tokens
    stop_token_ids = tokenizer.convert_tokens_to_ids(stop_tokens)
    print(f"Termination tokens: {stop_tokens}")
    print(f"Termination token IDs: {stop_token_ids}")
    stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)]) # definition of StoppingCriteria

    # create the pipeline
    pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

    # table: 'detailedstrain':
    table = "CREATE TABLE detailedstrain ( strainid number PRIMARY KEY, collection number, accessionnumber text, gmo boolean, gmoconstructioninfo text, registeredcollection boolean, collectiondate text, isolationdate text, identificationtechnique text, depositdate text, otherdenominations text, typestrain boolean, inclusiondate text, remarks text, pathogenicity text, beforenagoya boolean, availablefordis boolean, otherccnumbers text, commentontaxonomy text, mutantinformation text, userestrictions number, nagoyaconditions number, geoorigin number, cacronym text, confirmed boolean, organismtype text, riskgroup number, euquarantine boolean, strain number, recommendedtemperature number, dualuse boolean, substrate text, testedtgmin number, testedtgmax number, morganismtype text, sexualstate text, mtafile text, prodofmetabolites text, genus text, species text, meuquarantine boolean, isolationhabitat text, status text, infrasubspecificnames text, qps boolean, axenic boolean, genotype text, mriskgroup number, historyofdeposit text, enzymeproduction text, applications text, plasmids text, plasmidscollectionsfields text, ploidy text, interspecifichybrid boolean, scientificname text, ineva boolean, inmirri boolean, othernames text, isolationhost text, name text, abbreviation text, propagationhost text, cultivar text, lyticcicle text, pathotypeserotypetype text, storageconditions text, symptomatologyonisolationhost text, transmissionby text, contamination boolean, infectivitytested boolean, restrictions text, nagoyaprotocol text );"
    prompt = f"[INST] Write SQLite query to answer the following question given the database schema. Please wrap your code answer using ```: Schema: {table} Question: {question_NL} [/INST] Here is the SQLite query to answer to the question: {question_NL}: ``` "

    print(f"\n\nmodel.device: [{model.device}]\n\n")
    print(f"Current question:\n{question_NL}\n")
    print("Start generating query.")
    start_time = time.time()
    ans = pipe(prompt, max_new_tokens=200, stopping_criteria=stopping_criteria, temperature=1, top_k=1, top_p=0.0) # determinism
    end_time = time.time()
    print("End generating query.\n")
    print("Complete answer:")
    print(ans, "\n\n")
    print("ans[0]['generated_text'].split('```')[2]:")
    print(ans[0]['generated_text'].split('```')[2])
    original_query, ilike_query = post_processing_query(ans[0]['generated_text'].split('```')[2])
    print("Original Query after regex expression:", original_query)
    print("ilike Query after regex expression:", ilike_query)
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}\n\n\n\n")

    # saving the SQL query generated in the file with the session ID
    save_query_to_file(original_query, ilike_query, session_id)

    # Creating the dictionary with the output data
    result_dictionary = {
        "original SQL query generated": original_query,
        "Ilike SQL query generated": ilike_query,
        "Elapsed time": elapsed_time
    }
    print(f"Result dictionary: {result_dictionary}\n\n")


if __name__ == '__main__':
    main()