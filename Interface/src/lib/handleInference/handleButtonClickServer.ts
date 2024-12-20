"use server";

import { executeJobWithStatus } from '@/lib/slurm/executeJob';
import { executeQuery } from '@/lib/database/postgres/bioInfoDB/getData';
import { insertQueryData } from '@/lib/database/postgres/experimentalDB/insertData';

interface Response {
    output: string;
    columns: string[];
    rows: string[][];
    error: string | null;
}
export async function handleButtonClickServer(nlQuery: string, sessionId: string, model: string) {
    let response_execution_query_BioInfoDB = "";
    const result: Response = {
        output: '',
        columns: [],
        rows: [],
        error: null
    }
    try {
        // Step 1: Connect to SLURM to run the job
        const response_1 = await executeJobWithStatus(
            `${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES}/job_script_${model}.sh`,
            nlQuery,
            sessionId,
            model
        );
        if('output' in response_1){
            if(response_1.output.includes("Disallowed clause detected")){
                // I extract the disallowed clause and the generated query
                const backendOutput: string = response_1.output;
                const clauseMatch = backendOutput.match(/Disallowed clause detected: (GROUP BY|HAVING|ORDER BY|LIMIT|OFFSET|UNION|JOIN)/);
                const queryMatch = backendOutput.match(/Generated query (SELECT[\s\S]*?);/);
                const foundClause: string = clauseMatch ? clauseMatch[1] : "unknown";
                const generatedQuery: string = queryMatch ? queryMatch[1] : "no query detected";
                
                // I prepare the complete error message
                result.error = `Error: Disallowed clause (${foundClause}) in generated query: ${generatedQuery}`;
                response_execution_query_BioInfoDB = `Disallowed clause detected (${foundClause})`;

            } else if(response_1.output !== 'No SQL query found.'){
                result.output = response_1.output; // I immediately store the obtained querySQL in the object that I will return at the end so that I will definitely have it in the block finally
                
                // Step 2: Run the SQL query in the 'BioInfoDB' database to get the results
                const response_2 = await executeQuery(response_1.output);
                result.columns = response_2.columns;
                
                // I convert each line into an array of strings
                result.rows = response_2.rows.map(row =>
                    Object.values(row).map(value => String(value))
                );
                result.error = null;

            } else {
                result.error = "No SQL query generated. Try rephrasing the query using different words and/or change the model.";
                response_execution_query_BioInfoDB = 'No_SQL_query_found';
            }
        }
    } catch (error) {
        console.error("(handleButtonClickServer.ts) - " + error);
        response_execution_query_BioInfoDB = error instanceof Error ? error.message : 'GENERIC_ERROR';
        if (response_execution_query_BioInfoDB === 'CONNECTION_ERROR') {
            result.error = "Error connecting to database 'BioInfoDB', please try again.";
        } else if (response_execution_query_BioInfoDB === 'QUERY_ERROR') {
            result.error = "Error while executing query in 'BioInfoDB', please try again using different words and/or changing model.";
        } else if (response_execution_query_BioInfoDB.includes("The job was cancelled for the maximum waiting time")) {
            response_execution_query_BioInfoDB = 'TIMEOUT_ERROR';
            result.error = "Request cancelled due to exceeding maximum waiting time, please try again.";
        } else if(response_execution_query_BioInfoDB.includes("The job was deleted from SLURM")){
            response_execution_query_BioInfoDB = 'SLURM_ERROR';
            result.error = "Request deleted by SLURM, please try again.";
        } else {
            result.error = "Unexpected error, please try again. " + "(" + error + ")";
        }
    } finally{
        // Step 3: Insert a new row in the 'ExperimentalDB' database (asynchronous call)
        insertQueryData({
            question: nlQuery,
            querySQL: result.output,
            response_execute_query: response_execution_query_BioInfoDB,
            selectedModel: model,
        })
        .then(response => {
            if (response === 'insert_success') {
                console.log("(handleButtonClickServer.ts) Successful entry into 'ExperimentalDB' for the user: " + sessionId);
            }
        })
        .catch(error => {
            console.error("(handleButtonClickServer.ts) - Error during INSERT in 'ExperimentalDB' for user: " + sessionId + " ("+error+")");
        });
        
        return result; // return the result immediately without waiting for insertQueryData()
    }
}
