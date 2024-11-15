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
        // Step 1: Connettersi a SLURM per eseguire il job
        const response_1 = await executeJobWithStatus(
            `${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES}/job_script_${model}.sh`,
            nlQuery,
            sessionId,
            model
        );
        if ('output' in response_1 && response_1.output !== 'Nessuna query SQL trovata.') {
            result.output = response_1.output; // memorizzo subito la querySQL ottenuta, nell'oggetto che restituirò alla fine in modo tale da averla sicuramente nel blocco finally
            // Step 2: Eseguire la query SQL nel database 'BioInfoDB' per prendere i risultati
            const response_2 = await executeQuery(response_1.output);
            result.columns = response_2.columns;
            // converto ogni riga in un array di stringhe
            result.rows = response_2.rows.map(row =>
                Object.values(row).map(value => String(value))
            );
            result.error = null;
        } else{
            result.error = "Nessuna query SQL generata. Prova a riformulare la query utilizzando parole differenti e/o cambia modello.";
            response_execution_query_BioInfoDB = 'Nessuna_query_SQL_generata';
        }
    } catch (error) {
        console.error("(handleButtonClickServer.ts) - " + error);
        response_execution_query_BioInfoDB = error instanceof Error ? error.message : 'GENERIC_ERROR';
        if (response_execution_query_BioInfoDB === 'CONNECTION_ERROR') {
            result.error = "Errore di connessione al database 'BioInfoDB', riprova.";
        } else if (response_execution_query_BioInfoDB === 'QUERY_ERROR') {
            result.error = "Errore durante l'esecuzione della query in 'BioInfoDB', riprova utilizzando parole differenti e/o cambiando modello.";
        } else if (response_execution_query_BioInfoDB.includes("Il job è stato cancellato per il tempo massimo in attesa")) {
            response_execution_query_BioInfoDB = 'TIMEOUT_ERROR';
            result.error = "Richiesta cancellata per il superamento del tempo massimo di attesa, riprova.";
        } else if(response_execution_query_BioInfoDB.includes("l job è stato cancellato da SLURM")){
            response_execution_query_BioInfoDB = 'SLURM_ERROR';
            result.error = "Richiesta cancellata da SLURM, riprova.";
        } else {
            result.error = "Errore imprevisto, riprova. " + "(" + error + ")";
        }
    } finally{
        // Step 3: Inserire una nuova riga nel database 'ExperimentalDB' (chiamata asincrona)
        insertQueryData({
            question: nlQuery,
            querySQL: result.output,
            response_execute_query: response_execution_query_BioInfoDB,
            selectedModel: model,
        })
        .then(response => {
            if (response === 'insert_success') {
                console.log("(handleButtonClickServer.ts) Inserimento eseguito con successo in 'ExperimentalDB' per l'utente: " + sessionId);
            }
        })
        .catch(error => {
            console.error("(handleButtonClickServer.ts) - Errore durante l'INSERT in 'ExperimentalDB' per l'utente: " + sessionId + " ("+error+")");
        });
        
        return result; // restituisco subito il risultato senza attendere insertQueryData()
    }
}
