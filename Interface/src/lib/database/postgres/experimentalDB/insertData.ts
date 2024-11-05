import experimentalDBPool from '@/lib/database/postgres/experimentalDB/initDB';

interface InsertData {
  question: string;
  querySQL: string;
  response_execute_query: string | null;
  selectedModel: string;
}

export async function insertQueryData(data: InsertData): Promise<string> {
  try {
    const { question, querySQL, response_execute_query, selectedModel } = data;

    // - Determino lo stato in base alla risposta ottenuta dal modello
    let status: string;
    if (
      response_execute_query === 'QUERY_ERROR' ||
      response_execute_query === 'GENERIC_ERROR' ||
      response_execute_query === 'CONNECTION_ERROR' ||
      response_execute_query === 'TIMEOUT_ERROR' ||
      response_execute_query === 'SLURM_ERROR' ||
      response_execute_query === 'Nessuna_query_SQL_generata'
    ) {
      status = response_execute_query;
    } else {
      status = 'QUERY_EXECUTE_SUCCESSFULLY';
    }

    // - Costruzione query di inserimento
    const queryInsert = `
      INSERT INTO querytest (question, querySQL, response_execute_query, model)
      VALUES ($1, $2, $3, $4)
    `;
    // - Esecuzione della query di inserimento
    await experimentalDBPool.query(queryInsert, [question, querySQL, status, selectedModel]);

    return 'insert_success';
  } catch (error) {
    console.error("(insertData.ts) ERRORE: ", error)
    throw new Error('insert_execution_error');
  }
}
