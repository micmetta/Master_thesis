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

    // - I determine the state based on the response obtained from the model
    let status: string;
    if (
      response_execute_query === 'QUERY_ERROR' ||
      response_execute_query === 'GENERIC_ERROR' ||
      response_execute_query === 'CONNECTION_ERROR' ||
      response_execute_query === 'TIMEOUT_ERROR' ||
      response_execute_query === 'SLURM_ERROR' ||
      response_execute_query === 'No_SQL_query_found' || 
      response_execute_query?.includes("Disallowed clause detected")
    ) {
      status = response_execute_query;
    } else {
      status = 'QUERY_EXECUTE_SUCCESSFULLY';
    }

    // - Query construction
    const queryInsert = `
      INSERT INTO querytest (question, querySQL, response_execute_query, model)
      VALUES ($1, $2, $3, $4)
    `;
    // - Executing the insertion query
    await experimentalDBPool.query(queryInsert, [question, querySQL, status, selectedModel]);

    return 'insert_success';
  } catch (error) {
    console.error("(insertData.ts) ERROR: ", error)
    throw new Error('insert_execution_error');
  }
}
