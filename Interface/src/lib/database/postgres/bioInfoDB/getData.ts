import bioInfoDBPool from "@/lib/database/postgres/bioInfoDB/initDB";

interface QueryResult {
  columns: string[];
  rows: Record<string, unknown>[]; // array di oggetti con chiavi stringa e valori di tipo sconosciuto
}

export async function executeQuery(query: string): Promise<QueryResult> {
  try {
    if (!query) {
      throw new Error('(bioInfoDB/getData.ts) Query SQL mancante');
    }
    const result = await bioInfoDBPool.query(query); // eseguo query
    const columnNames = result.fields.map(field => field.name); // estraggo i nomi delle colonne

    return {
      columns: columnNames,
      rows: result.rows // resultset
    };
  } catch (error: unknown) {
    if (error instanceof Error) {
      if ('code' in error && typeof error.code === 'string') {
        if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
          throw new Error('CONNECTION_ERROR');
        } else if (error.code.startsWith('42')) {
          throw new Error('QUERY_ERROR');
        }
      }
      throw new Error('GENERIC_ERROR');
    }
    throw new Error('GENERIC_ERROR');
  }
}
