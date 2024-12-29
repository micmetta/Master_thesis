import { Pool } from 'pg';

const experimentalDBPool = new Pool({
    host: process.env.EXPERIMENTAL_DB_HOST,
    port: Number(process.env.EXPERIMENTAL_DB_PORT),
    user: process.env.EXPERIMENTAL_DB_USER,
    password: process.env.EXPERIMENTAL_DB_PASSWORD,
    database: process.env.EXPERIMENTAL_DB_NAME,
});

export default experimentalDBPool;