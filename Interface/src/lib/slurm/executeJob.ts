import { exec } from 'child_process';
import { addJob, getJob, updateJobCanRemoved} from '../sharedDictionaryJobs/dictJobs.mjs';

const MAX_RETRY_ATTEMPTS = 3; // Numero massimo di tentativi
const RETRY_DELAY = 10000; // Ritardo tra un tentativo e l'altro (10 secondi)
const WAITJOBSTATUS = 1000; // Tempo di attesa per il polling del job

// Funzione per eseguire un comando SSH con tentativi di riconnessione
async function executeWithRetries(command: string, maxRetries: number, delay: number, userId: string, selectedModel: string): Promise<string> {
  let attempts = 0;

  return new Promise((resolve, reject) => {
    const tryExec = () => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          attempts++;
          console.error(`(slurm/executeJob.ts/executeWithRetries) Errore durante l'esecuzione del comando: ${error.message} - Tentativo: ${attempts}`);

          if (attempts < maxRetries) {
            console.log(`(slurm/executeJob.ts/executeWithRetries) Riprovo tra ${delay / 1000} secondi...`);
            setTimeout(tryExec, delay);
          } else {
            reject(`superato il numero massimo di tentativi di connessione a SLURM (${maxRetries})`);
          }
        } else if (stderr) {
          console.error(`(slurm/executeJob.ts/executeWithRetries) Errore (stderr): ${stderr}`);
          reject(stderr);
        } else {
          console.log(`\n\n(slurm/executeJob.ts/executeWithRetries) Connessione a SLURM avvenuta con successo per l'utente: ${userId} - Modello selezionato: ${selectedModel}\n`);
          resolve(stdout); // successo
        }
      });
    };
    tryExec();
  });
}

// Funzione per monitorare il completamento del job
async function checkJobCompletion(jobId: string): Promise<string | null> {
  return new Promise((resolve) => {
    const pollJobStatus = async () => {
      const jobIsCompleted = await getJob(jobId);
      switch (jobIsCompleted.status) {
        case 'running':
          break; // il job è ancora in esecuzione quindi non risolvo la promise per permettere che 'pollJobStatus()' continui a monitare il job
        case 'completed':
          resolve(jobIsCompleted.result); // il job è stato completato e quindi restituisco la querySQL che è stata generata dal modello
          return;
        case 'cancelled-by-slurm':
          resolve('cancelled-by-slurm');
          return;
        default:
          resolve('cancelled-timeout');
          return;
        }
      setTimeout(pollJobStatus, WAITJOBSTATUS); // attendo 'WAITJOBSTATUS' secondi prima di rieseguire la 'pollJobStatus()'
    };

    pollJobStatus(); // chiamata
  });
}

// Funzione principale per eseguire la richiesta di esecuzione di un job su SLURM e monitorare il suo stato
export async function executeJobWithStatus(command: string, question: string, userId: string, selectedModel: string): Promise<{ userId: string, output: string } | { error: string }> {
  if (!command || !question || !userId || !selectedModel) {
    throw new Error('Comando, domanda, userId o modello mancanti');
  }
  
  let sshCommand = ""
  if(selectedModel=="8B"){
    sshCommand = `ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "cd ${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES} && sbatch ${command} \\"${question}\\" \\"${userId}\\" \\"${process.env.PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_8B_SCRIPT}\\" \\"${process.env.PATH_MODEL_USE_REMOTE_FOR_INFERENCE_8B_SCRIPT}\\" \\"${process.env.EXPORT_PYTHON_PATH}\\""`;
  }else{
    sshCommand = `ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "cd ${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES} && sbatch ${command} \\"${question}\\" \\"${userId}\\" \\"${process.env.PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_70B_SCRIPT}\\" \\"${process.env.PATH_MODEL_USE_REMOTE_FOR_INFERENCE_70B_SCRIPT}\\" \\"${process.env.EXPORT_PYTHON_PATH}\\""`;
  }

  let jobIdFinally: string = '';
  try {
    const stdout = await executeWithRetries(sshCommand, MAX_RETRY_ATTEMPTS, RETRY_DELAY, userId, selectedModel);
    const jobIdMatch = stdout.match(/Submitted batch job (\d+)/);
    const jobId: string | null = jobIdMatch ? jobIdMatch[1] : null;

    if (jobId) {
      await addJob(jobId, userId, selectedModel);
      jobIdFinally = jobId;
      const querySQL = await checkJobCompletion(jobId);

      if((querySQL) && (querySQL != 'cancelled-by-slurm') && (querySQL != 'cancelled-timeout')){
        return { userId, output: querySQL };
      } else if(querySQL === 'cancelled-timeout'){
        throw new Error("Il job è stato cancellato per il tempo massimo in attesa. Riprovare.");
      } else{
        throw new Error("Il job è stato cancellato da SLURM. Attendere qualche secondo e riprovare.");
      }
    } else {
      throw new Error("Impossibile ottenere l'ID del job.");
    }
  } catch (error) {
    throw new Error(`generato in executeJobWithStatus.ts): ${error}`);

  } finally{
    if(jobIdFinally != ''){
      updateJobCanRemoved(jobIdFinally); // avviso il pooling di 'jobsManager.js' che il job che veniva monitorato può essere rimosso dalla struttura 
    }
  }
}
