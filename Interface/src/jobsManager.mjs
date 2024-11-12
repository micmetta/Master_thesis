import { deleteALLInferenceFiles, getResultAfterWait, checkExistJob, cancelJobInSLURM } from './lib/slurm/helpersJobsManager.mjs';
import {
  getActiveJobs,
  getJob,
  getALLKeysJobs,
  getNumControlJob,
  getNumTotalControlJob,
  getSelectedModelJob,
  updateJobStatus,
  updateIncrementJobNumControl,
  updateResetJobNumControl,
  updateIncrementJobNumTotalControl,
  deleteALLCompletedORCancelledJobs 
} from './lib/sharedDictionaryJobs/dictJobs.mjs';


const NUM_MAX_AVAILABLE_MACHINES = 4 // numero max di gh disponibili
const NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS = 3 // numero massimo di tentativi consecutivi di cancellare i files su SLURM (creati dai jobs completati) che non sono andati a buon fine
const LIMIT_NUM_CONTROL_JOB_MODEL_8B = 16 //(8B: 12 solo per il caricamento del modello (2,5s * 12 = 30s) + 4 per queries più complesse (2.5s * 4 = 10s) = 16 (max 40s prima del controllo con 'checkExistJob(jobId)' - se è l'unico job in corso, altrimenti il tempo prima del controllo aumenterà))
const LIMIT_NUM_CONTROL_JOB_MODEL_70B = 42 //(70B: 30 solo per il caricamento del modello (2,5s * 30 = 75s) + 12 per queries più complesse (2.5s * 12 = 30s) = 42 (max 105s prima del controllo con 'checkExistJob(jobId)' - se è l'unico job in corso, altrimenti il tempo prima del controllo aumenterà))
const LIMIT_MAX_NUM_CONTROL_JOB_MODEL_8B = 40 // numero max di tentativi di lettura del file di output di un job dopo i quali cancellerò quest'ultimo da SLURM e notificherò l'utente (per il modello 8B) (sono circa 100s di attesa - se è l'unico job in corso, altrimenti il tempo prima della cancellazione aumenterà)
const LIMIT_MAX_NUM_CONTROL_JOB_MODEL_70B = 100 // numero max di tentativi di lettura del file di output di un job dopo i quali cancellerò quest'ultimo da SLURM e notificherò l'utente (per il modello 70B) (sono circa 250s di attesa - se è l'unico job in corso, altrimenti il tempo prima dela cancellazione aumenterà)
let limit_numControlJob_currentJob = 0 // valore iniziale è 0 tanto verrà modificato dinamicamente nel for che cicla sui jobs
let limit_numMaxTotalControlJob_currentJob = 0; // valore iniziale è 0 tanto verrà modificato dinamicamente nel for che cicla sui jobs
let max_limit = false // verrà settato a true nel momento in cui 'failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps' sarà arrivato almeno al valore di 'NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS' (questo booleano è utile nell'ottimizzazione delle richieste da fare a SLURM) 
let isPollingStarted = false; // serve per evitare di eseguire più di una volta la funzione 'startBatchPolling()' lanciata dallo script "instrumentation.js"
let POOLINGINTERVAL = 2500;
let WAITBEFOREEXEC = 1000;
let WAITBETWEENJOBS = 2500;
let failedAttempts = { // mi permette di passare per riferimento il numero massimo di tentativi consecutivi falliti attraverso un oggetto
  num_max_of_consecutive_failed_file_deletion_attemps: 0
};
async function startBatchPolling() {
  try {
    if (isPollingStarted) {
      return; // evito di avviare il polling più volte
    } else {
      isPollingStarted = true;
      console.log('(jobsManager.mjs) Inizio del batch polling per monitorare lo stato dei jobs...');
      
      while (true) {
        
        // reset delle tempi di attesa ai valori standard
        POOLINGINTERVAL = 2500;
        WAITBEFOREEXEC = 1000;
        
        console.log('\n(jobsManager.mjs) Sto controllando se ci sono jobs attivi...');
        console.log(`(jobsManager.mjs) Active jobs:`, await getActiveJobs());
        console.log(`(jobsManager.mjs) failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps:`, failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps);

        const jobIds = await getALLKeysJobs(); // utilizzo 'await' per attendere il risultato

        /*
        - Questi primi 3 controlli qui sotto (if - else if - else-if) servono per evitare di fare troppe richieste consecutive a SLURM nelle quali si cerca di cancellare
          eventuali files creati durante l'esecuzione dei jobs che sono già stati completati. In questo modo, si evita di continuare ad eseguire la funzione
          "deleteALLInferenceFiles(failedAttempts)" quando sono già state eseguite un numero pari a 'NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS'
          di richieste di cancellazione consecutive che non sono andate a buon fine poichè non c'erano files da cancellare.
          In particolare, solamente quando la condizione dell'if sarà verificata allora si invocherà "deleteALLInferenceFiles(failedAttempts)" altrimenti questa non verrà eseguita e questo continuerà
          fino a quando, continuando a decrementare il valore di 'failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps', questo non sarà tornato a 0.
        */
        if ((jobIds.length === 0) && (((!max_limit) && (failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps < NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS)) || ((max_limit) && (failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps === 0))) ) {
          max_limit = false
          console.log("(jobsManager.mjs) Nessun job attivo al momento. (Ho resettato a false il valore di 'max_limit').");
          console.log('(jobsManager.mjs) Inizio la verifica per la cancellazione di tutti i files creati durante le inferenze.');
          try {
            console.log('(jobsManager.mjs) Controllo se ci sono files su SLURM da cancellare.\n');
            await deleteALLInferenceFiles(failedAttempts); // attendo che la cancellazione di eventuali files su SLURM di jobs già completati venga terminata
            console.log('(jobsManager.mjs) Controllo per la cancellazione dei files completata con successo.\n');
          } catch (error) {
            console.log(`(jobsManager.mjs) Non ci sono files da cancellare in remoto: ${error.message}`);
          }
        } else if((jobIds.length === 0) && (!max_limit) && (failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps >= NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS)){
          max_limit = true  
          console.log('(jobsManager.mjs) Ho raggiunto il numero massimo di tentativi consecutivi di cancellazione dei files su SLURM che sono falliti.\n');

        } else if((jobIds.length === 0) && (max_limit)){
          failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps--;
          console.log('(jobsManager.mjs) Ho decrementato failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps di 1.\n');

        } else {
          // - Non appena ci sarà anche 1 solo job nella struttura dati, entrerò in questo else e 
          //   andrò a scorrere i primi 'NUM_MAX_AVAILABLE_MACHINES' jobs presenti in 'jobIds' per controllare quali sono stati completati.
          // - Non serve scorrere tutti i jobs perchè le macchine gh disponibili sono al massimo 'NUM_MAX_AVAILABLE_MACHINES':
          for (const [index, jobId] of jobIds.slice(0, NUM_MAX_AVAILABLE_MACHINES).entries()){
            
            WAITBETWEENJOBS = 2500; // reset tempo di attesa tra un job e un altro
            const job = await getJob(jobId);
            
            if (job.status !== "completed") { // entro solo se il job corrente non è ancora stato completato
              
              const numControlJob = await getNumControlJob(jobId);
              console.log(`(jobsManager.mjs) Valore 'NumControlJob' del job ${jobId}: ${numControlJob}`);

              const numTotalControlJob = await getNumTotalControlJob(jobId);
              console.log(`(jobsManager.mjs) Valore 'NumTotalControlJob' del job ${jobId}: ${numTotalControlJob}`);
              
              const selectedModelJob = await getSelectedModelJob(jobId);
              console.log(`(jobsManager.mjs) Valore 'selectedModel' del job ${jobId}: ${selectedModelJob}`);

              limit_numControlJob_currentJob = selectedModelJob === '8B' ? LIMIT_NUM_CONTROL_JOB_MODEL_8B : LIMIT_NUM_CONTROL_JOB_MODEL_70B
              console.log(`(jobsManager.mjs) limit_numControlJob_currentJob del job ${jobId}: ${limit_numControlJob_currentJob}`);
              
              limit_numMaxTotalControlJob_currentJob = selectedModelJob === '8B' ? LIMIT_MAX_NUM_CONTROL_JOB_MODEL_8B : LIMIT_MAX_NUM_CONTROL_JOB_MODEL_70B
              console.log(`(jobsManager.mjs) limit_numMaxTotalControlJob_currentJob del job ${jobId}: ${limit_numMaxTotalControlJob_currentJob}`);
              
              if(numTotalControlJob > limit_numMaxTotalControlJob_currentJob){
                /*
                Se entro qui vuol dire che il job ha raggiunto il limite massimo di tempo in coda (più precisamente il tempo passato tra i primi 4 job della struttura dati).
                - Questo controllo garantisce il fatto che un job non potrà continuare a rimanere in coda su SLURM all'infinito.
                - I passi che verranno eseguiti saranno i seguenti:
                  1) Eseguo "await cancelJobInSLURM(jobId)" per richiedere a SLURM di cancellare il job.
                    1.1) Se 1) va a buon fine allora eseguo "await updateJobStatus(jobId, 'cancelled-timeout');"
                    1.2) Se 1) fallisce allora stampo in console il fatto che si è verificato un errore durante la cancellazione e automaticamente
                         proverò a cancellare il job nel ciclo di pooling successivo.
                */
                const cancellation_result = await cancelJobInSLURM(jobId);
                if(cancellation_result === `Job cancellato con successo (tempo max raggiunto in coda)`){
                  console.log(`(jobsManager.mjs) Il job ${jobId} è stato cancellato con successo su SLURM (tempo max raggiunto in coda).\n`);
                  await updateJobStatus(jobId, 'cancelled-timeout'); // aggiorno lo stato del job in modo tale che la 'checkJobCompletion' presente in "slurm/executeJob.ts" veda la cancellazione e notifichi l'utente
                  console.log(`(jobsManager.mjs) Il job ${jobId} è stato cancellato per aver raggiunto il tempo max in coda e quindi ho impostato il suo stato a 'cancelled-timeout'.`);
                }else if(cancellation_result === `error`){
                  console.log(`(jobsManager.mjs) Si è verificato un errore (error) durante la cancellazione del job ${jobId} su SLURM (tempo max raggiunto in coda).\n`);
                }else{
                  console.log(`(jobsManager.mjs) E' stato ricevuto un messaggio di errore o di avviso (stderr) dopo la cancellazione del job ${jobId} su SLURM (tempo max raggiunto in coda).\n`);
                }
              }else if(numControlJob < limit_numControlJob_currentJob){
                // Se entro qui vuol dire che job ha ancora tempo a disposizione per essere completato.
                console.log(`\n(jobsManager.mjs) Verifico che il job ${jobId} sia stato completato contattando SLURM..`);
                try {
                  const result = await getResultAfterWait(jobId, job.userId, WAITBEFOREEXEC, updateJobStatus);
                  console.log(`(jobsManager.mjs) QuerySQL ottenuta [job ${jobId}]: ${result}\n`);
                } catch (error) {
                  console.log(`(jobsManager.mjs) Non è ancora stato generato il file di output per il job ${jobId}: \n${error}\n`);
                  await updateIncrementJobNumControl(jobId); // incremento di 1 il valore del campo 'numControl' del job corrente perchè con il controllo 
                  // fatto con la funzione 'getResultAfterWait(..)' ho capito che il suo file di output non è stato ancora generato
                  await updateIncrementJobNumTotalControl(jobId); // incremento anche il numero totale di controlli fatti per il job corrente

                  if (error && error.toString().includes('Connection reset')){ // se ottengo la chiusura della connessione da SLURM
                    if (index < 3) {
                      // Se non sto leggendo il quarto job nel for, aumento WAITBETWEENJOBS
                      WAITBETWEENJOBS = 4000;  // porto WAITBETWEENJOBS a 4s
                      console.log(`(jobsManager.mjs) Aumentato WAITBETWEENJOBS a: ${WAITBETWEENJOBS}`);
                    } else {
                      // Altrimenti, poichè sto leggendo il quarto job nel for, aumento POOLINGINTERVAL
                      POOLINGINTERVAL = 4000; // porto POOLINGINTERVAL a 4s
                      console.log(`(jobsManager.mjs) Aumentato POOLINGINTERVAL a: ${POOLINGINTERVAL}`);
                    }
                  }
                }
              }else {
                // Se entro qui vuol dire che il campo 'numControlJob' del job corrente è uguale a 'limit_numControlJob_currentJob' 
                // MA NON HA RAGGIUNTO ANCORA IL NUMERO MAX DI TEMPO DI ATTESA A DISPOSIZIONE e quindi devo controllare se questo job esiste
                // ancora su SLURM altrimenti vuol dire che è stato cancellato da quest'ultimo forse a causa di troppe richieste da parte degli utenti.
                const exist = await checkExistJob(jobId);
                if(exist === 'CANCELLED-BY-SLURM'){
                  // aggiorno lo stato del job in modo tale che la POST presente in "executeCommand/route.ts" sappia dell'annullamento e notifichi l'utente
                  await updateJobStatus(jobId, 'cancelled-by-slurm');
                  console.log(`(jobsManager.mjs) Il job ${jobId} è stato cancellato da SLURM prematuramente a causa del sovraccaricamento degli utenti e quindi ho impostato il suo stato a 'cancelled-by-slurm'.`);
                }else{
                  await updateResetJobNumControl(jobId); // resetto a 0 il contatore 'numControlJob' del job corrente
                  console.log(`(jobsManager.mjs) Il job ${jobId} è ancora in coda su SLURM e quindi ho resettato il suo NumControl.\n`);
                }
              }
            }
            // Attendo 'WAITBETWEENJOBS' secondi tra un job e l'altro (potrebbe anche NON esserci nessun job successivo)
            console.log(`(jobsManager.mjs) Attendo: ${WAITBETWEENJOBS/1000} secondi prima di passare al job successivo..\n`);
            await new Promise((resolve) => setTimeout(resolve, WAITBETWEENJOBS));
          }

          // Prima di ripetere il ciclo, cancello dalla struttura dati tutti i jobs che sono già stati completati oppure che sono stati cancellati ('cancelled-timeout' o 'cancelled-by-slurm')
          await deleteALLCompletedORCancelledJobs();
          console.log("\n(jobsManager.mjs) Ho terminato la pulizia della struttura dati, adesso riprendo il ciclo di controllo dei jobs.");
        }

        // Attendo 'POOLINGINTERVAL' secondi prima del prossimo ciclo di polling
        console.log(`(jobsManager.mjs) Attendo: ${POOLINGINTERVAL/1000} secondi prima di passare al prossimo ciclo di pooling..`);
        await new Promise((resolve) => setTimeout(resolve, POOLINGINTERVAL));
      }
    }
  } catch (error) {
    console.error('(jobsManager.mjs) ERRORE: Si è verificato un errore dentro la funzione -startBatchPolling-...', error);
  }
}


// Esporta le seguenti funzioni
export {
  startBatchPolling,
};
