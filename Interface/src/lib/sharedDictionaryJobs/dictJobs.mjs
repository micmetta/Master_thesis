import { Mutex } from 'async-mutex';

// istanzio il mutex
const mutex = new Mutex();

// struttura dati (dizionario) per memorizzare e gestire i jobs attivi:
global.activeJobs = global.activeJobs || {}; // global serve per la condivisione dello stesso dizionario tra le diverse funzioni di gestione presenti sotto


///////////////////////////////////////////////////// FUNZIONI PER LA GESTIONE DI global.activeJobs /////////////////////////////////////////////////////
// Funzione per aggiungere un job attivo con mutex
async function addJob(jobId, userId, selectedModel) {
  const release = await mutex.acquire();
  try {
    global.activeJobs[jobId] = {
      status: 'running', // valori possibili durante l'esecuzione: 'running', 'completed', 'failed' o 'cancelled'
      userId: userId,
      result: null, // querySQL generata dal modello
      numControl: 0, // numero di controlli fatti per vedere se il job è ancora in esecuzione o meno (che viene resettato a 0 ogni tot in base alla scelta del modello)
      numTotalControl: 0, // numero di controlli totali fatti per vedere se il job è ancora in esecuzione o meno (NON viene resettato a 0) 
                          // e una volta che raggiunge un certo valore (in base al modello scelto) il job verrà direttamente cancellato da SLURM.
                          // Questo campo mi permette quindi di cancellare il job dopo che è rimasto in coda per diverso tempo in modo da evitare che rimanga nella struttura e porti a dei rallentamenti inutili nella gestione del backend.
      selectedModel: selectedModel, // modello richiesto dall'utente per questo job
      canRemoved: false // sarà true solamente quando il job può essere rimosso in sicurezza dalla struttura
    };
  } finally {
    release();
  }
}

// Funzione per ottenere tutti i job attivi con mutex
async function getActiveJobs() {
  const release = await mutex.acquire(); // acquisisco il lock prima di leggere
  try {
    return global.activeJobs;
  } finally {
    release(); //rilascio il lock
  }
}

// Funzione per ottenere un job attivo con mutex
async function getJob(jobId) {
  const release = await mutex.acquire();
  try {
    return global.activeJobs[jobId] || null;
  } finally {
    release();
  }
}

// Funzione per ottenere tutte le chiavi dei job attivi con mutex
async function getALLKeysJobs() {
  const release = await mutex.acquire(); // Acquisisce il lock prima di leggere
  try {
    return Object.keys(global.activeJobs);
  } finally {
    release();
  }
}

// Funzione per ottenere il valore del campo 'numControl' del job con mutex
async function getNumControlJob(jobId) {
  const release = await mutex.acquire();
  try {
    return global.activeJobs[jobId].numControl;
  } finally {
    release();
  }
}

// Funzione per ottenere il valore del campo 'numTotalControl' del job con mutex
async function getNumTotalControlJob(jobId) {
  const release = await mutex.acquire();
  try {
    return global.activeJobs[jobId].numTotalControl;
  } finally {
    release();
  }
}

// Funzione per ottenere il valore del campo 'numControl' del job con mutex
async function getSelectedModelJob(jobId) {
  const release = await mutex.acquire();
  try {
    return global.activeJobs[jobId].selectedModel;
  } finally {
    release();
  }
}

// Funzione per aggiornare lo stato del job con mutex
async function updateJobStatus(jobId, status, result) {
  const release = await mutex.acquire();
  try {
    if (global.activeJobs[jobId]) {
      global.activeJobs[jobId].status = status;
      if (result) {
        global.activeJobs[jobId].result = result;
      }
    }
  } finally {
    release();
  }
}

async function updateJobCanRemoved(jobId) {
  const release = await mutex.acquire();
  try {
    if (global.activeJobs[jobId]) {
      global.activeJobs[jobId].canRemoved = true;
    }
  } finally {
    release();
  }
}

// Funzione per incrementare di 1 il valore di 'numControl' del job con mutex
async function updateIncrementJobNumControl(jobId) {
  const release = await mutex.acquire();
  try {
    if (global.activeJobs[jobId]) {
      global.activeJobs[jobId].numControl += 1;
    }
  } finally {
    release();
  }
}

// Funzione per resettare a 0 il valore di 'numControl' del job con mutex
async function updateResetJobNumControl(jobId) {
  const release = await mutex.acquire();
  try {
    if (global.activeJobs[jobId]) {
      global.activeJobs[jobId].numControl = 0;
    }
  } finally {
    release();
  }
}

// Funzione per incrementare di 1 il valore di 'numTotalControl' del job con mutex
async function updateIncrementJobNumTotalControl(jobId) {
  const release = await mutex.acquire();
  try {
    if (global.activeJobs[jobId]) {
      global.activeJobs[jobId].numTotalControl += 1;
    }
  } finally {
    release();
  }
}

// Funzione per rimuovere un job completato o fallito con mutex
async function removeJob(jobId) {
  const release = await mutex.acquire();
  try {
    delete global.activeJobs[jobId];
  } finally {
    release();
  }
}

async function deleteALLCompletedORCancelledJobs(){
  const jobIds = await getALLKeysJobs();
  for (const jobId of jobIds){
    const current_job = await getJob(jobId); 
    if(current_job.canRemoved){
      await removeJob(jobId); // rimuovo il job completato dalla struttura dei jobs
      console.log(`(jobManager.js)-(deleteALLCompletedJobs) Ho eliminato il job con id: ${jobId}\n`);
    }
  }
}


// Esporta le seguenti funzioni
export {
    addJob,
    getActiveJobs,
    getJob,
    getALLKeysJobs,
    getNumControlJob,
    getNumTotalControlJob,
    getSelectedModelJob,
    updateJobStatus,
    updateJobCanRemoved,
    updateIncrementJobNumControl,
    updateResetJobNumControl,
    updateIncrementJobNumTotalControl,
    deleteALLCompletedORCancelledJobs
  };