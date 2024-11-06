import { exec } from 'child_process';

///////////////////////////////////////////////////// FUNZIONI PER GESTIONE RICHIESTE A SLURM /////////////////////////////////////////////////////

async function checkIfJobExistsInSLURM(jobId) {
    return new Promise((resolve, reject) => {
      //es. exec(`ssh mmetta@c3sfr1.di.unito.it "squeue -u mmetta -j ${jobId}"`
      exec(`ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "squeue -u ${process.env.NEXT_PUBLIC_SSH_USER} -j ${jobId}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Errore durante il controllo di esistenza del job su SLURM: ${error.message}`);
            reject(`Errore durante il controllo di esistenza del job su SLURM: ${error.message}`);
          } else if (stderr) {
            console.error(`Stderr durante il controllo dello stato del job: ${stderr}`);
            reject(`Errore (stderr): ${stderr}`);
          } else {
            console.log(`stdout di squeue su SLURM: ${stdout}`);
            // promise completata con successo e restituisco true se il job è ancora presente su SLURM, altrimenti restituisco false
            resolve(stdout.includes(jobId));
          }
      });
    })
}
  
async function checkExistJob(jobId) {
    try {
      const jobExists = await checkIfJobExistsInSLURM(jobId);
      if (!jobExists) {
        console.log(`(jobManager.js)-(checkJobCompletion) Il job ${jobId} non esiste più su SLURM. Job cancellato.`);
        return 'CANCELLED-BY-SLURM'; // job non esiste più su SLURM (annullato da SLURM)
      }
      return 'RUNNING'; // job è ancora in esecuzione
    } catch (error) {
      console.error(`Errore durante la verifica dell'esistenza del job ${jobId} su SLURM: ${error}`);
      return 'CANCELLED-BY-SLURM'; // in caso di errore nella verifica, si considera il job come se fosse stato cancellato da SLURM
    }
}
  
// Funzione per cancellare un job da SLURM dopo aver raggiunto il tempo massimo a disposizione
async function cancelJobInSLURM(jobId) {
    return new Promise((resolve, reject) => {
      //es. exec(`ssh mmetta@c3sfr1.di.unito.it "scancel ${jobId}"`
      exec(`ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "scancel ${jobId}"`, (error, stdout, stderr) => {
        if (error) {
          console.error(`(error) Errore durante la cancellazione del job su SLURM (tempo max raggiunto in coda): ${error.message}`); // errore grave ottenuto durante l'esecuzione dell'exec (l'esecuzione dell'exec viene interrotta)
          reject(`error`);
        } else if (stderr) {
          console.error(`(stderr) durante la cancellazione del job su SLURM (tempo max raggiunto in coda): ${stderr}`); // errore o messaggio di avviso ottenuto durante l'esecuzione dell'exec (non interrompe per forza l'esecuzione dell'exec) 
          reject(`stderr`);
        } else {
          console.log(`Job con ID ${jobId} cancellato con successo.`);
          resolve(`Job cancellato con successo (tempo max raggiunto in coda)`);
        }
      });
    });
}
  
async function deleteALLInferenceFiles(failedAttempts){
    // Comando SSH per cancellare tutti i file che iniziano con 'output_querySQL' o 'slurm' su SLURM:
    // es. `ssh mmetta@c3sfr1.di.unito.it "rm /beegfs/home/mmetta/Test/TestTesi/output_querySQL* /beegfs/home/mmetta/Test/TestTesi/slurm*"`;
    const command = `ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "rm ${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES}/output_querySQL* ${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES}/slurm*"`;
    return new Promise((resolve, reject) => { // rende la chiamata di exec compatibile con await
      exec(command, (err) => { // eseguo il comando SSH per cancellare i files corrispondenti ai pattern
        if (err) {
          failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps++;
          reject(err)
        } else {
          console.log(`(jobManager.js - deleteALLInferenceFiles) Tutti i file che iniziano con 'output_querySQL' o 'slurm' sono stati cancellati.`);
          resolve();
        }
      });
    });
}
  
function getResultAfterWait(jobId, userId, waitBeforeExec, updateJobStatus) {
    return new Promise((resolve, reject) => {
      try {
        // Attendo 'waitBeforeExec' secondi prima di eseguire il comando SSH per leggere il file di output
        setTimeout(() => {
          // Eseguo un'unica exec per tentare di leggere il file di output
          //es. exec(`ssh mmetta@c3sfr1.di.unito.it "cat /beegfs/home/mmetta/Test/TestTesi/output_querySQL_${userId}.txt"`
          exec(`ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "cat ${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES}/output_querySQL_${userId}.txt"`, async (err, result) => {
            if (err) {
              // se si verifica un errore (ad esempio, il file non esiste) lo stampo
              console.error(`(jobManager.js - getResultAfterWait) Errore nel recupero del risultato del job ${jobId}: ${err.message}`);
              await updateJobStatus(jobId, 'running'); // aggiorno lo stato del job come "running"
              reject(`Errore nel recupero del risultato del job ${jobId}: ${err.message}`); // restituisco l'errore
            } else {
              // se l'output è stato trovato, aggiorno lo stato del job come completato con il risultato
              console.log(`(jobManager.js - getResultAfterWait) Risultato del job ${jobId}:\n${result}`);
              await updateJobStatus(jobId, 'completed', result);
              resolve(result); // risolvo la promise restituendo il risultato
            }
          });
        }, waitBeforeExec);
      } catch (retryError) {
        reject(retryError);
      }
    });
}


export {
  checkExistJob, cancelJobInSLURM, deleteALLInferenceFiles, getResultAfterWait
};