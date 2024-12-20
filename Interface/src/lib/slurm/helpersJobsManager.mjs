import { exec } from 'child_process';

//////////////////////////// FUNCTIONS FOR HANDLING REQUESTS TO SLURM ////////////////////////////

async function checkIfJobExistsInSLURM(jobId) {
    return new Promise((resolve, reject) => {
      exec(`ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "squeue -u ${process.env.NEXT_PUBLIC_SSH_USER} -j ${jobId}"`, (error, stdout, stderr) => {
        if (error) {
            console.error(`Error during job existence check on SLURM: ${error.message}`);
            reject(`Error during job existence check on SLURM: ${error.message}`);
          } else if (stderr) {
            console.error(`Stderr when checking the status of the job: ${stderr}`);
            reject(`Error (stderr): ${stderr}`);
          } else {
            console.log(`squeue stdout on SLURM: ${stdout}`);
            resolve(stdout.includes(jobId)); // promise completed successfully and return true if the job is still present on SLURM, otherwise return false
          }
      });
    })
}
  
async function checkExistJob(jobId) {
    try {
      const jobExists = await checkIfJobExistsInSLURM(jobId);
      if (!jobExists) {
        console.log(`(helpersJobsManager.mjs - checkJobCompletion) Job ${jobId} no longer exists on SLURM. Job deleted.`);
        return 'CANCELLED-BY-SLURM'; // job no longer exists on SLURM (cancelled by SLURM)
      }
      return 'RUNNING'; // job is still running
    } catch (error) {
      console.error(`Error when checking the existence of job ${jobId} on SLURM: ${error}`);
      return 'CANCELLED-BY-SLURM'; // in the event of a verification error, the job is considered to have been deleted from SLURM
    }
}
  
// Function to delete a job from SLURM after the maximum time has been reached
async function cancelJobInSLURM(jobId) {
    return new Promise((resolve, reject) => {
      exec(`ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "scancel ${jobId}"`, (error, stdout, stderr) => {
        if (error) {
          console.error(`(error) Error when deleting job on SLURM (max. time reached in queue): ${error.message}`); // serious error obtained during exec execution (exec execution is aborted)
          reject(`error`);
        } else if (stderr) {
          console.error(`(stderr) during job cancellation on SLURM (max. time reached in queue): ${stderr}`); // error or warning message obtained during the execution of the exec (does not necessarily interrupt the execution of the exec) 
          reject(`stderr`);
        } else {
          console.log(`Job with ID ${jobId} successfully deleted.`);
          resolve(`Job successfully cancelled (max. time reached in queue)`);
        }
      });
    });
}
  
async function deleteALLInferenceFiles(failedAttempts){
    // SSH command to delete all files starting with 'output_querySQL' or 'slurm' on SLURM:
    const command = `ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "rm ${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES}/output_querySQL* ${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES}/slurm*"`;
    return new Promise((resolve, reject) => { // makes the exec call compatible with await
      exec(command, (err) => { // I execute the SSH command to delete the files corresponding to the patterns
        if (err) {
          failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps++;
          reject(err)
        } else {
          console.log(`(helpersJobsManager.mjs - deleteALLInferenceFiles) All files starting with 'output_querySQL' or 'slurm' were deleted.`);
          resolve();
        }
      });
    });
}
  
function getResultAfterWait(jobId, userId, waitBeforeExec, updateJobStatus) {
    return new Promise((resolve, reject) => {
      try {
        // I wait 'waitBeforeExec' seconds before running the SSH command to read the output file
        setTimeout(() => {
          // I execute a single exec to attempt to read the output file
          exec(`ssh ${process.env.NEXT_PUBLIC_SSH_USER}@${process.env.NEXT_PUBLIC_SSH_SERVER} "cat ${process.env.NEXT_PUBLIC_PATH_FILE_INFERENCES}/output_querySQL_${userId}.txt"`, async (err, result) => {
            if (err) {
              // if a warning occurs (e.g. the file does not exist) I print it out
              console.log(`(helpersJobsManager.mjs - getResultAfterWait) Warning in retrieving the job result ${jobId}: ${err.message}`);
              await updateJobStatus(jobId, 'running'); // I update the job status as ‘running
              reject(`Warning in retrieving the job result ${jobId}: ${err.message}`); // I return the warning
            } else {
              // if the output was found, I update the job status as completed with the result
              console.log(`(helpersJobsManager.mjs - getResultAfterWait) Job result ${jobId}: ${result}`);
              await updateJobStatus(jobId, 'completed', result);
              resolve(result); // I solve the promise by returning the result
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