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


const NUM_MAX_AVAILABLE_MACHINES = 4 // maximum number of gracehopper available
const NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS = 3 // maximum number of consecutive attempts to delete files on SLURM (created by completed jobs) that failed
const LIMIT_NUM_CONTROL_JOB_MODEL_8B = 16 // (8B: 12 only for loading the model (2.5s * 12 = 30s) + 4 for more complex queries (2.5s * 4 = 10s) = 16 (max. 40s before check with 'checkExistJob(jobId)' - if it is the only job running, otherwise the time before check will increase))
const LIMIT_NUM_CONTROL_JOB_MODEL_70B = 42 // (70B: 30 only for loading the model (2.5s * 30 = 75s) + 12 for more complex queries (2.5s * 12 = 30s) = 42 (max. 105s before check with 'checkExistJob(jobId)' - if it is the only job running, otherwise the time before check will increase))
const LIMIT_MAX_NUM_CONTROL_JOB_MODEL_8B = 40 // max. number of attempts to read the output file of a job after which I will delete it from SLURM and notify the user (for the 8B model) (this is about 100s of waiting time - if it is the only job running, otherwise the time before deletion will increase)
const LIMIT_MAX_NUM_CONTROL_JOB_MODEL_70B = 100 // max. number of attempts to read the output file of a job after which I will delete it from SLURM and notify the user (for the 70B model) (this is about 250s waiting time - if it is the only job running, otherwise the time before deletion will increase)
let limit_numControlJob_currentJob = 0 // initial value is 0 so it will be changed dynamically in the for that loops on the jobs
let limit_numMaxTotalControlJob_currentJob = 0; // initial value is 0 so it will be changed dynamically in the for that loops on the jobs
let max_limit = false // will be set to true when 'failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps' has reached at least the value of 'NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS' (this boolean is useful in optimising requests to SLURM) 
let isPollingStarted = false; // serves to avoid running the 'startBatchPolling()' function launched by the 'instrumentation.js' script more than once
let POOLINGINTERVAL = 2500;
let WAITBEFOREEXEC = 1000;
let WAITBETWEENJOBS = 2500;
let failedAttempts = { // allows me to pass by reference the maximum number of consecutive failed attempts through an object
  num_max_of_consecutive_failed_file_deletion_attemps: 0
};
async function startBatchPolling() {
  try {
    if (isPollingStarted) {
      return; // I avoid starting polling several times
    } else {
      isPollingStarted = true;
      console.log('(jobsManager.mjs) Start of batch polling to monitor the status of jobs...');
      
      while (true) {
        
        // resetting of waiting times to standard values
        POOLINGINTERVAL = 2500;
        WAITBEFOREEXEC = 1000;
        
        console.log('\n(jobsManager.mjs) I am checking if there are any active jobs...');
        console.log(`(jobsManager.mjs) Active jobs:`, await getActiveJobs());
        console.log(`(jobsManager.mjs) failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps:`, failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps);

        const jobIds = await getALLKeysJobs(); // waiting for all job IDs

        /*
        - These first 3 checks below (if - else - if) serve to avoid making too many consecutive requests to SLURM in which it tries to delete
          any files created during the execution of jobs that have already been completed. In this way, we avoid continuing to execute the function
          ‘deleteALLInferenceFiles(failedAttempts)' function when a number of “NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS” have already been executed
          of consecutive deletion requests that failed because there were no files to be deleted.
          Specifically, only when the if condition is met will 'deleteALLInferenceFiles(failedAttempts)' be invoked otherwise it will not be executed and this will continue
          until, by continuing to decrement the value of 'failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps', it has returned to 0.
        */
        if ((jobIds.length === 0) && (((!max_limit) && (failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps < NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS)) || ((max_limit) && (failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps === 0))) ) {
          max_limit = false
          console.log("(jobsManager.mjs) No active job at the moment. (I reset the value of 'max_limit' to false).");
          console.log('(jobsManager.mjs) I start checking for the deletion of all files created during inference.');
          try {
            console.log('(jobsManager.mjs) Check if there are files on SLURM to be deleted.\n');
            await deleteALLInferenceFiles(failedAttempts); // waiting for the deletion of any files on SLURM of already completed jobs to be completed
            console.log('(jobsManager.mjs) Check for successful file deletion.\n');
          } catch (error) {
            console.log(`(jobsManager.mjs) There are no files to be deleted remotely: ${error.message}`);
          }
        } else if((jobIds.length === 0) && (!max_limit) && (failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps >= NUM_MAX_OF_CONSECUTIVE_FILE_DELETION_ATTEMPS)){
          max_limit = true  
          console.log('(jobsManager.mjs) I have reached the maximum number of consecutive attempts to delete files on SLURM that have failed.\n');

        } else if((jobIds.length === 0) && (max_limit)){
          failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps--;
          console.log('(jobsManager.mjs) I decreased failedAttempts.num_max_of_consecutive_failed_file_deletion_attemps by 1.\n');

        } else {
          // - As soon as there is even 1 job in the data structure, I will enter this else and I will scroll through 
          //   the first 'NUM_MAX_AVAILABLE_MACHINES' jobs in 'jobIds' to check which ones have been completed.
          // - There is no need to scroll through all the jobs because the available gh machines are at most 'NUM_MAX_AVAILABLE_MACHINES':
          for (const [index, jobId] of jobIds.slice(0, NUM_MAX_AVAILABLE_MACHINES).entries()){
            
            WAITBETWEENJOBS = 2500; // reset waiting time between jobs
            const job = await getJob(jobId);
            
            if (job.status !== "completed") { // within only if the current job has not yet been completed
              
              const numControlJob = await getNumControlJob(jobId);
              console.log(`(jobsManager.mjs) 'NumControlJob' value of the job ${jobId}: ${numControlJob}`);

              const numTotalControlJob = await getNumTotalControlJob(jobId);
              console.log(`(jobsManager.mjs) 'NumTotalControlJob' value of the job ${jobId}: ${numTotalControlJob}`);
              
              const selectedModelJob = await getSelectedModelJob(jobId);
              console.log(`(jobsManager.mjs) Job 'selectedModel' value ${jobId}: ${selectedModelJob}`);

              limit_numControlJob_currentJob = selectedModelJob === '8B' ? LIMIT_NUM_CONTROL_JOB_MODEL_8B : LIMIT_NUM_CONTROL_JOB_MODEL_70B
              console.log(`(jobsManager.mjs) job limit_numControlJob_currentJob ${jobId}: ${limit_numControlJob_currentJob}`);
              
              limit_numMaxTotalControlJob_currentJob = selectedModelJob === '8B' ? LIMIT_MAX_NUM_CONTROL_JOB_MODEL_8B : LIMIT_MAX_NUM_CONTROL_JOB_MODEL_70B
              console.log(`(jobsManager.mjs) job limit_numMaxTotalControlJob_currentJob ${jobId}: ${limit_numMaxTotalControlJob_currentJob}`);
              
              if(numTotalControlJob > limit_numMaxTotalControlJob_currentJob){
                /*
                If I enter here, it means that the job has reached the maximum time limit in the queue (more precisely the time spent between the first 4 jobs in the data structure).
                - This check ensures that a job cannot continue to be queued on SLURM indefinitely.
                - The steps that will be performed are as follows:
                  1) I execute 'await cancelJobInSLURM(jobId)' to request SLURM to cancel the job.
                    1.1) If 1) succeeds then I execute 'await updateJobStatus(jobId, “cancelled-timeout”);'
                    1.2) If 1) fails then I print the fact that an error occurred during the cancellation to the console and automatically try to delete the job in the next pooling cycle.
                */
                const cancellation_result = await cancelJobInSLURM(jobId);
                if(cancellation_result === `Job successfully cancelled (max. time reached in queue)`){
                  console.log(`(jobsManager.mjs) The job ${jobId} was successfully deleted on SLURM (max. time reached in the queue).\n`);
                  await updateJobStatus(jobId, 'cancelled-timeout'); // I update the job status so that the 'checkJobCompletion' in "slurm/executeJob.ts" sees the cancellation and notifies the user
                  console.log(`(jobsManager.mjs) The job ${jobId} was cancelled due to having reached the maximum time in the queue and therefore I set its status to 'cancelled-timeout'.`);
                }else if(cancellation_result === `error`){
                  console.log(`(jobsManager.mjs) An error occurred while deleting job ${jobId} on SLURM (max. time reached in the queue).\n`);
                }else{
                  console.log(`(jobsManager.mjs) An error or warning message (stderr) was received after deleting the job ${jobId} on SLURM (max. time reached in the queue).\n`);
                }
              }else if(numControlJob < limit_numControlJob_currentJob){
                // If I enter here, it means that job still has time to be completed.
                console.log(`\n(jobsManager.mjs) I check that the job ${jobId} has been completed by contacting SLURM..`);
                try {
                  const result = await getResultAfterWait(jobId, job.userId, WAITBEFOREEXEC, updateJobStatus);
                  console.log(`(jobsManager.mjs) QuerySQL obtained [job ${jobId}]: ${result}\n`);
                } catch (error) {
                  console.log(`(jobsManager.mjs) The output file for job ${jobId} has not yet been generated: \n${error}\n`);
                  await updateIncrementJobNumControl(jobId); // increase the value of the 'numControl' field of the current job by 1 because the check 
                  // done with the function 'getResultAfterWait(..)' I realised that its output file has not yet been generated
                  await updateIncrementJobNumTotalControl(jobId); // also increase the total number of checks made for the current job

                  if (error && error.toString().includes('Connection reset')){ // if I get the connection closed by SLURM
                    if (index < 3) {
                      // If I am not reading the fourth job in the for, WAITBETWEENJOBS increase
                      WAITBETWEENJOBS = 4000;
                      console.log(`(jobsManager.mjs) Increased WAITBETWEENJOBS to: ${WAITBETWEENJOBS}`);
                    } else {
                      // Otherwise, as I am reading the fourth job in the for, I increase POOLINGINTERVAL
                      POOLINGINTERVAL = 4000;
                      console.log(`(jobsManager.mjs) Increased POOLINGINTERVAL to: ${POOLINGINTERVAL}`);
                    }
                  }
                }
              }else {
                // If I enter here, it means that the 'numControlJob' field of the current job is equal to 'limit_numControlJob_currentJob' 
                // BUT HAS NOT YET REACHED THE MAXIMUM NUMBER OF WAITING TIME AVAILABLE so I have to check if this job exists
                // still on SLURM otherwise it means that it has been deleted from SLURM perhaps due to too many user requests.
                const exist = await checkExistJob(jobId);
                if(exist === 'CANCELLED-BY-SLURM'){
                  await updateJobStatus(jobId, 'cancelled-by-slurm'); // I update the job status so that the 'checkJobCompletion' in "slurm/executeJob.ts" sees the cancellation and notifies the user
                  console.log(`(jobsManager.mjs) The job ${jobId} was deleted from SLURM prematurely due to user overload and therefore I set its status to 'cancelled-by-slurm'.`);
                }else{
                  await updateResetJobNumControl(jobId); // reset the 'numControlJob' counter of the current job to 0
                  console.log(`(jobsManager.mjs) The job ${jobId} is still queued on SLURM and so I reset its NumControl.\n`);
                }
              }
            }
            // I wait for 'WAITBETWEENJOBS' seconds between jobs (there may NOT even be a next job)
            console.log(`(jobsManager.mjs) I wait: ${WAITBETWEENJOBS/1000} seconds before moving on to the next job..\n`);
            await new Promise((resolve) => setTimeout(resolve, WAITBETWEENJOBS));
          }

          // Before repeating the cycle, I delete from the data structure all jobs that have already been completed or that have been deleted ('cancelled-timeout' o 'cancelled-by-slurm')
          await deleteALLCompletedORCancelledJobs();
          console.log("\n(jobsManager.mjs) I have finished cleaning the data structure, now I will resume the job control cycle.");
        }

        // I wait for 'POOLINGINTERVAL' seconds before the next polling cycle
        console.log(`(jobsManager.mjs) I wait: ${POOLINGINTERVAL/1000} seconds before moving on to the next pooling cycle..`);
        await new Promise((resolve) => setTimeout(resolve, POOLINGINTERVAL));
      }
    }
  } catch (error) {
    console.error('(jobsManager.mjs) ERROR: An error occurred within the function -startBatchPolling-...', error);
  }
}


// Export the following functions
export {
  startBatchPolling,
};
