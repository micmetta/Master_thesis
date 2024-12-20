import { startBatchPolling } from './jobsManager.mjs';

function register() {
  console.log('(instrumentation.mjs - register) - Starting Batch Polling..');
  startBatchPolling();
  console.log('(instrumentation.mjs - register) - Batch polling started successfully.\n\n');
}

export default {
  register
};
