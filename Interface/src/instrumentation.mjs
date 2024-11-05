import { startBatchPolling } from './jobsManager.mjs';

function register() {
  console.log('Avvio del batch polling all\'avvio del backend');
  startBatchPolling();
  console.log('Batch polling partito con successo.\n\n');
}

export default {
  register
};
