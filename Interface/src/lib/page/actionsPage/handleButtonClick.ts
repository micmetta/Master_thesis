import { handleButtonClickServer } from '@/lib/handleInference/handleButtonClickServer';
import { generateUniqueSessionId } from '@/lib/page/utilsPage/helpers';

// Definizione dei tipi per ciascun callback passato
type SetState<T> = (value: T) => void;
type SetError = (error: string | null) => void;
type SetLoading = (loading: boolean) => void;
type SetResponseTime = (time: number | null) => void;
type SetResults = (results: string[][]) => void;

export async function handleButtonClick(
    nlQuery: string,
    model: string,
    setLoading: SetLoading,
    setError: SetError,
    setSqlQuery: SetState<string>,
    setColumnNames: SetState<string[]>,
    setQueryResults: SetResults,
    RESULTS_PER_PAGE: number,
    setDisplayedResults: SetResults,
    setPage: SetState<number>,
    setResponseTime: SetResponseTime,
    setUnavailabilityNode: SetState<string | null>
) {
    if (nlQuery.trim() === '') {
        alert("ERRORE: La barra di testo non può essere vuota.");
        return;
    }

    setLoading(true);
    setError(null);
    setSqlQuery('');
    setColumnNames([]);
    setQueryResults([]);
    setDisplayedResults([]); // (React crea un riferimento all'array 'displayedResults' vuoto)
    setPage(1);
    setResponseTime(null);  // reset del timer
    setUnavailabilityNode(null);

    // - Con questa piccola parte di codice gestisco il caso in cui i tempi d'inferenza diventino maggiori di quelli medi (rispetto al modello scelto)
    //   in modo tale da mostrare comunque un feedback all'utente con un messaggio in blu invece che rosso:
    const TIMEOUT_THRESHOLD_8B = 40000; // 40s sarà la soglia minima di attesa prima di notificare l'utente sul rallentamento del sistema (se l'utente ha scelto il modello a 8B)
    const TIMEOUT_THRESHOLD_70B = 100000; // 100s sarà la soglia minima di attesa prima di notificare l'utente sul rallentamento del sistema (se l'utente ha scelto il modello a 70B)
    const timeoutUnvailableNode = model === '8B' ? TIMEOUT_THRESHOLD_8B : TIMEOUT_THRESHOLD_70B;
    // - Invoco la funzione "setTimeout()"" che si preoccuperà di eseguire la callback dopo un certo tempo pari 'TIMEOUT_THRESHOLD' per notificare all'utente la causa 
    //   di un'eventuale ritardo nell'esecuzione della sua richiesta:
    const timer = setTimeout(() => {
      setUnavailabilityNode("Il tempo per eseguire la richiesta è aumentato a causa della momentanea indisponibilità di un nodo di calcolo, continua ad attendere per non perdere la priorità acquisita.");
    }, timeoutUnvailableNode);
    
    // - Invio richiesta a SLURM e attendo i risultati
    try {
        const sessionId = generateUniqueSessionId();
        const startTime = performance.now();
        const result = await handleButtonClickServer(nlQuery, sessionId, model);
        const endTime = performance.now();
        setResponseTime((endTime - startTime) / 1000); // tempo in secondi
        
        if (!result.error) {
            setSqlQuery(result.output);
            setColumnNames(result.columns);
            setQueryResults(result.rows);
            setDisplayedResults(result.rows.slice(0, RESULTS_PER_PAGE));
        }else {
            if(result.output != ''){
            setSqlQuery(result.output);
            }
            setError(result.error);
        }
    } catch (error) {
        setError("Si è verificato un errore durante l'invio della richiesta al server, riprova. " + "(" + error + ")");
    } finally {
        setUnavailabilityNode(null); // pulizia del messaggio di indisponibilità nodo
        setLoading(false); // fine caricamento
        clearTimeout(timer); // interrompe il timeout per il messaggio di indisponibilità
    }
}
