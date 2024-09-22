"use client"; // per specificare che questo file sarà eseguito sul client-side nell'app next.js 

// - useState per gestire lo stato di ogni component
// - useEffect è uno Hook (funzione speciale di Javascript che permette di utilizzare le funzionalità di React)
//   per gestire il caricamento dati dopo il rendering
// - useRef per creare una variabile in grado di gestire il comportamento dello scrolling della tabella
import { Button } from '@mantine/core';
import { useState, useEffect, useRef } from 'react';
import "./table.css";

export default function Home() {
  // Dichiarazione degli stati (valore attuale dello stato, funzione per aggiornare lo stato e forzare il re-render del componente):
  const [nlQuery, setNlQuery] = useState<string>('');  // stato per memorizzare la query NL (inizializzato con stringa vuota)
  const [sqlQuery, setSqlQuery] = useState<string>(''); // stato per memorizzare la query SQL generata dal backend
  const [columnNames, setColumnNames] = useState<string[]>([]); // stato per i nomi delle colonne
  const [queryResults, setQueryResults] = useState<any[][]>([]); // stato per le tuple dei risultati (queryResults conterrà le tuple del resultest restituito dal backend)
  const [displayedResults, setDisplayedResults] = useState<any[][]>([]); // stato per i risultati mostrati (gruppi di 100)
  const [loading, setLoading] = useState<boolean>(false); // stato di caricamento
  const [error, setError] = useState<string | null>(null); // stato per l'errore
  const [rowCounter, setRowCounter] = useState<number[]>([]); // stato per il contatore delle righe
  const [page, setPage] = useState<number>(1); // stato per il numero di pagine caricate
  const [responseTime, setResponseTime] = useState<number | null>(null); // stato per calcolare il tempo di risposta
  const RESULTS_PER_PAGE = 100; // numero di risultati per pagina
  const tableContainerRef = useRef<HTMLDivElement>(null); // riferimento per il contenitore della tabella scrollabile 
  // utilizzato per caricamento di più risultati quando l'utente scorre


  // Funzione asincrona che viene eseguita quando l'utente invierà la query in NL:
  const handleSubmit = async () => {

    // controllo se la query in NL è vuota
    if (nlQuery.trim() === '') {
      alert("ERRORE: La barra di testo non può essere vuota.");
      return;
    }

    // setting degli stati
    setLoading(true);
    setError(null);
    setSqlQuery('');
    setColumnNames([]);
    setQueryResults([]);
    setDisplayedResults([]); //(React crea un riferimento all'array 'displayedResults' vuoto)
    setRowCounter([]);
    setPage(1);
    setResponseTime(null);  // reset del timer


    const startTime = performance.now(); // start timer
    try {
      ////////////////////////////////////////////////////////////////////////////////
      // richiesta inviata al backend (fetch) e attesa della risposta (await):
      const response = await fetch(`http://localhost:4000/queryNL?text=${nlQuery}`);
      ////////////////////////////////////////////////////////////////////////////////

      const endTime = performance.now(); // end timer
      const durationInSeconds  = (endTime - startTime) / 1000; // durata in secondi
      setResponseTime(durationInSeconds); // imposto il tempo di risposta
      
      ////////////////////////////////////////////////////////////////////////////////
      // Gestione errori:
      if (!response.ok) { // se la risposta del server non è un 'successo' allora mostro un messaggio di errore
        throw new Error('Errore nel recuperare la query SQL. Errore: ${response.status} ${response.statusText}');
      }
      const data = await response.json(); // estraggo la risposta in un oggetto json
      console.log("Risposta dal backend:", data);
      if (data.SQL_query_generated === 'Nessuna query SQL trovata.') {
        alert("Nessun risultato trovato nel database.");
      }
      if (data.response_execution_query === 'connection_error') {
        setError("Errore di connessione al database.");
        return;
      }
      if (data.response_execution_query === 'query_error') {
        setError("Errore durante l'esecuzione della query nel database.");
        return;
      }
      if (Array.isArray(data.response_execution_query) && data.response_execution_query.length === 0) {
        setError("Il resultset è vuoto. Nessun risultato trovato.");
      }
      if (data.response_execution_query === 'generic_error') {
        setError("Si è verificato un errore imprevisto.");
        return;
      }
      ////////////////////////////////////////////////////////////////////////////////

      //////////////////////////////////////////////////////////////////////////////////
      // Aggiorno gli stati con tutte le info ottenute nella risposta data dal backend:
      setSqlQuery(data.SQL_query_generated);
      setColumnNames(data.columns_names);
      setQueryResults(data.response_execution_query);
      setDisplayedResults(data.response_execution_query.slice(0, RESULTS_PER_PAGE));
      setRowCounter(Array.from({ length: RESULTS_PER_PAGE }, (_, i) => i + 1));
      //////////////////////////////////////////////////////////////////////////////////

    } catch (error) {
      console.error('Errore nella chiamata API REST:', error);
      setError('Errore nel recuperare la query SQL, riprova. ' + error);
    } finally {
      setLoading(false);
    }
  };

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Funzione invocata nel momento in cui l'utente scorre verso il fondo della tabella
  // per caricare più risultati
  const loadMoreResults = () => {
    const nextPage = page + 1; // calcolo pagina successiva (prima pagina = 1)
    const startIndex = (nextPage - 1) * RESULTS_PER_PAGE; // aggiorno l'indice di partenza delle nuove tuple da mostrare basandomi sul numero di pagine precedenti
    // a cui sono arrivato prima di aggiungere la pagina corrente (nextPage - 1) * il numero massimo di tuple che ho già mostrato in ciascuna pagina precedente (RESULTS_PER_PAGE)
    const endIndex = startIndex + RESULTS_PER_PAGE; // aggiorno l'indice di fine delle nuove tuple da mostrare che sarà dato dalla quantità di tuple
    // già mostrate (startIndex) + il numero massimo di tuple nuove che verranno mostrate (RESULTS_PER_PAGE)

    // seleziono dal resultest le tuple che vanno da 'startIndex' (estremo incluso) a 'endIndex' (estremo escluso)
    // (quindi la prima volta: startIndex=0, endIndex=99; la seconda volta startIndex=100, endIndex=199; e così via..)
    const moreResults = queryResults.slice(startIndex, endIndex);
    //console.log("startIndex:", startIndex);
    //console.log("endIndex:", endIndex);
    //console.log("moreResults:", moreResults)

    // aggiorno i risultati (le tuple) da mostrare, in particolare vengono presi i risultati attualmente 
    // visualizzati (prev) e concatenati con i nuovi risultati (moreResults). 
    // - viene eseguita una shallowCopy (maggiore efficienza) e quindi non viene creato un nuovo array ricopiando in altre celle di memoria 
    //   tutti gli elementi precedenti ma semplicemente viene creato un nuovo array che punta non solo ai vecchi elementi (rimasti fisicamente nelle stesse celle) 
    //   ma anche a quelli nuovi che sono stati aggiunti in coda.
    // - dopo l'esecuzione dell'istruzione qui sotto React sostituirà non farà più riferimento al vecchio array ma bensì a quello nuovo (più completo) e quindi
    //   il garbage collector potrà eliminare il vecchio array.
    setDisplayedResults(prev => [...prev, ...moreResults]);
    
    // Ogni volta che nuove tuple vengono caricate e aggiunte all'array 'displayedResults', il contatore di righe viene aggiornato (esteso) 
    // per riflettere il numero totale di righe visualizzate.
    // In questo modo, il contatore di righe si aggiorna dinamicamente ogni volta che carichi nuovi risultati, 
    // garantendo che ogni riga della tabella abbia un numero univoco crescente.
    // E quindi qui sotto aggiorno l'array che contiene gli indici incrementali di riga delle tuple (prima colonna vuota della tabella dei risultati):
    // - prev: rappresenta il valore attuale di rowCounter (cioè l'array contenente i numeri di riga già mostrati prima del caricamento delle nuove tuple)
    // - [...prev, ...Array.from(...)]: crea un nuovo array concatenando il vecchio array prev (cioè l'attuale contatore di righe) 
    //                                  con un nuovo array che contiene i contatori aggiuntivi per le nuove tuple
    // - ...prev: copia tutti gli elementi del vecchio array prev nel nuovo array
    // - Array.from({ length: moreResults.length }: crea un array con una lunghezza pari al numero delle nuove tuple che l'utente vuole visualizzare perchè ha eseguito lo scrolling verticale
    // - (_, i) => prev.length + i + 1: per ogni nuovo risultato, crea un numero di riga che continua dal punto in cui si era fermato 
    //                                  l'array precedente (prev.length). 'i' è l'indice corrente del nuovo array.
    // - (_, i):  funzione callback usata da Array.from per creare ciascun elemento. 
    //            Il primo parametro (indicato come _) non viene utilizzato, mentre i è l'indice dell'array, che parte da 0
    // - prev.length + i + 1:
    //      - Qui si calcola il nuovo valore del contatore:
    //          prev.length: questo è il numero di righe che sono già state contate. Quindi, il nuovo contatore inizierà da questo valore 
    //          i + 1: si aggiunge i + 1 per far sì che il nuovo contatore inizi dalla riga successiva all'ultima già presente in prev
    setRowCounter(prev => [...prev, ...Array.from({ length: moreResults.length }, (_, i) => prev.length + i + 1)]);

    setPage(nextPage); // aggiornamento numero pagine totali
  };
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Funzione speciale per attivare il caricamento di più risultati man mano che l'utente scorre verso il fondo della tabella (Lazy loading).
  useEffect(() => {
    const handleScroll = () => {
      const container = tableContainerRef.current; // prendo il riferimento alla tabella
      if (container) {
        // con l'if di sotto controllo se l'utente ha scrollato abbastanza da essere vicino al fondo del contenitore che contiene la tabella:
        // - container.clientHeight: è l'altezza complessiva del container (comprende solamente il contenuto del container che si può vedere senza scorrere - ad esempio al max: 450px - valore statico non varia)
        // - container.scrollHeight: è l'altezza complessiva del contenuto container che contiene la tabella COMPRESO il contenuto che si può vedere solo con lo scrolling (quindi in genere è molto più alto di 'container.clientHeight')
        //                           ed è variabile (anche se varia meno frequentemente rispetto a 'container.scrollTop' perchè si aggiorna solamente quando si scrolla fino alla fine di tutto il contenuto corrente)
        // - container.scrollTop: è la distanza tra la parte superiore del container e la posizione attuale dello scroll (quindi quanti pixel l'utente ha scrollato verso il basso rispetto all'inizio del container)
        //                        e quindi è variabile e varia molto frequentemente perchè varia ogni volta che l'utente scrolla (in verticale verso l'alto o verso il basso) però aumenterà solamente
        //                        quando si scrolla verso il basso (verso l'alto diminuisce). All'inizio sarà 0 ma continuando a scrollare verso il basso sicurametne supererà facilmente il valore di 'container.clientHeight'.
        //                        Potrebbe arrivare poco sotto al valore di 'container.scrollHeight' ma ci arriverà solamente quando non ci saranno altre tuple da aggiungere nella visualizzazione della tabella
        //                        perchè il valore di 'container.scrollHeight' continua ad aumentare ad ogni aggiornamento di caricamento delle nuove tuple e quindi del nuovo contenuto.
        // - Quindi se l'if di sotto è vero vuol dire che l'utente ha raggiunto (oppure è vicino) al fondo 'visualizzabile' della tabella perchè il valore [container.scrollHeight - container.scrollTop] sarà molto
        //   vicino a [container.clientHeight + 10] - il +10 serve per fare l'aggiornamento un pò prima ovvero senza aspettare che veramente l'utente arrivi con lo scrolling fino alla fine corrente e dare quindi una maggiore fluidità all'aggiornamento dell'interfaccia.
        //console.log("container.clientHeight:", container.clientHeight);
        //console.log("container.scrollHeight:", container.scrollHeight);
        //console.log("container.scrollTop:", container.scrollTop);
        //console.log("container.scrollHeight - container.scrollTop:", container.scrollHeight - container.scrollTop);
        //console.log("container.clientHeight + 10:", container.clientHeight + 10);
        //console.log("");
        //console.log("");
        if (container.scrollHeight - container.scrollTop <= container.clientHeight + 10) {
          if (!loading && displayedResults.length < queryResults.length) { // l'aggiornamento verrà richiesto solo se non c'è già un caricamento e ci sono ancora delle tuple da mostrare
            loadMoreResults();
          }
        }
      }
    };

    const container = tableContainerRef.current; // prendo il riferimento alla tabella
    if (container) {
      container.addEventListener('scroll', handleScroll); // aggiungo un listener all'evento di scroll del contenitore della tabella (ogni volta che ci sarà lo scrolling sulla tabella verrà eseguita la funzione 'handleScroll')
    }

    return () => {
      if (container) {
        container.removeEventListener('scroll', handleScroll); // rimuovo il listener quando il contenitore della tabella viene rimosso dal DOM
      }
    };
  }, [displayedResults, queryResults, loading]); // la funzione 'useEffect' verrà eseguita ogni volta che i valori di 'displayedResults', 'queryResults' o 'loading' cambiano.
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


  return (
    <div>
      {/* Titolo principale della pagina */}
      <h1 style={{ textAlign: 'center', marginTop: '40px', fontWeight: 'bold', fontSize: '36px' }}>
        Generatore di Query SQL
      </h1>

      {/* Sotto-paragrafo */}
      <p style={{ textAlign: 'center', marginBottom: '10px', marginTop: '20px', fontWeight: 'bold', fontSize: '18px', lineHeight: '1.5'}}>
        Scrivi qui sotto la query in linguaggio naturale (Inglese/Italiano) e dopo clicca sul button per generare la query SQL e ottenere i risultati:
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '20vh' }}>
        {/* TextArea */}
        <textarea
          value={nlQuery}
          onChange={(e) => setNlQuery(e.target.value)}
          placeholder="Inserisci query in linguaggio naturale"
          style={{
            padding: '12px',
            width: '450px',
            height: '160px',
            border: '2px solid #ccc',  // colore bordo più morbido
            borderRadius: '15px',  // bordi più arrotondati
            resize: 'none', // impedisco il ridimensionamento verticale da parte dell'utente
            boxShadow: '0px 4px 10px rgba(0, 0, 0, 0.1)', // leggera ombreggiatura
            fontSize: '16px',
            transition: 'border-color 0.3s ease, box-shadow 0.3s ease', // transazioni più fluide
            outline: 'none'
          }}
          onFocus={(e) => {
            e.target.style.borderColor = '#007BFF';  // colore del bordo blu chiaro quando viene cliccata la textarea (feedback visivo)
            e.target.style.boxShadow = '0px 4px 12px rgba(0, 123, 255, 0.4)'; // ombreggiatura accentuata al click della textarea (feedback visivo)
          }}
          onBlur={(e) => {
            e.target.style.borderColor = '#ccc';  // ritorna al bordo originale
            e.target.style.boxShadow = '0px 4px 10px rgba(0, 0, 0, 0.1)'; // ritorna con l'ombreggiatura leggera
          }}
        />
        {/* Button personalizzato */}
        <button
          onClick={handleSubmit}
          disabled={loading} // il button verrà disabilitato fino alla terminazione del caricamento
          style={{
            padding: '10px 20px',
            marginTop: '10px',
            backgroundColor: loading ? '#999' : '#E19243',
            color: 'white',
            border: 'none',
            borderRadius: '5px',
            cursor: 'pointer',
            fontSize: '16px',
            transition: 'background-color 0.3s ease',
          }}
          onMouseOver={(e) => !loading && (e.currentTarget.style.backgroundColor = '#C87A39')}
          onMouseOut={(e) => !loading && (e.currentTarget.style.backgroundColor = "#E19243")}
        >
          Genera SQL
        </button>
        {loading && <p>Caricamento...</p>}
        {error && <p style={{ color: 'red' }}>{error}</p>}
      </div>

      {sqlQuery && (
        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', marginTop: '20px' }}>
          <h3 style={{ fontWeight: 'bold' }}>Query SQL generata:</h3>
          <pre style={{ textAlign: 'center', whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
            {sqlQuery}
          </pre>

          {responseTime !== null && (
            <>
              <h3 style={{ fontWeight: 'bold', marginTop: '20px' }}>Tempo impiegato per la risposta:</h3>
              <pre style={{ textAlign: 'center', whiteSpace: 'pre-wrap', wordWrap: 'break-word' }}>
                {responseTime.toFixed(2)} secondi
              </pre>
            </>
          )}
        </div>
      )}

      {columnNames.length > 0 && displayedResults.length > 0 && (
        <div style={{ marginTop: '10px', maxWidth: '100%' }}>
          <h3 style={{ fontWeight: 'bold', marginBottom: '10px' }}>
            Risultati della Query (Mostrati {displayedResults.length} di {queryResults.length} risultati):
          </h3>
          {/* Contenitore scrollabile per la tabella */}
          <div
            ref={tableContainerRef}
            style={{
              maxHeight: '450px',
              overflowY: 'scroll',
              overflowX: 'auto',
              border: '1px solid #ccc',
            }}
          >
            <table className='table'>
              <thead> {/* intestazione tabella */}
                <tr> 
                  {/* successive colonne dell'intestazione (nomi colonne) */}
                  {columnNames.map((col, index) => (
                    <th key={index}
                      style={{
                        position: 'sticky',
                        top: 0,
                        zIndex: 1,
                        border: '1px solid black',
                        padding: '10px',
                        fontWeight: 'bold',
                      }}
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody> {/* corpo tabella */}
                {/* con '.map' itero sull'array 'displayedResults', ogni elemento sarà una riga della tabella */}
                {/* row: array contenente i valori di ogni riga della tabella (no riga intestazione) */}
                {/* rowIndex: indice incrementale univoco di riga generato dalla map (parte da 0) */}
                {displayedResults.map((row, rowIndex) => (
                  <tr key={rowIndex}> {/* per ogni nuova riga creo un nuovo <tr> */}
                    
                    {/* itero nuovamente su 'row' per ottenere i singoli valori (cell) di ogni riga*/}
                    {row.map((cell, cellIndex) => (
                      <td key={cellIndex}> {/* proprietà che in React server per identificare univocamente ogni cella all'interno della riga e ottimizzare la gestione del DOM */}
                        {cell !== null ? cell.toString() : 'NULL'} {/* se il valore della cella corrente è null allora mostrerò la stringa 'NULL' altrimenti converto il valore in stringa */}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
