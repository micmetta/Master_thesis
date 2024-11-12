"use client"; // per specificare che questo file sarà eseguito sul client-side nell'app next.js e quindi tutte le le stampe con console.log saranno mostrate nel broswer

// - useState per gestire lo stato di ogni component
// - useEffect è uno Hook (funzione speciale di Javascript che permette di utilizzare le funzionalità di React) per gestire il caricamento dati dopo il rendering
// - useRef per creare una variabile in grado di gestire il comportamento dello scrolling della tabella
import React, { useState, useEffect, useRef, useCallback} from 'react';
import { handleButtonClick } from '@/lib/page/actionsPage/handleButtonClick';
import "./table.css";


export default function Home() {
  // - Dichiarazione degli stati (valore attuale dello stato, funzione per aggiornare lo stato e forzare il re-render del componente):
  const [nlQuery, setNlQuery] = useState<string>('');  // stato per memorizzare la query NL (inizializzato con stringa vuota)
  const [sqlQuery, setSqlQuery] = useState<string>(''); // stato per memorizzare la query SQL generata dal backend
  const [columnNames, setColumnNames] = useState<string[]>([]); // stato per i nomi delle colonne
  const [queryResults, setQueryResults] = useState<string[][]>([]); // stato per le tuple dei risultati (queryResults conterrà le tuple del resultest restituito dal backend)
  const [displayedResults, setDisplayedResults] = useState<string[][]>([]); // stato per i risultati mostrati (gruppi di 100)
  const [loading, setLoading] = useState<boolean>(false); // stato di caricamento durante l'inferenza richiesta
  const [error, setError] = useState<string | null>(null); // stato per l'errore
  const [unavailabilityNode, setUnavailabilityNode] = useState<string | null>(null); // stato per informare l'utente sull'indisponibilità momentanea di un nodo di esecuzione dell'inferenza richiesta
  const [page, setPage] = useState<number>(1); // stato per il numero di pagine caricate
  const [responseTime, setResponseTime] = useState<number | null>(null); // stato per calcolare il tempo di risposta
  const RESULTS_PER_PAGE = 100; // numero di risultati per pagina
  const tableContainerRef = useRef<HTMLDivElement>(null); // riferimento per il contenitore della tabella scrollabile utilizzato per caricamento di più risultati quando l'utente scorre
  const [model, setModel] = useState<string>('8B'); // stato che memorizza la scelta del modello selezionato dall'utente (Default: modello 8B) 
  const [tooltipText, setTooltipText] = useState<string>(''); // Stato per il testo del tooltip
  
  // - Definisco 'handleButtonClickWrapper' (compatibile con 'onClick') che al suo interno invoca la 'handleButtonClick' che si preoccuperà di
  //   gestire la richiesta.
  const handleButtonClickWrapper = async () => {
    await handleButtonClick(
        nlQuery,
        model,
        setLoading,
        setError,
        setSqlQuery,
        setColumnNames,
        setQueryResults,
        RESULTS_PER_PAGE,
        setDisplayedResults,
        setPage,
        setResponseTime,
        setUnavailabilityNode
    );
  };

  // Funzione ausiliaria alla 'useEffect()' di sotto.
  const loadMoreResults = useCallback(() => {
    const nextPage = page + 1; // calcolo pagina successiva (prima pagina = 1)
    const startIndex = (nextPage - 1) * RESULTS_PER_PAGE; // aggiorno l'indice di partenza delle nuove tuple da mostrare basandomi sul numero di pagine precedenti
    // a cui sono arrivato prima di aggiungere la pagina corrente (nextPage - 1) * il numero massimo di tuple che ho già mostrato in ciascuna pagina precedente (RESULTS_PER_PAGE)
    const endIndex = startIndex + RESULTS_PER_PAGE; // aggiorno l'indice di fine delle nuove tuple da mostrare che sarà dato dalla quantità di tuple
    // già mostrate (startIndex) + il numero massimo di tuple nuove che verranno mostrate (RESULTS_PER_PAGE)

    // seleziono dal resultest le tuple che vanno da 'startIndex' (estremo incluso) a 'endIndex' (estremo escluso)
    // (quindi la prima volta: startIndex=0, endIndex=99; la seconda volta startIndex=100, endIndex=199; e così via..)
    const moreResults = queryResults.slice(startIndex, endIndex);

    // aggiorno i risultati (le tuple) da mostrare, in particolare vengono presi i risultati attualmente visualizzati (prev) 
    // e concatenati con i nuovi risultati (moreResults). 
    // - viene eseguita una shallowCopy (maggiore efficienza) e quindi non viene creato un nuovo array ricopiando in altre celle di memoria 
    //   tutti gli elementi precedenti ma semplicemente viene creato un nuovo array che punta non solo ai vecchi elementi (rimasti fisicamente nelle stesse celle) 
    //   ma anche a quelli nuovi che sono stati aggiunti in coda.
    // - dopo l'esecuzione dell'istruzione qui sotto, React non farà più riferimento al vecchio array ma bensì a quello nuovo (più completo) e quindi
    //   il garbage collector potrà eliminare il vecchio array.
    setDisplayedResults(prev => [...prev, ...moreResults]);
    setPage(nextPage); // aggiornamento numero pagine totali
  }, [page, RESULTS_PER_PAGE, queryResults]);
  
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
  }, [displayedResults, queryResults, loading, loadMoreResults]); // la funzione 'useEffect' verrà eseguita ogni volta che i valori di 'displayedResults', 'queryResults' o 'loading' cambieranno.
  

  return (
    <div>
      {/* Titolo principale della pagina */}
      <h1 style={{ textAlign: 'center', marginTop: '40px', fontWeight: 'bold', fontSize: '36px' }}>
        Generatore di Query SQL
      </h1>

      {/* Sotto-paragrafo (fontSize: 'portato da 18px a16px') */}
      <p style={{ textAlign: 'center', marginBottom: '30px', marginTop: '20px', fontWeight: 'bold', fontSize: '17px', lineHeight: '1.5'}}>
        Seleziona prima il modello da utilizzare durante l&apos;inferenza, dopodichè scrivi la query in linguaggio naturale (Inglese/Italiano) e infine clicca sul button per generare la query SQL e ottenere i risultati.
      </p>

      <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', minHeight: '20vh' }}>

        {/* Selettore per scegliere il modello */}
        <label htmlFor="modelSelect" style={{ marginBottom: '10px', fontWeight: 'bold', fontSize: '17px' }}>Scegli il modello LLM:</label>
        
        <div style={{ position: 'relative', display: 'inline-block' }}>
        <select
          id="modelSelect"
          value={model}
          onChange={(e) => setModel(e.target.value)}
          onMouseOver={(e) => {
            const target = e.target as HTMLSelectElement; // cast a HTMLSelectElement
            if (target.value === '8B') {
              setTooltipText("Accuratezza minore (92%-Inglese, 83%-Italiano) - tempo di attesa medio minore (30 secondi)");
            } else {
              setTooltipText("Accuratezza maggiore (100%-Inglese, 96%-Italiano) - tempo di attesa medio maggiore (70 secondi)");
            }
          }}
          onMouseOut={() => setTooltipText('')} // nascondo il tooltip quando il mouse esce
          style={{
            padding: '5px',
            fontSize: '16px',
            marginBottom: '20px',
            border: '2px solid #ccc',  // colore del bordo
            borderRadius: '8px',  // per smussare leggermente gli angoli
            transition: 'border-color 0.3s ease, box-shadow 0.3s ease',  // transizione fluida
            outline: 'none',
            height: '40px',
            width: 'auto',
          }}
          onFocus={(e) => {
            e.target.style.borderColor = '#007BFF';  // colore del bordo quando viene cliccato
            e.target.style.boxShadow = '0px 4px 12px rgba(0, 123, 255, 0.4)'; // ombreggiatura accentuata al click
          }}
          onBlur={(e) => {
            e.target.style.borderColor = '#ccc';  // ritorna al colore del bordo originale
            e.target.style.boxShadow = 'none';  // ritorna senza ombreggiatura
          }}
        >
          <option value="8B">Modello 8B</option>
          <option value="70B">Modello 70B</option>
        </select>

          {/* Tooltip dinamico */}
          {tooltipText && (
            <div
              style={{
                position: 'absolute',
                top: '0%', // centro il tooltip verticalmente rispetto al selettore
                left: '105%', // sposto il tooltip a destra del selettore
                backgroundColor: '#333',
                color: '#fff',
                padding: '5px 10px',
                borderRadius: '5px',
                marginTop: '5px',
                fontSize: '14px',
                whiteSpace: 'nowrap',
                zIndex: 1
              }}
            >
              {tooltipText}
            </div>
          )}
        </div>

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
        {/* Button personalizzato - 'Genera SQL' */}
        <button
          onClick={handleButtonClickWrapper} // con file batch
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

        {loading && (
          <>
            <div className="spinner" style={{
              width: '24px',
              height: '24px',
              border: '4px solid #f3f3f3', // colore esterno
              borderTop: '4px solid #3498db', // colore rotellina
              borderRadius: '50%',
              animation: 'spin 1s linear infinite', // animazione circolare
              marginTop: '20px'
            }}></div>
            <p>Caricamento richiesta..</p>
          </>
        )}

        {/* se si verifica un errore, esso verrà mostrato in rosso subito sotto il button */}
        {error && (
          <p style={{
            color: 'red',
            whiteSpace: 'pre-wrap',  // gestisce meglio le nuove righe e gli spazi
            wordWrap: 'break-word',  // permette al testo di andare a capo senza sforzarsi di stare su una sola linea
            maxWidth: '100%'  // assicura che il contenitore non abbia una larghezza limitata
          }}>
            {error}
          </p>
        )}

        {/* se si verifica un tempo eccessivo di attesa, verrà mostrato in blu subito sotto il button questo avviso per l'utente */}
        {unavailabilityNode && (
          <p style={{
            color: 'blue',
            whiteSpace: 'pre-wrap',  // gestisce meglio le nuove righe e gli spazi
            wordWrap: 'break-word',  // permette al testo di andare a capo senza sforzarsi di stare su una sola linea
            maxWidth: '100%'  // assicura che il contenitore non abbia una larghezza limitata
          }}>
            {unavailabilityNode}
          </p>
        )}
      
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
