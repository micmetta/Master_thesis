# Installazione

Per installare l'applicazione TEXT2SQL, segui i seguenti passaggi:

## 1 - Scaricare il progetto
Puoi scaricare il progetto dal repository GitHub tramite il seguente comando:

```bash
git clone git@github.com:qBioTurin/NL2SQL.git
```

## 2 - Setting cartella db_scripts (opzionale)
All'interno della cartella **"db_scripts"** sono presenti questi 2 files:

1) **'bioinfoDB.sql'**: contenente i comandi SQL utili per creare localmente il database biologico. Qualora avessi a disposizione una versione più
aggiornata di tale DB, potrai semplicemente sostituire questo file con il tuo.
                    
2) **'experimentalDB.sql'**: contenente i comandi SQL utili per creare localmente il database che conterrà le queries richieste dagli utenti. 
**NON c'è bisogno di modificare/sostituire questo file tranne nel caso in cui si volessero apportare delle modifiche a questo DB**.

## 3 - Setting file .env.local
All'interno della cartella **"Interface"** è presente un file di configurazione chiamato **'.env.local'**.  
Esso contiene al proprio interno tutte le variabili di configurazione necessarie per il corretto funzionamento dell'applicazione. 

Il file è suddiviso in 4 parti:

1) Variabili di **configurazione** per l'accesso al database **"bioInfoDB"** **(modifica opzionale)**: 

    - **POSTGRES_HOST**: Specifica l'host del database ma poichè utilizziamo Docker, al posto di localhost, bisogna inserire il nome del servizio definito 
                     nel docker-compose.yml (che nel nostro caso è 'db') poichè i containers comunicano tra loro utilizzando i nomi dei servizi.
    - **POSTGRES_PORT**: Indica la porta su cui il servizio PostgreSQL del database bioInfoDB è in ascolto all'interno del container. Solitamente, la 
                     porta interna del container è 5432, anche se volendo si potrebbe mappare una porta esterna diversa.
    - **POSTGRES_USER**: L'utente utilizzato per accedere al database.
    - **POSTGRES_PASSWORD**: La password associata all'utente per accedere al database.
    - **POSTGRES_DB**: Il nome del database al quale l'applicazione si connette.

2) Variabili di **configurazione** per l'accesso al database **"experimentalDB"** **(modifica opzionale)**: 

    - **POSTGRES_HOST**: Specifica il nome del servizio definito nel docker-compose.yml (che in questo caso è 'experimentaldb').
    - **POSTGRES_PORT**: Indica la porta su cui il servizio PostgreSQL del database experimentalDB è in ascolto all'interno del container.
    - **POSTGRES_USER**: L'utente utilizzato per accedere al database.
    - **POSTGRES_PASSWORD**: La password associata all'utente.
    - **POSTGRES_DB**: Il nome del database sperimentale al quale l'applicazione si connette.

3) Variabili di **configurazione** sia per la **connessione SSH al server remoto**, sia per specificare i percorsi rilevanti per i **file generati durante l'esecuzione delle inferenze su SLURM** **(modifica obbligatoria)** e sia per gestire **i percorsi dei modelli da richiamare per eseguire le inferenze**:

    - **NEXT_PUBLIC_SSH_USER**: Assegnare a questa variabile il nome utente che verrà utilizzato per effettuare la connessione SSH al server remoto.   

    - **NEXT_PUBLIC_SSH_SERVER**: Assegnare a questa variabile l'indirizzo del server remoto al quale verrà effettuata la connessione. Sia la connessione, che tutte le successive richieste di esecuzione di script verso questo server, avverranno tramite l'utilizzo del protocollo SSH.

    - **NEXT_PUBLIC_PATH_FILE_INFERENCES**: Assegnare a questa variabile il percorso base che rappresenterà la directory principale sul server remoto nella quale ci saranno, sia gli script che si preoccuperanno di eseguire l'inferenza e sia i file di output delle inferenze richieste da parte degli utenti che poi verranno cancellati una volta che questi non saranno più necessari.  
          
        In particolare, quest'ultimo percorso verrà utilizzato all'interno dell'applicazione nei seguenti modi:
        
        **1)** Come percorso base per salvare i file di inferenza.  

        **2)** Per specificare il percorso completo di uno script .sh che sarà eseguito da SLURM (in base al modello selezionato dall'utente).  
               (ad esempio, `NEXT_PUBLIC_PATH_FILE_INFERENCES/job_script_${model}.sh`).  

        **3)** Per costruire il percorso del file di output dell'inferenza di uno specifico utente.  
               (ad esempio, `NEXT_PUBLIC_PATH_FILE_INFERENCES/output_querySQL_${userId}.txt`).

        **4)** Per costruire i percorsi dei file di output che dovranno essere cancellati.  
               (ad esempio, `NEXT_PUBLIC_PATH_FILE_INFERENCES/output_querySQL*` e `NEXT_PUBLIC_PATH_FILE_INFERENCES/slurm*`) 
    
    - **EXPORT_PYTHON_PATH**: Assegnare a questa variabile il percorso della directory presente sulla macchina remota contenente tutte le librerie di Python (con le versioni corrette) utili per il corretto funzionamento dei file '.sh' e '.py' per l'esecuzione delle inferenze.

    - **PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_8B_SCRIPT**: Assegnare a questa variabile il percorso del **modello base a 8B** presente nella cartella del tuo account sul **server remoto**.

    - **PATH_MODEL_USE_REMOTE_FOR_INFERENCE_8B_SCRIPT**: Assegnare a questa variabile il percorso del modello a **8B riaddestrato** presente nella cartella del tuo account sul **server remoto**. Questo sarà il modello che verrà utilizzato per gestire l'inferenza quando l'utente invierà una richiesta specificando l'uso del modello ad 8B.

    - **PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_70B_SCRIPT**: Assegnare a questa variabile il percorso del **modello base a 70B** presente nella cartella del tuo account sul **server remoto**.

    - **PATH_MODEL_USE_REMOTE_FOR_INFERENCE_70B_SCRIPT**: Assegnare a questa variabile il percorso del modello a **70B riaddestrato** presente nella cartella del tuo account sul **server remoto**. Questo sarà il modello che verrà utilizzato per gestire l'inferenza quando l'utente invierà una richiesta specificando l'uso del modello a 70B.

4) Variabile di **configurazione** del file **contenente la tua chiave privata** **(modifica obbligatoria)**:
    - **PRIVATE_KEY_PATH**: Assegnare a questa variabile il **percorso del file 'id_rsa'** presente sulla tua macchina che contiene la tua **chiave privata** con la quale ti collegherai al server remoto tramite SSH.

## 4 - Eseguire il software

**Da questo step in poi dovrai avere Docker in esecuzione sulla tua macchina.** 

- Per lanciare l'applicazione la prima volta e costruire quindi i containers con le immagini e avviarli, basterà posizionarsi all'interno della root del progetto, aprire il terminale ed eseguire questo comando:

```bash
docker-compose --env-file ./interface/.env.local up --build
```

- Se invece la build è stata già eseguita, basterà sempre posizionarsi all'interno della root del progetto, aprire il terminale ed eseguire questo comando:
```bash
docker-compose --env-file ./interface/.env.local up
```

- Per essere sicuri che l'intera applicazione sia pronta per l'utilizzo, bisognerà attendere fino a quando nel terminale compariranno solamente messaggi  relativi a "nextjs_app", dopodichè si potrà procedere con lo step 6.

## 5 - Utilizzare l'interfaccia grafica

A questo punto basterà aprire un browser qualsiasi e inserire l'indirizzo "http://localhost:3000/" per poter inviare le richieste all'applicazione.
