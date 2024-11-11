
## INFORMAZIONI GENERALI

- **Guida per copiare tutti i file necessari per l'inferenza direttamente sul server remoto dall'account "mmetta" ad un altro account generico**. 

- In questa guida, per essere generici, si suppone che il percorso base dell'account nel quale verranno copiati i file sul server remoto sia: **base_path_new_account**. 


## COPIA FILE - REMOTE SERVER

1) Creare una **nuova cartella** per "base_path_new_account" sul server remoto chiamandola ad esempio **"FilesInference"**.  
<br>  

2) Creare all'interno della cartella creata al passo precedente **2 sottocartelle** chiamate:  
    - scripts
    - models
    <br>  

3) Creare queste sottocartelle nella sottocartella **models**:
    - model8B
    - baseModel70B
    - model70B
    <br>

4) Eseguire questi comandi per **copiare gli script delle inferenze** necessari nella sottocartella **scripts**:
    
    - Copia "job_script_8B.sh":
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/job_script_8B.sh <base_path_new_account>/FilesInference/scripts
        ```
    
    - Copia "inference_8B.py":
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/inference_8B.py <base_path_new_account>/FilesInference/scripts
        ```

    - Copia "job_script_70B.sh":
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/job_script_70B.sh <base_path_new_account>/FilesInference/scripts
        ```
    
    - Copia "inference_70B.py":
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/inference_70B.py <base_path_new_account>/FilesInference/scripts
        ```
    <br>  

5) Eseguire questi comandi per **copiare tutti i file dei modelli** necessari nella sottocartella **models**:

    - Copia modello riaddestrato a 8B:
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/modelli_riaddestrati/meta-llama/Meta-Llama-3-8B-Instruct-base-finetune_TRAINING_2_16bit_fp16_10Epochs <base_path_new_account>/FilesInference/models/model8B
        ```

    - Copia modello base a 70B:
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/modelli_riaddestrati/meta-llama/Meta-Llama-3-70B-Instruct-base-finetune-text2sql_4bit <base_path_new_account>/FilesInference/models/baseModel70B
        ```
    - Copia modello riaddestrato a 70B:
        ```bash
        /beegfs/home/mmetta/Test/TestTesi/modelli_riaddestrati/meta-llama/Meta-Llama-3-70B-Instruct-base-finetune-text2sql-finetune-TRAINING_2_4bit_4bit_10Epochs <base_path_new_account>/FilesInference/models/model70B
        ```

## CONFIGURAZIONE DEL file ".env.local" DEL PROGETTO

- Ora è necessario **andare all'interno della cartella "interface/.env.local" del progetto** e settare in questo modo le seguenti variabili di configurazione:

    1) **NEXT_PUBLIC_PATH_FILE_INFERENCES=<base_path_new_account>**/FilesInference/scripts  
        <br>  

    2) **PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_8B_SCRIPT=meta-llama/Meta-Llama-3-8B-Instruct**  
        <br>

    3) **PATH_MODEL_USE_REMOTE_FOR_INFERENCE_8B_SCRIPT=<base_path_new_account>**/FilesInference/models/model8B  
        <br>

    4) **PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_70B_SCRIPT=<base_path_new_account>**/FilesInference/models/baseModel70B  
        <br>

    5) **PATH_MODEL_USE_REMOTE_FOR_INFERENCE_8B_SCRIPT=<base_path_new_account>**/FilesInference/models/model70B  
        <br>


## END
- A questo punto all'interno della cartella **"FilesInference"** ci saranno tutti gli script e i modelli necessari per il corretto funzionamento del backend dell'applicazione.