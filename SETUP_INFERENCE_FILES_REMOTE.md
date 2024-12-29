
## GENERAL INFORMATION

- **Guide to copy all files needed for inference directly to the remote server from the "mmetta" account to another generic account**. 

- In this guide, to be generic, **it is assumed** that the **base path** of the **account where the files will be copied** on the remote server is: **"base_path_new_account"**. 


## COPY FILE - REMOTE SERVER

1) Create a **new folder** for "base_path_new_account" on the remote server **called** for example **"FileInference"**. 

2) **Create in "FileInference"** 3 subfolders called:  
    - scripts
    - models
    - pythonPath

3) **Create** these subfolders **in the "models" subfolder**:
    - baseModel8B
    - model8B
    - baseModel70B
    - model70B

4) Run these commands to **copy the necessary inference scripts** into the **scripts** subfolder:

    - Copy the **"utils"** folder that contains the functions in common between the 'inference_8B.py' and 'inference_70B.py' scripts:
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/script_to_use_syntheticstrain/utils <base_path_new_account>/FileInference/scripts
        ```
    
    - Copy "job_script_8B.sh":
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/script_to_use_syntheticstrain/job_script_8B.sh <base_path_new_account>/FileInference/scripts
        ```
    
    - Copy "inference_8B.py":
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/script_to_use_syntheticstrain/inference_8B.py <base_path_new_account>/FileInference/scripts
        ```

    - Copy "job_script_70B.sh":
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/script_to_use_syntheticstrain/job_script_70B.sh <base_path_new_account>/FileInference/scripts
        ```
    
    - Copy "inference_70B.py":
        ```bash
        sudo cp /beegfs/home/mmetta/Test/TestTesi/script_to_use_syntheticstrain/inference_70B.py <base_path_new_account>/FileInference/scripts
        ```

5) Run these commands to **copy all the necessary model files** into the **models** subfolder:

    - Copy basic model to 8B:
        ```bash
        sudo cp /beegfs/home/mmetta/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B-Instruct/snapshots/5f0b02c75b57c5855da9ae460ce51323ea669d8a <base_path_new_account>/FileInference/models/baseModel8B
        ```

    - Copy retrained model to 8B:
        ```bash
        sudo cp /beegfs/home/mmetta/Text2SQL/FineTuning_modelli/Modelli_riaddestrati/Cache_dir/Llama-3-8B-Instruct-base-finetune_BioInfoDataset_v3_4Epochs_TRAIN_BATCH_SIZE_1 <base_path_new_account>/FileInference/models/model8B
        ```

    - Copy basic model to 70B:
        ```bash
        sudo cp /beegfs/home/mmetta/Text2SQL/Download_modelli/Modelli_base/Cache_dir/Cache_dir_Llama3-70B-Instruct/models--meta-llama--Meta-Llama-3-70B-Instruct/snapshots/05917295788658563fd7ef778b6240ad9867d6d1 <base_path_new_account>/FileInference/models/baseModel70B
        ```
    - Copy retrained model to 70B:
        ```bash
        /beegfs/home/mmetta/Text2SQL/FineTuning_modelli_quantizzati/Modelli_riaddestrati_quantizzati/Cache_dir/Llama-3-70B/Llama-3-70B-Instruct-base-finetune_BioInfoDataset_v3_1Epoch_TRAIN_BATCH_SIZE_1_4bit <base_path_new_account>/FileInference/models/model70B
        ```

6) Run this command to **copy all the PyTorch library files** needed to load models and run inferences:

    - Copy export PYTHONPATH:
        ```bash
        sudo cp /beegfs/home/mmetta/pytorch-2.3.0/lib/python3.10/site-packages/ <base_path_new_account>/FileInference/pythonPath
        ```

## CONFIGURING THE PROJECT ".env.local" FILE

- Now you need to **go inside the "interface/.env.local" folder of the project** and set the following configuration variables like this:

    ```bash
    NEXT_PUBLIC_PATH_FILE_INFERENCES=<base_path_new_account>/FileInference/scripts
    ```

    ```bash
    EXPORT_PYTHON_PATH=<base_path_new_account>/FileInference/pythonPath
    ```

    ```bash
    PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_8B_SCRIPT=<base_path_new_account>/FileInference/models/baseModel8B  
    ```

    ```bash
    PATH_MODEL_USE_REMOTE_FOR_INFERENCE_8B_SCRIPT=<base_path_new_account>/FileInference/models/model8B  
    ```

    ```bash
    PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_70B_SCRIPT=<base_path_new_account>/FileInference/models/baseModel70B  
    ```

    ```bash
    PATH_MODEL_USE_REMOTE_FOR_INFERENCE_8B_SCRIPT=<base_path_new_account>/FileInference/models/model70B  
    ```


## END
- At this point inside the **"FileInference"** folder there will be all the scripts and models needed for the application backend to function correctly.