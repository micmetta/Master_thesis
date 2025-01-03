## General information

- In this subdirectory there is a single script **'FT_quantized_model.py'** which will allow the fine-tuning of any LLM on any dataset which can be for example 'BioInfoDataset' or 'Text2SQLDataset' or some other generic dataset by exploiting 4-bit quantization.

- In particular, the weights modified during fine-tuning in this case will be 4-bit quantized and stored separately from the base model used.

- The configuration file called **'configuration.env'**, contains within it **both the variables with the paths** to be configured in order to execute the 'FT_quantized_model.py' script correctly and **both those referring to the configuration of the LLM hyperparameter values**.


## 1 - Setting paths

The steps to follow are as follows.

 - Open the configuration file called **'configuration.env'**.

    - **BASE_PATH**: Assign to this variable the path to your base root on the remote machine. (e.g. ‘/beegfs/home/mmetta).

    - **PATH_DATASET**: Contains the relative path to the fine-tuning dataset.

    - **PATH_BASE_MODEL**: Assign to this variable the relative path to the folder containing all components of the model that you want to finetune using 4-bit quantization (weights, tokenizers and other configuration files typically in .json format).

    - **PATH_OFFLOAD_BASE_MODEL_DIR**: Assign to this variable the relative path to the folder where the 'dynamic loading' (offloading) of parts of the base model will be applied, i.e. temporarily moving parts of the base model to disk when not in active use, instead of keeping them in RAM or VRAM.

    - **NAME_NEW_MODEL**: Assign to this variable the name of the new model that will be obtained after quantized fine-tuning.

    - **NAME_OFFLOAD_CACHE_DIR_NEW_MODEL**: Assign this variable the name of the folder where the 'dynamic loading' (offloading) of parts of the new model that will be created after quantized fine-tuning will be applied.


## 2 - LLM Hyperparameter Settings

Always stay inside the **'configuration.env'** configuration file.

- The **most important hyperparameters** to set are the following:

  - **WAIT**: Actually, this is NOT a hyperparameter of the LLM but it is used internally by the script to monitor the memory occupation during the fine-tuning and return the results in a .csv file at the end of the computation. To this variable you must assign the number of seconds that must pass between one memory read and another. Consider that the greater the fine-tuning time, the larger the size of the output file that you will obtain at the end. So, if the dataset you plan to use is very large or the LLM has a higher number of weights (greater than 8B), it is recommended to increase the value of this variable.

  - **NUM_TRAIN_EPOCHS**: Assign the number of epochs for the fine-tuning of the model.

  - **PER_DEVICE_TRAIN_BATCH_SIZE**: Number of samples loaded into memory for each forward and backward pass until the entire dataset is exhausted.

  - **PER_DEVICE_EVAL_BATCH_SIZE**: Number of samples loaded into memory to perform the forward pass (without updating the model weights, since it is the evaluation phase and not the training phase).

  - **GRADIENT_ACCUMULATION_STEPS**: Number of batches for which the gradients will be accumulated in memory before performing the update of the model weights (backward pass).  
  **REMEMBER: Batch_size_effective = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS so if for example PER_DEVICE_TRAIN_BATCH_SIZE=8 and GRADIENT_ACCUMULATION_STEPS=2 then the weights update will happen only after the network has viewed 16 examples and in the meantime the gradients will be accumulated**.

  - **DEVICE_MAP**: Assign the device on which the LLM will be transferred to perform fine-tuning. ("cuda:0" to use the GPU, "cpu" to use the system RAM and run the FT with cpu or "auto" to let the transformers library automatically balance the load).

  - The part of the variables that refer to the **bitsandbytes parameters** are used to activate the 4-bit quantization and can be left unchanged unless there are particular needs.

- **All other remaining hyperparameters** can be modified if desired to influence the LLM learning.


## 3 - Perform quantized fine-tuning

- To launch the **'FT_model_quantized.py' script, after opening the ssh connection with the remote server** perform these steps:

    1) Position yourself on a machine, for example on a Gracehopper:
        ```bash
        srun -p gracehopper --gres=gpu:1 --pty bash
        ```

    2) Active the virtual environment:
        ```bash
        source venvText2SQL/bin/activate
        ```

    3) Add the PyTorch library present on your remote account to the Python package search path, by executing the command (**e.g. <BASE_PATH> = /beegfs/home/mmetta**):
        ```bash
        export PYTHONPATH="<BASE_PATH>/venvText2SQL/models/_lib/pytorch-2.3.0/lib/python3.10/site-packages/"
        ```

    4) Place yourself in the "FineTuning_quantization_models" folder with the following command: 
        ```bash
        cd venvText2SQL/models/FineTuning_quantization_models
        ```

    5) Run the script with the command:
        ```bash
        python FT_quantized_model.py
        ```


## 4 - Results information
- After fine-tuning:

  - Inside the folder with the path shown below there will be **all the files of the new re-trained and 4-bit quantized model**:

      **venvText2SQL/models/FineTuning_quantization_models/Fine_tuned_quantized_models/Cache_dir/<NAME_NEW_MODEL>**  
      <br>

  - Inside the folder with the path shown below you will find the subfolder that will be used for offloading the new model after 4-bit fine-tuning:

      venvText2SQL/models/FineTuning_quantization_models/Fine_tuned_quantized_models/Offload_dir/<NAME_OFFLOAD_CACHE_DIR_NEW_MODEL>  
     <br>

  - Inside the folder with the path shown below you will find the **memory_usage.csv** file which will contain the data on memory usage:

      **venvText2SQL/models/FineTuning_quantization_models/Fine_tuned_quantized_models/Memory_occupation/<NAME_NEW_MODEL>/memory_usage.csv**  
      <br>

  - The "results" folder generated in "FineTuning_quantization_models" can be ignored and even deleted if desired.

