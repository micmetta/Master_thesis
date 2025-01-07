
## General information

- In this folder there is a single script **'FT_model.py'** that will allow you to fine-tune an LLM on a particular dataset.

    - You can use the above script in **different ways**:

        1) **Fine-tune an LLM once on a certain dataset, another time possibly on another dataset, and so on.**
        2) **Decide whether to quantize the base model to 4 bits or not for each fine-tuning.**

- The configuration file called **'configuration.env'**, contains **both the variables with the paths** to configure to run the script 'FT_model.py' correctly and **both those that refer to the configuration of the values ​​of the hyperparameters of the LLM**.


## 1 - Setting paths

The steps to follow are the following.

  - Open the configuration file called **'configuration.env'** and set this variables:

    - **BASE_PATH**: Set this variable to the path to your base root on the remote machine. (e.g. "/beegfs/home/mmetta").

    - **PATH_DATASET**: Set this variable to the relative path to the fine-tuning dataset.

    - **PATH_BASE_MODEL**: Set this variable to the relative path to the folder containing the base model you want to retrain. This path can also be that of a model that has already been fine-tuned once or more.

    - **PATH_OFFLOAD_BASE_MODEL_DIR**: Set this variable to the relative path to the folder where you will "offload" parts of the base model, i.e. temporarily move parts of the base model to disk when not in active use, instead of keeping them in RAM or VRAM.

    - **NAME_NEW_MODEL**: Give this variable the name of the new model that will be obtained after fine-tuning.

    - **NAME_OFFLOAD_CACHE_DIR_NEW_MODEL**: Give this variable the name of the folder in which the "dynamic loading" (offloading) of parts of the new model that will be created after fine-tuning will be applied.


## 2 - LLM Hyperparameter Settings

Always stay inside the **'configuration.env'** configuration file.

- The **most important hyperparameters** to set are the following:

  - **WAIT**: Actually, this is NOT a hyperparameter of the LLM but it is used internally by the script to monitor the memory occupation during the fine-tuning and return the results in a .csv file at the end of the computation. To this variable you must assign the number of seconds that must pass between one memory read and another. Consider that the greater the fine-tuning time, the larger the size of the output file that you will obtain at the end. So, if the dataset you plan to use is very large or the LLM has a higher number of weights (greater than 8B), it is recommended to increase the value of this variable.

  - **QUANTIZATION_BASE_MODEL**: This hyperparameter is used in the script to specify whether or not you want to quantize (to 4-bit) the base model.

  - **NUM_TRAIN_EPOCHS**: Assign the number of epochs for the fine-tuning of the model.

  - **PER_DEVICE_TRAIN_BATCH_SIZE**: Number of samples loaded into memory for each forward and backward pass until the entire dataset is exhausted.

  - **PER_DEVICE_EVAL_BATCH_SIZE**: Number of samples loaded into memory to perform the forward pass (without updating the model weights, since it is the evaluation phase and not the training phase).

  - **GRADIENT_ACCUMULATION_STEPS**: Number of batches for which the gradients will be accumulated in memory before performing the update of the model weights (backward pass).  
  **REMEMBER: Batch_size_effective = PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS so if for example PER_DEVICE_TRAIN_BATCH_SIZE=8 and GRADIENT_ACCUMULATION_STEPS=2 then the weights update will happen only after the network has viewed 16 examples and in the meantime the gradients will be accumulated**.

  - **DEVICE_MAP**: Assign the device on which the LLM will be transferred to perform fine-tuning. ("cuda:0" to use the GPU, "cpu" to use the system RAM and run the FT with cpu or "auto" to let the transformers library automatically balance the load).

- **All other remaining hyperparameters**, if desired, can be modified to influence the learning of the LLM.


## 3 - Perform fine-tuning

- To launch the **'FT_model.py' script, after opening the ssh connection with the remote server** perform these steps:

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

  4) Place yourself in the "FineTuning_models" folder with the following command: 
      ```bash
      cd venvText2SQL/models/FineTuning_models
      ```


  5) Run the script with the command:
      ```bash
      python FT_model.py
      ```


## 4 - Results information
- After fine-tuning:

  - Inside the folder with the path shown below you will find **all the files of the new fine-tuned model**:

      **venvText2SQL/models/FineTuning_models/Fine_tuned_models/Cache_dir/<NAME_NEW_MODEL>**  
    <br>

  - Inside the folder with the path shown below you will find the subfolder that will be used for offloading the new fine-tuned model:

      venvText2SQL/models/FineTuning_models/Fine_tuned_models/Offload_dir/<NAME_OFFLOAD_CACHE_DIR_NEW_MODEL>    
    <br>

  - Inside the folder with the path shown below you will find the **memory_usage.csv** file which will contain the data on memory usage:

      **venvText2SQL/models/FineTuning_models/Fine_tuned_models/Memory_occupation/<NAME_NEW_MODEL>/memory_usage.csv**  
    <br>

  - The "results" folder generated in "FineTuning_models" can be ignored and even deleted if desired.
