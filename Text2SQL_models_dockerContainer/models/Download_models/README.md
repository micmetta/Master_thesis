
## General information
- In this folder there is a single script **'download_models.py'** that can be used to download a specific LLM.

- The configuration file called **'configuration.env'** contains all the variables to be configured to run the script 'download_models.py' correctly.


## 1 - Setting paths
1) Open the configuration file called **'configuration.env'**.

2) Change the variables like this:

    - **LOGIN_TOKEN_HUGGING_FACE**: Assign to this variable your **token obtained from Hugging Face** to be able to download the model (**Typically, to download a model after you have requested access to that model you will need to use your Hugging Face token**).

    - **BASE_PATH**: Assign to this variable the path of your base root on the remote machine. (eg. "/beegfs/home/mmetta").

    - **MODEL_NAME**: Assign to this variable the name of the LLM you want to download from Hugging Face (same name present on Hugging Face, e.g. "meta-llama/Meta-Llama-3-8B-Instruct").

    - **NAME_CACHE_DIR**: Assign this variable the name of the cache folder where Hugging Face will save all the model files that will be downloaded (weights, tokenizers and other configuration files typically in .json format) to avoid having to re-download them each time.

    - **NAME_OFFLOAD_CACHE_DIR**: Assign this variable the name of the folder in which the "dynamic loading" (offloading) of parts of the model will be applied, that is, temporarily moving parts of the model to disk when they are not in active use, instead of keeping them in RAM or VRAM memory.


## 2 - Download the model

- To **launch the script, after opening the ssh connection with the remote server**, you need to execute these commands in the terminal:

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

    4) Position yourself in the **"Download_models"** folder with the following command: 
        ```bash
        cd venvText2SQL/models/Download_models
        ```

    5) Run the script with the command:
        ```bash
        python download_models.py
        ```
----------------------------------------------------------------------------------------------------------------------------------------------

## 3 - Results information
- After the execution:

    - Inside the folder with a **similar path** to the one shown below there will be the **base model downloaded from Hugging Face with its layers and all the other .json files** useful for its loading:

        **"venvText2SQL/models/Download_models/Base_models/Cache_dir/<NAME_CACHE_DIR>/models--meta-llama--Meta-Llama-<info_model>/snapshots/ <model_identifier>"**

- Typically a generic LLama model will have a path similar to the example just mentioned.
----------------------------------------------------------------------------------------------------------------------------------------------