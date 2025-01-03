## GENERAL INFORMATION

- In this folder **"models"** you will find **all the subfolders** that will allow you to:  

    1) Get support libraries like PyTorch and Bitsandbytes **(_lib)**
    2) Download an LLM from HuggingFace **(Download_models)**
    3) Have training and test datasets available to fine-tune and test an LLM **(Fine_tuning_dataset)**
    4) Perform standard fine-tuning of an LLM **(FineTuning_models)**
    5) Perform double fine-tuning of an LLM **(FineTuning_models_double)**
    6) Perform 4-bit fine-tuning of an LLM **(FineTuning_quantization_models)**
    7) Test the performance of an LLM **(Testing)**
    8) Create charts to compare different LLMs **(Graphs)**

- **Before trying to follow any of the above points you need to follow the following steps first to be able to create a virtual environment and install all the necessary libraries inside it.**

## STEP 1: Download the **pytorch-2.3.0** library

1) Download the zip archive at the following link: https://drive.google.com/file/d/1t3V6iuZJscoW7Dt1ur5gEw6hQKqnQCjH/view?usp=drive_link

2) Unzip the downloaded file to obtain the folder containing the library

3) **Copy** the folder you just obtained **to the following path: "Text2SQL_models_dockerContainer\models\\_lib"**


## STEP 2: Download the **bitsandbytes** library

1) Download the zip archive at the following link: https://drive.google.com/file/d/1Snq27PBPR9syZMgGFN9LN15APl1ZdjZu/view?usp=drive_link

2) Unzip the downloaded file to obtain the folder containing the library

3) **Copy** the folder you just obtained **to the following path: "Text2SQL_models_dockerContainer\models\\_lib"**


## STEP 3: Create the virtual environment

1) Place yourself on a gracehopper (gh) with this command:
    ```bash
    srun -p gracehopper --gres=gpu:1 --pty bash
    ```

2) Create a new venv called **"venvText2SQL"** on gh:
    ```bash
    python3 -m venv venvText2SQL
    ```

3) Copy the **"models"** folder into the venvText2SQL virtual environment

4) Activate the venv just created always on the gh:
    ```bash
    source venvText2SQL/bin/activate
    ```

5) If you run pip freeze you get nothing because there are no libraries installed yet:
    ```bash
    pip freeze
    ```

6) Then install all the dependencies present in the 'requirements.txt':
    ```bash
    pip install -r venvText2SQL/models/requirements.txt
    ```

7) Now if you run pip freeze you get all the installed libraries:
    ```bash
    pip freeze
    ```

## STEP 4: Install Bitsandbytes library

1) Place yourself in "venvText2SQL/models/_lib/bitsandbytes":
    ```bash
    cd venvText2SQL/models/_lib/bitsandbytes
    ```

3) Install all dependencies present in the "requirements-dev.txt":
    ```bash
    pip install -r requirements-dev.txt
    ```


4) Cleaning a possible cache file 'CMakeCache.txt':
    ```bash
    rm -rf CMakeCache.txt CMakeFiles 
    ```

5) Run:
    ```bash
    cmake -DCOMPUTE_BACKEND=cuda -S . 
    ```

6) Run:
    ```bash
    make
    ```

7) Run:
    ```bash
    pip install . 
    ```

8) Run (--user is used to avoid having to enter the administrator password): 
    ```bash
    python setup.py install --user
    ```

## CONCLUSION 
- At this point, you can run any script present in any of the 6 main folders mentioned at the beginning of this file (from point 2 to 8).