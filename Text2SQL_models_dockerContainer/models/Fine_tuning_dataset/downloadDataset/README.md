## General information

- In this folder there is a single script **'download_dataset.py'** that was used to download from **Hugging Face**, the larger dataset called **"bugdaryan/sql-create-context-instruction"** that was later stored as **"Text2SQLDataset.json"** inside the **'venvText2SQL/models/Fine_tuning_dataset/dataset/training' folder**.

- The **script just mentioned** can be easily used to download new datasets always from Hugging Face and **convert them to the correct format required by the fine-tuning scripts of this project**.


## 1 - Setting paths

The steps to follow are the following.

- Open the configuration file called **'configuration.env'**:

    - **BASE_PATH**: Assign this variable the path of your base root on the remote machine. (eg. "/beegfs/home/mmetta").

    - **NAME_DOWNLOAD_DATASET**: Assign this variable the name of the dataset you want to download from Hugging Face. (eg. "bugdaryan/sql-create-context-instruction").

    - **NAME_DATASET**: Assign this variable the name you want to give to the dataset after it has been downloaded.


## 2 - Run the download

- To run the **'download_dataset.py' script, after opening the ssh connection with the remote server** perform these steps:

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

    4) Place yourself in the  "createDataset" folder with the following command: 
        ```bash
        cd <BASE_PATH>/venvText2SQL/models/Fine_tuning_dataset/downloadDataset
        ```

    5) Run the script with the command:
        ```bash
        python download_dataset.py
        ```


## 3 - Results Information
- After the download is complete:

- Inside the folder with the path shown below you will find **the downloaded dataset**:

    **<BASE_PATH>/venvText2SQL/models/Fine_tuning_dataset/dataset/training/<NAME_DATASET>**
