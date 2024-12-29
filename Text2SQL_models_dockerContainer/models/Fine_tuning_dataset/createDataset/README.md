## General information

- In this folder there are 2 scripts called **'queriesAnswersBioInfoDB_v3.py'** and **'createDatasetBioInfoDB_v3.py'**.

- In **'queriesAnswersBioInfoDB_v3.py'** there are all the functions needed to create the training dataset called 'BioInfoDataset_v3.json'.

- In **'createDatasetBioInfoDB_v3.py'** there is instead the code that will take care of **creating the file 'BioInfoDataset_v3.json'** by invoking all the functions present in 'queriesAnswersBioInfoDB_v3.py'.

- If you want to create a new training dataset, just follow the same logic by first creating a new file 'queriesAnswersBioInfoDB_vX.py' and then the related file 'createDatasetBioInfoDB_vX.py' and run only the latter as shown in step 2.


## 1 - Setting paths

The steps to follow are the following:

- Open the configuration file called **'configuration.env'**.

- **BASE_PATH**: Assign to this variable the path of your base root on the remote machine. (e.g. "/beegfs/home/mmetta").

- **NAME_DATASET**: Assign to this variable the name you want to give to the dataset that will be created.


## 2 - Run 'queriesAnswersBioInfoDB_v3.py'

- To launch the **'queriesAnswersBioInfoDB_v3.py' script, after opening the ssh connection with the remote server** perform these steps:

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
      cd venvText2SQL/models/Fine_tuning_dataset/createDataset
      ```

  5) Run the script with the command:
      ```bash
      python createDatasetBioInfoDB_v3.py
      ```


## 3 - Results Information
- After the download is complete:

- Inside the folder with the path shown below you will find the **dataset just created**:

**<BASE_PATH>/venvText2SQL/models/Fine_tuning_dataset/dataset/training/<NAME_DATASET>**



