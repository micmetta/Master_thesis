## General information

- In this folder there is 1 scripts called **'duplicate_check.py'**.

- In **'duplicate_check.py'** there is the code that was used to check if there were overlaps both within the training, validation and test sets and between them.


## 1 - Setting paths

The steps to follow are the following:

- Open the configuration file called **'configuration.env'**.

- **BASE_PATH**: Assign to this variable the path of your base root on the remote machine. (e.g. "/beegfs/home/mmetta").

- **PATH_DATASET_TRAINING**: Path of training set.

- **PATH_DATASET_VALIDATION**: Path of validation set.

- **PATH_DATASET_TEST**: Path of test set.


## 2 - Run 'main.py'

- To launch the **'duplicate_check.py'** script, after opening the ssh connection with the remote server** perform these steps:

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

  4) Place yourself in the  "automatic_queries" folder with the following command:  
      ```bash
      cd venvText2SQL/models/Fine_tuning_dataset/createDataset/automatic_queries/duplicate_check
      ```

  5) Run the script with the command:
      ```bash
      python duplicate_check.py
      ```