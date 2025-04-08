## General information

- In this folder there is 1 scripts called **'main.py'**.

- In **'main.py'** there is the code that was used to create the BioInfoDataset. The same code can be used to extend this dataset to increase the variety of examples.

- You can specify in which dataset (training or validation or test) you want to add new examples. Regardless of the specific dataset you choose, the new examples will never be identical to those in the other datasets to avoid overlapping examples.


## 1 - Setting paths

The steps to follow are the following:

- Open the configuration file called **'configuration.env'**.

- **BASE_PATH**: Assign to this variable the path of your base root on the remote machine. (e.g. "/beegfs/home/mmetta").

- **PATH_COLUMN_VALUES_DB**: Assign this variable the path associated with the json file from which the column names and values ​​that refer to the biological database will be taken.

- **PATH_COLUMNS_DATE_VALUES_DB**: Assign this variable the path of the json file that contains the name of the date columns within biological database.

- **PATH_AUTOMATIC_QUERIES_FOLDER**: This is the path of the "automatic_queries" folder (You can use the same path already setting).

- **NAME_LAST_TRAINING_SET**: Name of the last training set created. Setting this name **is important because it prevents examples from being generated that are already in the training set**.

- **NAME_FILE_NAME_OUTPUT_TRAINING_SET**: Name of the new training set that will be created.

- **NAME_LAST_VALIDATION_SET**: Name of the last validation set created. Setting this name **is important because it prevents examples from being generated that are already in the validation set**.

- **NAME_FILE_NAME_OUTPUT_VALIDATION_SET**: Name of the new validation set that will be created.

- **NAME_LAST_TEST_SET_EN**: Name of the last validation set created. Setting this name **is important because it prevents generating examples that are already in the English test set**.

- **NAME_LAST_TEST_SET_IT**: Name of the last validation set created. Setting this name **is important because it prevents generating examples that are already in the Italian test set**.

- **NAME_FILE_NAME_OUTPUT_TEST_SET_EN**: Name of the new English test set that will be created.

- **NAME_FILE_NAME_OUTPUT_TEST_SET_IT**: Name of the new Italian test set that will be created.

- **DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES**: Generic name of the dataset which will be create and so that will contain the new generated queries (the possible values are: "training" or "validation" or "test").

- **N_QUERIES_TO_GENERATE_C1**: You can specify the number of queries of category C1 that you want generate.
- **N_QUERIES_TO_GENERATE_C2**: You can specify the number of queries of category C2 that you want generate.
- **N_QUERIES_TO_GENERATE_C3**: You can specify the number of queries of category C3 that you want generate.
- **N_QUERIES_TO_GENERATE_C4**: You can specify the number of queries of category C4 that you want generate.
- **N_QUERIES_TO_GENERATE_C5**: You can specify the number of queries of category C5 that you want generate.



## 2 - Run 'main.py'

- To launch the **'main.py'** script, after opening the ssh connection with the remote server** perform these steps:

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
      cd venvText2SQL/models/Fine_tuning_dataset/createDataset/automatic_queries
      ```

  5) Run the script with the command:
      ```bash
      python main.py
      ```


## 3 - Results Information
After that the execution is completed:

Inside the folder with the following path, you will get the new dataset: 

- venvText2SQL/models/Fine_tuning_dataset/dataset/**DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES/<NAME_DATASET>**

Where **<NAME_DATASET>** can be the name of the json file assigned to one of this variables:  
- **NAME_FILE_NAME_OUTPUT_TRAINING_SET** (If **DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES="training"**) or 
- **NAME_FILE_NAME_OUTPUT_VALIDATION_SET** (If **DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES="validation"**) or 
- **NAME_FILE_NAME_OUTPUT_TEST_SET_EN** and **NAME_FILE_NAME_OUTPUT_TEST_SET_IT** (If **DATASET_IN_WHICH_TO_ADD_THE_NEW_GENERATED_QUERIES="test"**)