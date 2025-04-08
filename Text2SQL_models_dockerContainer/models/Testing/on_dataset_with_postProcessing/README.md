## General information

- **The only purpose of this script is to see how the model structures the generated queries according to the post-processing used in the final application.**

- In this folder there are 2 scripts, **'test_dataset.py'** and **test_dataset_improved.py**, that allows you to observe what queries the model will generate using post-processing. The queries are taken from a particular test dataset.

- **The difference between 'test_dataset.py' and 'test_dataset_improved.py'** is that the first script does not take advantage of the different acceptable versions of the test sql queries. So since the models might generate a slightly different query than a single reference query, using 'test_dataset.py' will generally yield lower accuracy results than using 'test_dataset_improved.py'. 

- The **"utils" folder** contains both the file **'post_processing_functions.py'**, which contains the functions that implement the application's post-processing, and the file **'stopping_criteria.py'**, which instead contains the class that allows the LLM to stop generating the text as soon as one of the possible termination tokens is generated.

- You will be able to configure the values ​​of the model hyperparameters in a custom way.

- The configuration file, called **'configuration.env'**, contains **variables to set the paths needed to correctly execute the 'test_dataset.py' script, for the configuration of the table structure and for the values ​​of the LLM hyperparameters**.


## 1 - Setting paths

The steps to follow are the following:

- Open the configuration file called **'configuration.env'**.

  - **BASE_PATH**: Assign to this variable the path of your base root on the remote machine. (e.g. "/beegfs/home/mmetta").

  - **PATH_TEST_SET**: Assign to this variable the relative path of the test dataset you want to use.

  - **PATH_BASE_MODEL_TEST**: Assign to this variable the relative path of the folder that contains the base model you want to test.

  - **PATH_OFFLOAD_BASE_MODEL_TEST**: Assign to this variable the relative path of the folder in which you will apply the "dynamic loading" (offloading) of parts of the base model, that is, temporarily move parts of the base model to disk when they are not in active use, instead of keeping them in RAM or VRAM.

  - **PATH_FT_MODEL_TEST**: Assign to this variable the path of the model that has already been fine-tuned and that you want to test **(if you want to test only the base model then this variable must be set empty: "")**.

  - **PATH_OFFLOAD_MODEL_TEST**: Assign to this variable the name of the folder in which the "dynamic loading" (offloading) of parts of the already retrained model will be applied **(if you want to test only the base model then this variable can be set empty: "")**.

  - **NAME_DIR_QUERY_RESULS**: Assign to this variable the name of the folder that will contain the results on the accuracy of the model and on the memory occupation during the inferences of the model.


## 2 - Database tables settings

Always stay within the **'configuration.env'** configuration file.

  - **TABLE**: If you want, you can assign to this variable the structure of the tables that will be inserted in the prompt that will be provided as input to the model for each inference. It is advisable to **modify this variable only if the model you intend to test has been fine-tuned or on different tables than the default one ('detailedstrain') or if it has been fine-tuned on the default table but with a different structure than the one already present.** In all other cases it is not necessary to modify this variable.  


## 3 - LLM hyperparameter settings

Always stay inside the **'configuration.env'** configuration file.

- The most important **hyperparameters** to set are the following:

  - **QUANTIZATION_BASE_MODEL**: Specifies whether or not the 4-bit quantization of the base model should be applied during inferences.

  - **MAX_NEW_TOKENS**: Maximum number of tokens that the fine-tuned model can generate in the answers to the queries present in the test set. During the experimentation, the value 200 was always used since it proved to be sufficient for generating the answers.

  - **STOPPING_CRITERIA** (True or False): It is used to specify whether or not to use the stop criterion to stop the generation as soon as one of the possible termination tokens is generated.

  - **TEMPERATURE**: Controls the randomness in text generation. Low values, close to 0, make the model more deterministic (usually the tokens with the highest probability are selected, since these probabilities will be further accentuated), while high values, greater than 1, increase randomness (even the tokens with the lowest probability will be chosen, since the probability distribution will be more balanced). With temperature = 1, the model maintains the original probability distribution, without adding or reducing randomness.

  - **TOP_K**: Value that indicates the range of the first 'k' tokens (ordered according to their probability values) from which to choose the one to generate in the response. For example, if set to k = 50, the model will choose only among the 50 tokens with the highest probability, discarding the others. Instead, with k = 1 the model will always select the token at the top and therefore the most probable. This parameter controls diversity, reducing the options when k is low and increasing creativity when k is high.

  - **TOP_P**: Hyperparameter similar to top_k, but dynamic, since it selects the smallest set of tokens that reaches at least a certain cumulative probability value equal to x% (assuming top_p = x). With top_k = 1, the value of top_p is overridden, making it irrelevant. In essence, unlike top_k, top_p adapts the number of tokens considered based on the cumulative probability distribution up to a certain threshold.
      
      
## 4 - Run the testing

- To launch the **'test_dataset.py' script, after opening the ssh connection with the remote server** perform these steps:

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

  4) Place yourself in the  "on_dataset" folder with the following command:  
      ```bash
      cd venvText2SQL/models/Testing/on_dataset_with_postProcessing
      ```

  5) Run the script with the command:
      ```bash
      python test_dataset.py
      ```


## 5 - Results information
- At the end of the execution:

  - Inside the folder with the path shown below there will be **2 files with different results**:

      **<BASE_PATH>/venvText2SQL/models/Testing/on_dataset_with_postProcessing/Queries_results/<NAME_DIR_QUERY_RESULS>**  
    <br>  

  1) Inside the **'query_results.txt'** file with the path shown below, there will be all the queries of the test set with the related answers generated by the model and the time taken to generate each answer:

      **<BASE_PATH>/venvText2SQL/models/Testing/on_dataset_with_postProcessing/Queries_results/<NAME_DIR_QUERY_RESULS>/query_results.txt**  
    <br>  

  2) Inside the **'memory_usage.csv'** file with the path shown below, there will be all the information related to the memory usage during the generation of each response:

      **<BASE_PATH>/venvText2SQL/models/Testing/on_dataset_with_postProcessing/Queries_results/<NAME_DIR_QUERY_RESULS>/memory_usage.csv**  
    <br>
