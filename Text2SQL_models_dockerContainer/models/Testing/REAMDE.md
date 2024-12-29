## General information

- In this folder there are **2 subfolders**:

1) **"on_dataset_without_postProcessing"**: Contains the files needed to test an LLM on the test set without any post-processing in order to evaluate the basic performance of a model without possibly applying any corrections to the generated queries.

2) **"on_dataset_with_postProcessing"**: Contains the files needed to test an LLM on the test set using the same post-processing applied to the model in production. In this case the goal will be to observe what queries the model will generate using post-processing.