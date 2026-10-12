# Installation

To install the TEXT2SQL application, follow these steps:

## 1 - Download the project
You can download the project from the GitHub repository using the following command:

```bash
git clone -b code https://github.com/micmetta/Master_thesis.git
```

## 2 - Setting cartella db_scripts (optional)
In the 'db_scripts' directory you will find these 2 files:

1) **'bioinfoDB.sql'**: which contains SQL commands to create the biological database locally. If you have a new version of this DB, you can simply add this file to your.
                    
2) **'experimentalDB.sql'**: which contains SQL commands useful for creating the biological database locally. This DB will contain the queries requested by the users.
**There is NO need to edit/replace this file unless you want to make changes to this DB**.


## 3 - Setup file on the remote machine
- Open the file **'SETUP_INFERENCE_FILES_REMOTE.md' e follow the steps in the file**. This will allow you to correctly setup all models and all scripts on the remote machine that will be called by the Next.js application. 

- **Then, come back here and continue with the step 4**.


## 4 - Setting file .env.local
Inside the **"Interface"** folder there is a configuration file called **'.env.local'**.
It contains all the configuration variables needed for the application to work correctly. 

The file consists of 4 parts:

1) **Configuration** variables for accessing the **bioInfoDB** database **(optional update)**: 

    - **POSTGRES_HOST**: Specify the database host, but since we are using Docker, instead of 'localhost' we need to specify the name of the service defined in docker-compose.yml (which in our case is 'db'), because the containers communicate with each other using the names of the services.
    - **POSTGRES_PORT**: Specifies the port on which the bioInfoDB database PostgreSQL service listens inside the container. Typically, the internal container port is 5432, although you may want to map a different external port.
    - **POSTGRES_USER**: The user used to access the database.
    - **POSTGRES_PASSWORD**: The password associated with the user to access the database.
    - **POSTGRES_DB**: The name of the database to which the application will connect.

2) **Configuration** variables for accessing the **"experimentalDB"** database **(optional update)**: 

    - **POSTGRES_HOST**: Specify the name of the service specified in the docker-compose.yml (in this case is 'experimental_db').
    - **POSTGRES_PORT**: Specify the port on which the PostgreSQL service of the experimentalDB database is listening in the container. 
    - **POSTGRES_USER**: The user used to access the database.
    - **POSTGRES_PASSWORD**: The password associated with the user to access the database.
    - **POSTGRES_DB**: The name of the experimental database to which the application connects.

3) **Configuration** variables both for **SSH connection to remote server**, and to specify relevant paths for **files generated when running inferences on SLURM** **(required update)**, and to manage **model paths to call to run inferences**:

    - **NEXT_PUBLIC_SSH_USER**: Assign this variable the username that will be used to make the SSH connection to the remote server.  

    - **NEXT_PUBLIC_SSH_SERVER**: Assign this variable the address of the remote server to which the connection will be made. Both the connection and all subsequent script execution requests to this server will be made using the SSH protocol.

    The values ​​of the **subsequent variables will depend on the paths set during step [3 - Setup file on the remote machine]**:

    - **NEXT_PUBLIC_PATH_FILE_INFERENCES**: Assign to this variable the base path that will represent the main directory on the remote server in which there will be both the scripts that will take care of executing the inference and the output files of the inferences requested by the users that will then be deleted once they are no longer needed. **You must make sure that in the environment referred to with this path, all the libraries necessary for the correct functioning of the scripts 'inference_8B.py' and 'inference_70B.py' have already been installed, which will be respectively invoked by 'job_script_8B.sh' and 'job_script_70B.sh'.**
          
        In particular, this last path will be used within the application in the following ways:
        
        **1)** As a base path to save inference files.  

        **2)** To specify the full path to a .sh script that will be executed by SLURM (based on the model selected by the user). 
               (for instance, `NEXT_PUBLIC_PATH_FILE_INFERENCES/job_script_${model}.sh`).  

        **3)** To build the path of the inference output file for a specific user.  
               (for instance, `NEXT_PUBLIC_PATH_FILE_INFERENCES/output_querySQL_${userId}.txt`).

        **4)** To build the paths of the output files that will be deleted.  
               (for instance, `NEXT_PUBLIC_PATH_FILE_INFERENCES/output_querySQL*` e `NEXT_PUBLIC_PATH_FILE_INFERENCES/slurm*`) 
    
    - **EXPORT_PYTHON_PATH**: Assign this variable the path to the folder on the remote machine containing the **correct version of the PyTorch library**.

    - **PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_8B_SCRIPT**: Assign this variable the path to the **8B base model** present in your account folder on the **remote server**.

    - **PATH_MODEL_USE_REMOTE_FOR_INFERENCE_8B_SCRIPT**: Assign this variable the path to the retrained **8B** model in your account folder on the **remote server**. This will be the model that will be used to handle inference when the user sends a request specifying the use of the 8B model.

    - **PATH_BASE_MODEL_REMOTE_FOR_INFERENCE_70B_SCRIPT**: Assign this variable the path to the **base model 70B** present in your account folder on the **remote server**.

    - **PATH_MODEL_USE_REMOTE_FOR_INFERENCE_70B_SCRIPT**: Assign this variable the path to the retrained **70B** model in your account folder on the **remote server**. This will be the model that will be used to handle inference when the user sends a request specifying the use of the 70B model.

4) **Configuration** variable of the file **containing your private key** **(required update)**:
    - **PRIVATE_KEY_PATH**: Assign to this variable the **path to the 'id_rsa'** file on your machine that contains your **private key** with which you will connect to the remote server via SSH.

## 5 - Run the software

**From this step onwards you will need to have Docker running on your machine.** 

- To launch the application for the first time and then build the containers with the images and start them, just position yourself inside the root of the project, open the terminal and execute this command:

```bash
docker-compose --env-file ./interface/.env.local up --build
```

- If the build has already been executed, you can always just position yourself inside the root of the project, open the terminal and execute this command:
```bash
docker-compose --env-file ./interface/.env.local up
```

- To make sure that the entire application is ready for use, you will need to **wait until only messages related to "nextjs_app" appear in the terminal**, after which you can proceed with step 6.

## 6 - Using the UI

At this point, just open any browser and enter the address "http://localhost:3000/" to be able to send requests to the application.
