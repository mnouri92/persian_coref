# Introduction
**Pronouns references recognition**, is one of the NLP tasks with goal of detect the references of each pronouns which have been used in text. This project with inspiration of **e2e-coref** detection, is developed to detect the antecedents in persian texts.

# Usage
You can clone and run the code of this project by follow this document step by step.

## step-1
in first step you must clone the project on your system as follow :

```git clone https://github.com/mnouri92/persian_coref.git```
> if you dont have the git on your system you can download and install that from [Git](https://git-scm.com/downloads)

## Step-2
The project is developed by **Python** programming language and using some important packages related to Machine-Learning programming such as TensorFlow. 
> if you haven't installed Python on your system you can download and install from [Python.org](https://www.python.org/downloads/)

Therefore after cloning the project, move to project's root directory and install requirements by :

``` python -m pip install -r requirements.txt```

## step-3
For train the model you need some file such as Persian word embedding, train and test data and convert the data to desirable format which is required to train the model that all of these files will be downloaded and convert by running the pre_process.sh file. You can do this by run the bellow command in terminal :

``` ./pre_process.sh ```

## step-4
After that all of required files have been provided you can train the model by run the bellow command in terminal :

``` python train.py best ```

> In above command the 'best' keyword refer to configuration details that have been defined for model. the 'best' configuration set is defined for train step and the 'final' configuration set is defined for evaluate and prediction. you can change them from **experiments.conf** file.

After train the model you can evaluate it by running the **evaluate.sh** file using bellow command in terminal :

``` ./evaluate.sh ```

## step-5

After running the codes , the trained model will be saved in ./logs/final directory. The trained model that saved on .logs/final directory is in checkpoints format that you will need SavedModel format of that to deploy as web API using TensorFlow Serving module. The export_saved_model.sh file has been provided for this reason that you can use that as follow to export the SavedModel format of checkpoints saved model.

``` python export_saved_model.py final <export_path> <version_of_model> ```


## step-6

Now the SavedModel format of trained model is ready in your chosen directory.

**TensorFlow Serving** is one of the useful module from TFX that is used to serve the SavedModel as a web API by implements the TF Server on the host system and running the web API services on it. you can use this module directly but simpler way is using TensorFlow/Serving image of **Docker** that let you to implement the TensorFlow server as an **Docker Container** on the host system.

> If you haven't installed Docker in your system you can download and install it from [Docker](https://www.docker.com/)

You can run the web API service by trained model as Docker Container using the fhe following command terminal :

``` docker run -p 8501:8501 -p 8500:8500 --name <Container_name> --mount type=bind,source=<path_of_SavedModel>,target=/models/<model_name> -e MODEL_NAME=<model_name> -t tensorflow/serving & ```

The above code will implement the TF Server as a Docker container on the host system and then run the web API service on TF Server. Finally map the port of container to the system's port (the 8501 port are used to communicate with the web API by RestFul standard and 8500 are used to communicate by gRPC standard).

After running the above code the implemented web API can be accessed via the **http://localhost:8501/v1/models/model_name:predict**
