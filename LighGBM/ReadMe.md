# LightGBM Training

This Repository goes over SageMAker Training, Hyperparameter Tuning and SageMaker Pipelines

There are two notebooks for Training:
1. **sagemaker-lightgbm-distributed-training-dask-Classification**: Walks through using LighGBM on SageMaker for distributed training and hyperparameter tuning for ML classification usecase
2. **sagemaker-lightgbm-distributed-training-dask-Regression**: Walks through using LightGBM on SageMaker for distributed Regression model training and hyperparameter tuning

There are two notebooks for SageMkaer Pipelines:
1. **sagemaker-pipelines-train-pipeline-arch-1**: Walks through using SageMaker pipeline for processing and training a model. This leverages SaegMaker Pipelines parameterization to create a robust pipeline where parameters for model tuning can be changed during execution. In this approach a pipeline is created for Classification or Regression and the user can choose what branch of teh pipeline to excute using parameters. It standardizes the pipeline for tuning a model and allows the user to pass the location of training data in s3.

1. **sagemaker-pipelines-train-pipeline-arch-2**: Walks through using SageMaker pipeline for processing and training a model. This leverages SaegMaker Pipelines parameterization to create a robust pipeline where parameters for model tuning can be changed during execution. In this approach there is more flexibility. Here the user can pass the S3 location of a training logic to be used in the model tuning job. This gives more flexibility to the user on how the training job should be executed.

**model_cat** contains all the dependencies and training logic for the Classification model training

**model_reg** contains all the dependencies and training logic for Regression model training


