#  Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License").
#  You may not use this file except in compliance with the License.
#  A copy of the License is located at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  or in the "license" file accompanying this file. This file is distributed
#  on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
#  express or implied. See the License for the specific language governing
#  permissions and limitations under the License.
from __future__ import print_function

import argparse
import json
import logging
import os
import pickle as pkl

import pandas as pd
import xgboost as xgb
from sagemaker_containers import entry_point
from sagemaker_xgboost_container import distributed
from sagemaker_xgboost_container.data_utils import get_dmatrix
from sagemaker_xgboost_container.algorithm_mode import hyperparameter_validation as hpv
from sagemaker_xgboost_container.algorithm_mode import metrics as metrics_mod
from sagemaker_xgboost_container.algorithm_mode import channel_validation as cv
from sagemaker_xgboost_container.constants import sm_env_constants
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import mlflow

def _xgb_train(params, dtrain, evals, num_boost_round, model_dir, is_master):
    """Run xgb train on arguments given with rabit initialized.

    This is our rabit execution function.

    :param args_dict: Argument dictionary used to run xgb.train().
    :param is_master: True if current node is master host in distributed training,
                        or is running single node training job.
                        Note that rabit_run will include this argument.
    """

    logging.basicConfig(level=logging.INFO) 
    
    booster = xgb.train(params=params, dtrain=dtrain, evals=evals, num_boost_round=num_boost_round)

    if is_master:
        model_location = os.path.join(model_dir, "xgboost-model")
        booster.save_model(model_location)
        logging.info(f"Stored trained model at {model_location}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Hyperparameters are described here.    
    parser.add_argument("--eta", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--min_child_weight", type=float)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--verbosity", type=int)
    parser.add_argument("--objective", type=str)
    parser.add_argument("--num_round", type=int)
    parser.add_argument("--tree_method", type=str, default="auto")
    parser.add_argument("--predictor", type=str, default="auto")
    parser.add_argument("--grow_policy", type=str, default="depthwise")
    parser.add_argument("--max_bin", type=int, default=256)
    parser.add_argument("--max_depth", type=int, default=6)
    parser.add_argument("--max_leaves", type=int, default=0)
    parser.add_argument("--early_stopping_rounds", type=int, default=5)
    parser.add_argument("--csv_weights", type=int, default=0)
    parser.add_argument("--eval_metric", type=str)

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--validation", type=str, default=os.environ.get("SM_CHANNEL_VALIDATION"))
    parser.add_argument("--sm_hosts", type=str, default=os.environ.get("SM_HOSTS"))
    parser.add_argument("--sm_current_host", type=str, default=os.environ.get("SM_CURRENT_HOST"))
    parser.add_argument("--content_type", type=str, default="csv")

    args, _ = parser.parse_known_args()
    
    logging.basicConfig(level=logging.INFO) 

    # Get SageMaker host information from runtime environment variables
    sm_hosts = json.loads(args.sm_hosts)
    sm_current_host = args.sm_current_host
        
    #Only logging from scheduler node
    if sm_current_host == sm_hosts[0]:
        ml_arn = os.environ['MLFLOW_TRACKING_ARN']
        mlflow.set_tracking_uri(ml_arn)
        mlflow.xgboost.autolog()
        mlflow.start_run()
    
    
    dtrain = get_dmatrix(args.train, args.content_type, csv_weights = args.csv_weights)
    dval = get_dmatrix(args.validation, args.content_type, csv_weights = args.csv_weights)
    watchlist = (
        [(dtrain, "train"), (dval, "validation")] if dval is not None else [(dtrain, "train")]
    )
    
    # Get the number of samples in the training dataset
    train_length = dtrain.num_row()
    # Get the number of samples in the validation dataset
    val_length = dval.num_row()

    print(f"Number of samples in training dataset: {train_length}")
    print(f"Number of samples in validation dataset: {val_length}")

    train_hp = {
        "max_depth": args.max_depth,
        "eta": args.eta,
        "gamma": args.gamma,
        "min_child_weight": args.min_child_weight,
        "subsample": args.subsample,
        "verbosity": args.verbosity,
        "objective": args.objective,
        "tree_method": args.tree_method,
        "predictor": args.predictor,
        "csv_weights": args.csv_weights,
        "eval_metric":args.eval_metric
    }
        
    if args.grow_policy == "depthwise":
        train_hp["grow_policy"] = "depthwise"
        train_hp["max_depth"] = args.max_depth
    
    if args.grow_policy == "lossguide":
        train_hp["grow_policy"] = "lossguide"
        train_hp["max_leaves"] = args.max_leaves  

    xgb_train_args = dict(
        params=train_hp,
        dtrain=dtrain,
        evals=watchlist,
        num_boost_round=args.num_round,
        model_dir=args.model_dir,
    )

    if len(sm_hosts) > 1:
        # Wait until all hosts are able to find each other
        entry_point._wait_hostname_resolution()

        # Execute training function after initializing rabit.
        distributed.rabit_run(
            exec_fun=_xgb_train,
            args=xgb_train_args,
            include_in_training=(dtrain is not None),
            hosts=sm_hosts,
            current_host=sm_current_host,
            update_rabit_args=True,
        )
    else:
        # If single node training, call training method directly.
        if dtrain:
            xgb_train_args["is_master"] = True
            _xgb_train(**xgb_train_args)
        else:
            raise ValueError("Training channel must have data to train model.")
    
    # Only logging from scheduler node
    if sm_current_host == sm_hosts[0]:
        # logging number of nodes used for training
        mlflow.log_params({"instance_size":len(sm_hosts)})
        mlflow.end_run()
