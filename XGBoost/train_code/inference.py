import os
import json
import pickle
import xgboost as xgb
import numpy as np
import io
import pandas as pd

def model_fn(model_dir):
    """
    Load the XGBoost model from the specified directory.

    Args:
        model_dir (str): The directory where the model file is saved.

    Returns:
        xgb.Booster: The loaded XGBoost model.
    """
    model_file = os.path.join(model_dir, "xgboost-model")
    model = xgb.Booster()
    model.load_model(model_file)
    return model



def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the prediction input.

    Args:
        request_body (str or bytes): The request payload.
        request_content_type (str): The content type of the request.

    Returns:
        numpy.ndarray: The input data as a numpy array.
    """
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        return np.array(input_data)

    elif request_content_type == "text/csv":
        # Read the csv file into a pandas dataframe
        input_data = pd.read_csv(io.StringIO(request_body),header=None)
        return input_data.values

    elif request_content_type == "application/octet-stream":
        # Assume the bytes contain a CSV file
        input_data = pd.read_csv(io.BytesIO(request_body),header=None)
        return input_data.values

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")
def predict_fn(input_data, model):
    """
    Make predictions using the XGBoost model.

    Args:
        input_data (numpy.ndarray): The input data.
        model (xgb.Booster): The XGBoost model.

    Returns:
        numpy.ndarray: The model's predictions.
    """
    dmatrix = xgb.DMatrix(input_data)
    return model.predict(dmatrix)
