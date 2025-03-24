import io
import logging
import os
from typing import Any
from typing import Union

import joblib
import numpy as np
import pandas as pd
from constants import constants
from lightgbm import Booster
from lightgbm import LGBMRegressor
from sagemaker_inference import encoder


def model_fn(model_dir: str) -> Union[Booster, LGBMRegressor]:
    """Read model saved in model_dir and return a object of lightgbm model.

    Args:
        model_dir (str): directory that saves the model artifact.

    Returns:
        obj: lightgbm model.
    """
    try:
        return joblib.load(os.path.join(model_dir, "model.pkl"))
    except Exception:
        logging.exception("Failed to load model from checkpoint")
        raise


def transform_fn(task: Union[Booster, LGBMRegressor], input_data: Any, content_type: str, accept: str) -> np.array:
    """Make predictions against the model and return a serialized response.

    The function signature conforms to the SM contract.

    Args:
        task (lightgbm.Booster or lightgbm.LGBMRegressor): model loaded by model_fn.
        input_data (obj): the request data.
        content_type (str): the request content type.
        accept (str): accept header expected by the client.

    Returns:
        obj: the serialized prediction result or a tuple of the form
            (response_data, content_type)
    """
    if content_type == constants.REQUEST_CONTENT_TYPE:
        data = pd.read_csv(io.StringIO(input_data), sep=",", header=None)
        try:
            if isinstance(task, Booster):
                best_iteration = task.best_iteration
                col_names = task.feature_name()
            elif isinstance(task, LGBMRegressor):
                best_iteration = task.best_iteration_
                col_names = task.feature_name_
            data.columns = col_names
            model_output = task.predict(data, num_iteration=best_iteration)
            output = {constants.PREDICTION: model_output}
            if accept.endswith(constants.VERBOSE_EXTENSION):
                accept = accept.rstrip(constants.VERBOSE_EXTENSION)
            return encoder.encode(output, accept)
        except Exception:
            logging.exception("Failed to do transform")
            raise
    raise ValueError('{{"error": "unsupported content type {}"}}'.format(content_type or "unknown"))
