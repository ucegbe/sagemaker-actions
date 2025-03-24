import argparse
from typing import Dict

from constants import constants


def configure_parameters(args: argparse.Namespace, is_for_dask_train: bool) -> Dict[str, str]:
    """Configure parameters for single instance and distributed training.

    Args:
        args (argparse.Namespace): model arguments passed by SageMaker DLC.
        is_for_dask_train (bool): whether the parameters are used for dask distributed training o not.

    Returns:
        A parameter-value pair dictionary.
    """

    params = {
        "device_type": "cpu",  # TODO: support GPU training.
        "boosting_type": "gbdt",
        "objective": constants.PROBLEM_TYPE_OBJECTIVE,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "max_depth": args.max_depth,
        "min_data_in_leaf": args.min_data_in_leaf,
        "max_delta_step": args.max_delta_step,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "boosting": args.boosting,
        "min_gain_to_split": args.min_gain_to_split,
        "tree_learner": args.tree_learner,
        "feature_fraction_bynode": args.feature_fraction_bynode,
        "is_unbalance": args.is_unbalance == "True",
        "max_bin": args.max_bin,
        "tweedie_variance_power": args.tweedie_variance_power,
        "num_threads": args.num_threads,
        "verbosity": args.verbosity,
    }
    if is_for_dask_train:
        params.update(
            n_estimators=args.num_boost_round,
        )
    return params
