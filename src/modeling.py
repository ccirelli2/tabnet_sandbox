"""

Hyperopt
- Tutorial: https://www.analyticsvidhya.com/blog/2020/09/alternative-hyperparameter-optimization-technique-you-need-to-know-hyperopt/



"""
import os
import logging
import pandas as pd
import numpy as np
import lightgbm as lgb
import matplotlib.pyplot as plt
import gc
from hyperopt import hp, tpe, Trials, STATUS_OK
from hyperopt.fmin import fmin
from hyperopt.pyll.stochastic import sample

from src.base import ModelingBase

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")


def quick_hyperopt(
    train: lgb.Dataset,
    space_config: dict,
    objective_config: dict,
    num_evals: int = 5,
    num_cv_folds: int = 5,
    early_stopping_rounds: int = 100,
    stratified: bool = False,
    diagnostic: bool = True,
    integer_params: list = [
        "max_depth",
        "num_leaves",
        "max_bin",
        "min_data_in_leaf",
        "min_data_in_bin",
    ],
):
    """
    Function to run hyperopt with LightGBM, obtain logs and return best parameters.

    Parameters
    ::train:: (lgb.Dataset): Training dataset
    ::space_config:: (dict): Hyperopt search space
    ::objective_config:: (dict): Objective function
    ::num_evals:: (int): Number of evaluations

    References:
    - fmin: https://docs.databricks.com/en/machine-learning/automl-hyperparam-tuning/hyperopt-concepts.html#fmin
    - lgb.cv: https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.cv.html
    """
    # clear space
    gc.collect()

    # Define Objective Function (What Hyperopt will minimize)
    def objective(param_space):
        """
        Objective function to optimize.
        """
        # Cast integer params from float to int
        for parameter in integer_params:
            param_space[parameter] = int(param_space[parameter])

        # TODO: Check Hyperopt docs for param nesting.
        # Choose Booster
        subsample = param_space["boosting"].get("subsample", 1.0)
        # Update our parameters
        param_space["boosting"] = param_space["boosting"]["boosting"]
        param_space["subsample"] = subsample
        logger.info(f"Boosting type: {param_space['boosting']}")
        logger.info(f"Boosting Type sub-sample: {subsample}")

        # Cross Validation
        cv_results = lgb.cv(
            params=param_space,
            train_set=train,
            nfold=num_cv_folds,
            stratified=stratified,
            num_boost_round=early_stopping_rounds,
        )

        # Return Best Loss
        loss_metric = f"valid {param_space['metric']}-mean"
        scores = cv_results[loss_metric]
        best_loss = scores[np.array(scores).argmin()]
        logger.info(f"\t\t Best Loss => {best_loss}\n\n\n")
        return {"loss": best_loss, "status": STATUS_OK}

    # Execute Hyperopt Run
    """
    We utilize hp.fmin to execute the hyperopt run, which attempts to minimize out objective function.
    ::fmin: minimization algorithm.
    ::space: defines the hyperparameter space to search
    ::algo: defines the search algorithm to use
    ::max_evals: defines the maximum number of hyperparameter settings to try.
    """
    # Trials Object Tracks Iterations
    trials = Trials()

    # Minimize Objective Function
    best = fmin(
        fn=objective,
        space=space_config,
        algo=tpe.suggest,
        max_evals=num_evals,
        trials=trials,
    )

    # fmin() will return the index of values chosen from the lists/arrays in 'space'
    # to obtain actual values, index values are used to subset the original lists/arrays
    best["boosting"] = objective_config["boosting_list"][best["boosting"]][
        "boosting"
    ]  # nested dict, indexed twice
    best["metric"] = objective_config["metric_list"][best["metric"]]
    best["objective"] = objective_config["objective_list"][best["objective"]]

    # cast floats of integer params to int
    for param in integer_params:
        best[param] = int(best[param])

    print("{" + "\n".join("{}: {}".format(k, v) for k, v in best.items()) + "}")

    if diagnostic:
        return best, trials
    else:
        return best


class LgbmModelingPrepare(ModelingBase):
    """

    Methods - Inherited
    --------------------------------
    __init__: Initialize class
    _get_sample:
    _limit_columns:
    _get_null_pct_by_feature:
    _nan_fill:
    _split_x_y:
    _get_train_test_split:
    _save_results:

    Methods - New
    --------------------------------

    TODO:
        Monotonic columns
        Feature set inspection by data type.


    """

    def __init__(
        self,
        data: pd.DataFrame,
        output_dir: str,
        feature_column_names: list,
        target_column_name: str = "TARGET",
        pkey_column_name: str = "CASEID",
        column_type_config: dict = {},
        sample: bool = True,
        num_rows: int = 10_000,
        nan_fill: bool = True,
        nan_val: int = -9,
        test_pct: float = 0.33,
        save_results: bool = False,
        include_index: bool = True,
    ):
        self.data = data
        self.output_dir = output_dir
        self.feature_column_names = feature_column_names
        self.target_column_name = target_column_name
        self.pkey_column_name = pkey_column_name
        self.column_type_config = column_type_config
        self.sample = sample
        self.num_rows = num_rows
        self.nan_fill = nan_fill
        self.nan_val = nan_val
        self.test_pct = test_pct
        self.save_results = save_results
        self.include_index = include_index
        self.lgb_train = None
        self.lgb_test = None

        ModelingBase.__init__(
            self,
            data=data,
            output_dir=output_dir,
            feature_column_names=feature_column_names,
            target_column_name=target_column_name,
            pkey_column_name=pkey_column_name,
            sample=sample,
            num_rows=num_rows,
            nan_fill=nan_fill,
            nan_val=nan_val,
            test_pct=test_pct,
            save_results=save_results,
            include_index=include_index,
            X_data=pd.DataFrame(),  # noqa
            y_data=pd.DataFrame(),  # noqa
            X_train=pd.DataFrame(),  # noqa
            X_test=pd.DataFrame(),  # noqa
            y_train=pd.DataFrame(),  # noqa
            y_test=pd.DataFrame(),  # noqa
        )

        logger.info(f"Class {self.__class__} instantiated successfully")

    def _convert_data_objects_to_arrays(self):
        logger.info(
            f"Converting dataframe objects X_train, X_test, y_train, y_test to arrays"
        )
        self.X_train_array = self.X_train.values
        self.X_test_array = self.X_test.values
        self.y_train_array = self.y_train.values
        self.y_test_array = self.y_test.values
        logger.info("Dataframe objects converted to arrays successfully")
        return self

    def _create_lgb_train_dataset(self):
        logger.info(f"Creating LGBM Train dataset")
        self.lgb_train = lgb.Dataset(
            self.X_train_array,
            self.y_train_array,
            feature_name=self.X_feature_names,
            # categorical_feature=self.column_type_config['categorical'],
            free_raw_data=False,
        )
        return self

    def _create_lgb_test_dataset(self):
        logger.info(f"Creating LGBM Test dataset")
        self.lgb_test = lgb.Dataset(
            self.X_test_array,
            self.y_test_array,
            feature_name=self.X_feature_names,
            # categorical_feature=self.column_type_config['categorical'],
            free_raw_data=False,
        )
        return self


def line_plot_two_axis(
    x: list,
    y1: list,
    y2: list,
    y1_label: str,
    y2_label: str,
    title: str,
    dir_output: str,
    file_name: str,
    savefig: bool,
):
    # Plot Area
    fig, ax1 = plt.subplots()

    color = "tab:red"
    ax1.set_xlabel("Trial Number")
    ax1.set_ylabel(y1_label, color=color)
    ax1.plot(x, y1, color=color)
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.set_ylim(0)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_xlabel("Trial Number")
    ax2.set_ylabel(y2_label, color=color)
    ax2.plot(x, y2, color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    ax2.set_ylim(0)

    # plt.legend([y1_label, y2_label])
    ax1.grid(which="both")
    ax1.legend([y1_label, y2_label], loc="upper right")
    plt.title(title)
    if savefig:
        plt.savefig(os.path.join(dir_output, f"{file_name}.png"))
        plt.close()
    return None
