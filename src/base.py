"""
Base class object for feature importance and modeling.
"""

import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from src import utils

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")


class ModelingBase:
    def __init__(
        self,
        data: pd.DataFrame,
        output_dir: str,
        feature_column_names: list,
        sample: bool = False,
        num_rows: int = 10_000,
        nan_fill: bool = False,
        nan_val: int = -9,
        test_pct: float = 0.33,
        target_column_name: str = "TARGET",
        pkey_column_name: str = "CASEID",  # noqa
        save_results: bool = False,
        include_index: bool = False,
        X_data=pd.DataFrame(),  # noqa
        y_data=pd.DataFrame(),  # noqa
        X_train=pd.DataFrame(),  # noqa
        X_test=pd.DataFrame(),  # noqa
        y_train=pd.DataFrame(),  # noqa
        y_test=pd.DataFrame(),  # noqa
    ):
        """
        data: dataframe containing training data.
        output_dir: output directory where to write results.
        sample: whether tot ake a sample of data attribute.
        num_rows: number of sample rows to take.
        nan_fill: function to fill any na values w/ a constant number.
        nan_val: value to impute to nan values.
        target_column_name: if there exists a target column within the dataframe, this is the default name.
        pkey_column_name: name of primary key column.
        feature_column_names: if there are any columns to be dropped from input dataframe.
        save_results: whether to save results to the desired output directory.
        eda: tuple of form (feature_names, feature_values)
        """
        self.data = data
        self.output_dir = output_dir
        self.feature_column_names = feature_column_names
        self.sample = sample
        self.num_rows = num_rows
        self.nan_fill = nan_fill
        self.nan_val = nan_val
        self.test_pct = test_pct
        self.target_column_name = target_column_name
        self.pkey_column_name = pkey_column_name  # noqa
        self.save_results = save_results
        self.include_index = include_index
        self.X_data = X_data
        self.y_data = y_data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def _get_sample(self):
        if self.sample:
            logger.info(f"Down-sampling data to {self.num_rows} rows")
            self.data = self.data.sample(self.num_rows)
        return self

    def _limit_columns(self):
        logger.info(f"Limiting feature columns to {self.feature_column_names}")
        self.data = self.data[self.feature_column_names]
        logger.info(f"Post transformation dimensions => {self.data.shape}")
        return self

    def _drop_pkey_column(self):
        logger.info(f"Dropping primary key column {self.pkey_column_name}")
        self.data = self.data.drop(self.pkey_column_name, axis=1)
        assert (
            self.pkey_column_name not in self.data.columns
        ), "Primary key column still exists"
        logger.info(f"Post transformation dimensions => {self.data.shape}")
        return self

    def _get_null_pct_by_feature(self):
        logger.info("Calculating null percentage by feature")
        self.null_df = self.data.isna().sum() / self.data.shape[0]
        return self

    def _nan_fill(self):
        if self.nan_fill:
            logger.info(f"Filling NaN values with {self.nan_val}")
            self.data = self.data.fillna(value=self.nan_val)
            assert self.data.isna().sum().sum() == 0, "NaN values still exist"
        return self

    def _split_x_y(self):
        logger.info("Generating train test splits")
        self.y_data = self.data[self.target_column_name]
        self.X_data = self.data.drop(self.target_column_name, axis=1)
        del self.data  # garbage collection
        logger.info(
            f"X dimension => {self.X_data.shape}, y dimensions => {self.y_data.shape}"
        )
        return self

    def _capture_feature_names(self):
        logger.info("Capturing feature names")
        self.X_feature_names = self.X_data.columns
        return self

    def _get_train_test_split(self):
        # Create Train Test Split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_data, self.y_data, test_size=self.test_pct, random_state=123
        )
        del self.X_data  # garbage collection
        del self.y_data  # garbage collection
        logger.info(
            f"X_train dimensions => {self.X_train.shape}, y_train dimensions => {self.y_train.shape}"
        )
        return self

    def _save_results(self, data: pd.DataFrame, filename: str):
        if self.save_results:
            utils.write_dataframe(
                pd_df=data,
                directory=self.output_dir,
                filename=filename,
                index=self.include_index,
            )
