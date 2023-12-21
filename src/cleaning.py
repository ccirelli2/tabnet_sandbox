"""
Functions to clean dataframe.
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def map_val_to_null(dataframe: pd.DataFrame, column: [list, str], val: int):
    """
    Replace column value with null.

    pd_df: pandas dataframe.
    column: column name or list of column names.
    val: value to replace with null.

    return: pandas dataframe with null value.
    """
    mapping = {val: None}

    if isinstance(column, str):
        logger.info(f"Mapping value {val} to null in column {column}.")
        dataframe[column] = list(map(lambda x: mapping.get(x, x), dataframe[column]))
    elif isinstance(column, list):
        for c in column:
            logger.info(f"Value {val} mapped to null in column {c}.")
            dataframe[c] = list(map(lambda x: mapping.get(x, x), dataframe[c]))
    else:
        tp = type(column)
        raise TypeError(f"Column must be of type str or list, not {tp}")
    return dataframe


class DropNullColumns:
    def __init__(self, dataframe: pd.DataFrame, threshold: float = 0.7):
        """

        Args:
            dataframe:
            threshold:
            null_df: placeholder for dataframe containing null pct. by column.
            null_cols: placeholder for list of columns whose null pct. >= threshold.
        """
        self.dataframe = dataframe
        self.threshold = threshold
        self.cols_to_drop = []
        self.null_df = None
        self.clean_df = None
        assert isinstance(self.dataframe, pd.DataFrame)
        logger.info(f"Instantiated successfully {self.__class__}")

    def _get_column_null_pct(self):
        """

        Returns:
        """
        logger.info("Calculating null percentage per column.")
        self.null_df = self.dataframe.isna().sum() / self.dataframe.shape[0]
        self.null_df.sort_values(inplace=True, ascending=False)
        return self

    def _get_columns_to_drop(self):
        """

        Returns:
        """
        self.cols_to_drop = self.null_df[
            self.null_df.values >= self.threshold
        ].index.tolist()
        logger.info(f"Threshold found {len(self.cols_to_drop)} columns to drop.")
        return self

    def _drop_columns(self) -> pd.DataFrame:
        """
        Remove null columns from dataframe.

        Return:
            pandas dataframe with columns removed.
        """
        logger.info(f"Dropping columns => {self.cols_to_drop}")
        self.clean_df = self.dataframe[
            [c for c in self.dataframe.columns if c not in self.cols_to_drop]
        ]
        logger.info(f"Columns dropped: {self.cols_to_drop}")
        return self

    def clean(self):
        """

        Returns:

        """
        self._get_column_null_pct()
        self._get_columns_to_drop()
        self._drop_columns()
        return self.clean_df
