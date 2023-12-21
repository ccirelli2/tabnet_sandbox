"""
Dataset enhancements
"""
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class NewColMapper:
    """Class that creates a new column from an existing column using a mapping file."""

    def __init__(
        self, dataframe: pd.DataFrame, mapper: dict, input_col: str, output_col: str
    ):
        self.output_col = output_col
        self.input_col = input_col
        self.dataframe = dataframe
        self.mapper = mapper
        self.default = "Unknown"
        assert (
            input_col in dataframe.columns
        ), f"Input column {input_col} not in dataframe."
        logger.info(f"NewColMapper initialized successfully.")

    def _convert_none_to_nan(self):
        """Yaml file cannot represent np.nan.  Function replaces None w/ np.nan"""
        if None in self.mapper.keys():
            self.mapper.pop(None)
            self.mapper[float("nan")] = float("nan")
        return self

    def _create_new_column(self):
        logger.info(
            f"Creating new column {self.output_col} from {self.input_col} using mapping {self.mapper}"
        )
        self.dataframe[self.output_col] = self.dataframe[self.input_col].replace(
            self.mapper
        )
        return self

    def _check_mapping(self):
        if self.default in self.dataframe[self.output_col].unique():
            logger.warning(
                f"New column {self.output_col} contains {self.default} values."
            )

        if map_diff := set(self.mapper.values()) - set(
            self.dataframe[self.output_col].unique()
        ):
            logger.warning(
                f"New column {self.output_col} missing {map_diff} from mapping."
            )
        return self

    def transform(self):
        self._convert_none_to_nan()
        self._create_new_column()
        self._check_mapping()
        return self


class CreateTargetColumn:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        service_col: str,
        los_col: str,
        target_col: str = "TARGET",
        inpatient_val: str = "Inpatient",
        outpatient_val: str = "Outpatient",
    ):
        self.dataframe = dataframe
        self.service_col = service_col
        self.los_col = los_col
        self.target_col = target_col
        self.inpatient_val = inpatient_val
        self.outpatient_val = outpatient_val
        logger.info(f"CreateTargetColumn initialized successfully.")

    def transform(self):
        """Create Target Column."""
        logger.info(f"Creating target column {self.target_col}.")
        self.dataframe[self.target_col] = list(
            map(
                lambda x, y: 1
                if all([x in [self.inpatient_val, "1", "2"], y > 30])
                | (x in [self.outpatient_val] and y > 33)
                else 0,
                self.dataframe[self.service_col],
                self.dataframe[self.los_col],
            )
        )
        return self
