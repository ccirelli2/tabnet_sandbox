"""
Utility functions

# TODO: Add log handler for both stdout and write to log file.
"""
import os
import git
import yaml
import pandas as pd
import logging
from time import time
from functools import wraps

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def timeit(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        logger.info(f"func:{f.__name__,} took: {round(te - ts, 4)} seconds")
        return result

    return wrap


def apply_threshold(prediction: list, threshold: float = 0.5) -> list:
    """
    Apply threshold to predictions.
    """
    return [1 if y >= threshold else 0 for y in prediction]


@timeit
def load_config(directory: str = None, filename: str = "config.yaml"):
    """
    Load configuration from a YAML file.
    """
    directory = (
        directory
        if directory
        else git.Repo(".", search_parent_directories=True).working_tree_dir
    )
    filename = filename if filename else "config.yaml"
    assert os.path.isfile(os.path.join(directory, filename)), f"Not a file: {filename}"
    path = os.path.join(directory, filename)

    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    logger.info(f"Config file loaded successfully")
    return config


@timeit
def write_dataframe(
    pd_df: pd.DataFrame,
    directory: str,
    filename: str,
    extension: str = "csv",
    index: bool = False,
):
    """
    Write dataframe to CSV file.
    """
    logger.info(
        f"Writing dataframe w/ dim {pd_df.shape} to {directory} as {filename}.{extension}"
    )
    if "." in filename:
        filename = filename.split(".")[0]
    path = os.path.join(directory, f"{filename}.{extension}")
    pd_df.to_csv(path, index=index)
    logger.info(f"Dataframe written to {path}\n")
    return None


@timeit
def load_dataframe(
    directory: str,
    filename: str,
    extension: str = "csv",
    sep: str = ",",
    sample: bool = False,
    nrows: int = 100,
):
    """
    Load dataframe from CSV file.
    TODO: For better unit testing convert this to a class object.
    """
    # Clean up
    if "." in filename:
        filename = filename.split(".")[0]
    if "." in extension:
        extension = extension.split(".")[1]
    # Define Path
    path = os.path.join(directory, f"{filename}.{extension}")
    # Load DataFrame
    if sample:
        logger.info(f"Loading dataframe with sample of {nrows} rows")
        pd_df = pd.read_csv(path, nrows=nrows, sep=sep)
    else:
        logger.info(f"Loading dataframe")
        pd_df = pd.read_csv(path, sep=sep)
    logger.info(
        f"Dataframe  with dimensions {pd_df.shape} loaded successfully from {path}\n"
    )
    return pd_df


def get_cnt_feature_target(
    dataframe: pd.DataFrame, feature_col: str, target_col: str
) -> pd.DataFrame:
    """
    Args:
        target_col:
        dataframe:
        feature_col:

    """
    logger.info(
        f"Calculating count of observations by feature {feature_col} and target"
    )

    FRAMES = []

    for t in dataframe[target_col].unique():
        groupby_df = (
            dataframe[dataframe[target_col] == t]
            .groupby(feature_col)[feature_col]
            .count()
        )

        frame = pd.DataFrame(
            {
                "TARGET": [t for t in range(groupby_df.shape[0])],
                "FEATURE": [feature_col for f in range(groupby_df.shape[0])],
                "VALUE": groupby_df.index,
                "COUNT": groupby_df.values,
                "PCT": groupby_df.values / dataframe.shape[0],
            }
        )
        FRAMES.append(frame)

    return pd.concat(FRAMES)


class Logger:
    def __init__(self, directory: str, filename: str):
        """ """
        self.directory = directory
        self.filename = f"{filename}.logs"
        self.log_path = os.path.join(directory, self.filename)
        print("Logger Initialized successfully")

    def _get_root_logger(self):
        """ """
        self.root_logger = logging.getLogger()
        return self

    def _get_log_formatter(self):
        """ """
        self.log_formatter = logging.Formatter(
            "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
        )
        return self

    def _get_file_handler(self):
        """ """
        self.file_handler = logging.FileHandler(self.log_path)
        self.file_handler.setFormatter(self.log_formatter)
        return self

    def _add_handlers(self):
        """ """
        self.root_logger.addHandler(self.file_handler)
        return self

    def get_logger(self):
        """ """
        logging.basicConfig(level=logging.INFO)
        self._get_root_logger()
        self._get_log_formatter()
        self._get_file_handler()
        self._add_handlers()
        return self.root_logger

    def log_dict(self, obj: dict, level: str = "INFO"):
        pass
