"""

"""
import pandas as pd
import logging

# Settings
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class DataDescriber:
    def __init__(self, pd_df: pd.DataFrame, sample_num: int = 3, precision: int = 4):
        self.pd_df = pd_df
        self.sample_num = sample_num
        self.describe_df = pd.DataFrame({}, index=self.pd_df.columns)
        self.precision = precision
        logger.info("Data Describer Instantiated Successfully.")
        logger.info(f"Dataset dimensions (rows, columns): {self.pd_df.shape}")

    def _get_schema(self):
        self.schema_df = pd.DataFrame(self.pd_df.dtypes).rename(columns={0: "SCHEMA"})
        logger.info("Generating schema description")
        return self

    def _get_new_idx(self):
        new_idx = {}
        idx = self.sample_df.index
        for i in range(len(idx)):
            new_idx[idx[i]] = f"SAMPLE_{i}"
        return new_idx

    def _get_sample(self):
        self.sample_df = self.pd_df.sample(n=self.sample_num)
        self.sample_df = self.sample_df.rename(index=self._get_new_idx()).transpose()
        logger.info("Generating data sample")
        return self

    def _get_levels(self):
        """
        Returns number of unique levels per column.
        infers primary key where num_levels == num_rows.
        """
        levels = {c: len(self.pd_df[c].unique()) for c in self.pd_df.columns}
        self.levels_df = pd.DataFrame(levels, index=["LEVELS"]).transpose()
        self.levels_df["PRIMARY_KEY"] = list(
            map(
                lambda x: True if x == self.pd_df.shape[0] else False,
                self.levels_df.LEVELS,
            )
        )
        logger.info("Generating data levels description")
        return self

    def _get_null_pct(self):
        self.null_df = pd.DataFrame((self.pd_df.isna().sum())).rename(
            columns={0: "NULL_COUNT"}
        )
        self.null_df["NULL_PCT"] = self.null_df.NULL_COUNT / self.pd_df.shape[0]
        return self

    def _get_stats(self):
        self.stats_df = self.pd_df.describe().transpose()
        self.stats_df = self.stats_df.rename(
            columns={c: c.upper() for c in self.stats_df.columns}
        ).round(self.precision)
        self.stats_df["COUNT"] = self.stats_df["COUNT"].apply(int)
        logger.info("Generating data statistics description")
        return self

    def describe(self):
        self._get_sample()
        self._get_schema()
        self._get_levels()
        self._get_null_pct()
        self._get_stats()

        descriptions = [
            self.sample_df,
            self.schema_df,
            self.levels_df,
            self.null_df,
            self.stats_df,
        ]

        for descr in descriptions:
            self.describe_df = self.describe_df.merge(
                descr, left_index=True, right_index=True, how="left"
            )
        self.describe_df = self.describe_df.reset_index().rename(
            columns={"index": "COLUMN"}
        )
        logger.info("Returning data description")
        return self.describe_df
