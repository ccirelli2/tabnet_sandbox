"""
Class objects to run feature importance

TODO: Ranking column needs to start from the feature with the least importance value after 0
  We can achieve this by separating the list of features with a value of 0 and those that don't.
  Then create the rank adn join.

"""
import pandas as pd
import logging
from src import utils
from src.utils import timeit
from src.base import ModelingBase
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import chi2
import xgboost as xgb

# Settings
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)
logger = logging.getLogger(__name__)
logger.setLevel(level="INFO")


class FeatureImportanceChi2(ModelingBase):
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
        save_results: bool = False,
        include_index: bool = False,
    ):
        self.data = (data,)
        self.output_dir = (output_dir,)
        self.feature_column_names = (feature_column_names,)
        self.sample = (sample,)
        self.num_rows = (num_rows,)
        self.nan_fill = (nan_fill,)
        self.target_column_name = (target_column_name,)
        self.save_results = (save_results,)
        self.include_index = include_index

        ModelingBase.__init__(
            self,
            data=data,
            output_dir=output_dir,
            feature_column_names=feature_column_names,
            sample=sample,
            num_rows=num_rows,
            nan_fill=nan_fill,
            nan_val=nan_val,
            test_pct=test_pct,
            target_column_name=target_column_name,
            save_results=save_results,
            include_index=include_index,
        )
        self.feature_importance = {"FEATURE": [], "STATISTIC": [], "P_VALUE": []}
        self.feature_importance_dataframe = pd.DataFrame({})
        logger.info("Chi2 feature importance class instantiated successfully.")

    def _get_feature_importance(self):
        """ """
        logger.info(f"Executing Chi2 test for feature set")
        # Iterate feature columns less target column.
        for c in [c for c in self.data.columns if c not in [self.target_column_name]]:
            # Drop Nan Values for column C + Target
            clean_df = self.data[[c, self.target_column_name]].dropna()
            # Separate X & y data
            y_data = clean_df[self.target_column_name]
            X_data = clean_df.drop(self.target_column_name, axis=1)
            # Execute Chi2 Test
            chi2_test = chi2(X_data, y_data)
            # Save Results
            self.feature_importance["FEATURE"].append(c)
            self.feature_importance["STATISTIC"].append(chi2_test[0][0])
            self.feature_importance["P_VALUE"].append(chi2_test[1][0])
        logger.info("Test finished.")
        return self

    def _build_feature_importance_dataframe(self):
        """
        Create dataframe from chi2 test results.
        """
        logger.info("Building feature importance dataframe")
        self.feature_importance_dataframe = pd.DataFrame(
            {
                "FEATURE": self.feature_importance["FEATURE"],
                "STATISTIC": self.feature_importance["STATISTIC"],
                "P_VALUE": self.feature_importance["P_VALUE"],
            }
        )
        return self

    def _is_feature_significant(self):
        """
        Create a binary flag that indicates if a feature is significant or not.
        """
        logger.info("Creating feature significance flag")
        self.feature_importance_dataframe["IS_SIGNIFICANT"] = list(
            map(
                lambda x: 1 if x < 0.05 else 0,
                self.feature_importance_dataframe["P_VALUE"],
            )
        )
        return self

    def get_importance(self):
        logger.info("Executing feature importance class")
        self._get_sample()
        self._limit_columns()
        self._get_null_pct_by_feature()
        self._nan_fill()
        self._get_feature_importance()
        self._build_feature_importance_dataframe()
        self._is_feature_significant()
        self._save_results(data=self.null_df, filename="feature_set_pct_null")
        self._save_results(
            data=self.feature_importance_dataframe, filename="feature_importance"
        )
        return self


class FeatureImportanceModels(ModelingBase):
    def __init__(
        self,
        data: pd.DataFrame,
        output_dir: str,
        feature_column_names: list,
        sample: bool = True,
        num_rows: int = 10_000,
        nan_fill: bool = True,
        nan_val: int = -9,
        test_pct: float = 0.33,
        target_column_name: str = "TARGET",
        save_results: bool = False,
        include_index: bool = True,
    ):
        """ """
        self.data = (data,)
        self.output_dir = (output_dir,)
        self.feature_column_names = (feature_column_names,)
        self.sample = (sample,)
        self.num_rows = (num_rows,)
        self.nan_fill = (nan_fill,)
        self.nan_val = (nan_val,)
        self.test_pct = (test_pct,)
        self.target_column_name = (target_column_name,)
        self.save_results = (save_results,)
        self.include_index = include_index

        ModelingBase.__init__(
            self,
            data=data,
            output_dir=output_dir,
            feature_column_names=feature_column_names,
            sample=sample,
            num_rows=num_rows,
            nan_fill=nan_fill,
            nan_val=nan_val,
            test_pct=test_pct,
            target_column_name=target_column_name,
            save_results=save_results,
            include_index=include_index,
        )
        self.feature_importance_dataframe = pd.DataFrame({})
        self.feature_importance = {
            "DECISION_TREE": (list, list),
            "LOGISTIC_REGRESSION": (list, list),
            "RANDOM_FOREST": (list, list),
            "XG_BOOST": (list, list),
        }
        assert self.nan_fill, "NaN fill must be set to True."
        logger.info("Model Feature Selection Class Instantiated Successfully.")

    def _create_base_dataframe(self):
        """Create shell dataframe whose index is the list of columns in X-train."""
        logger.info("Creating base feature importance dataframe")
        self.feature_importance_dataframe = pd.DataFrame(index=self.X_train.columns)
        return self

    @timeit
    def _decision_tree(self):
        logger.info("\nTraining Decision Tree.")
        clf = DecisionTreeClassifier()
        clf.fit(self.X_train, self.y_train)
        self.feature_importance["DECISION_TREE"] = (
            self.X_train.columns.tolist(),
            clf.feature_importances_,
        )
        return self

    @timeit
    def _random_forest(self):
        logger.info("\nTraining Random Forest.")
        clf = RandomForestClassifier()
        clf.fit(self.X_train, self.y_train)
        self.feature_importance["RANDOM_FOREST"] = (
            self.X_train.columns.tolist(),
            clf.feature_importances_,
        )
        return self

    @timeit
    def _logistic_regression(self):
        logger.info("\nTraining Logistic Regression.")
        clf = LogisticRegression(penalty="l1", solver="liblinear", max_iter=500)
        clf.fit(self.X_train, self.y_train)
        self.feature_importance["LOGISTIC_REGRESSION"] = (
            self.X_train.columns,
            clf.coef_[0],
        )
        return self

    @timeit
    def _xg_boost(self):
        logger.info("\nTraining XG Boost.")
        clf = xgb.XGBClassifier()
        clf.fit(self.X_train, self.y_train)
        self.feature_importance["XG_BOOST"] = (
            self.X_train.columns,
            clf.feature_importances_,
        )
        return self

    @timeit
    def _build_feature_importance_dataframe(self):
        """
        Create feature importance dataframe by join each models's feature importance results to the base dataframe.
        """
        logger.info("Building feature importance dataframe")

        for model_name in self.feature_importance.keys():
            if not self.feature_importance.get(model_name, None):
                logger.warning(
                    f"Feature importance has not been generated for {model_name}"
                )
            else:
                features = self.feature_importance[model_name][0]
                importance = self.feature_importance[model_name][1]
                importance_df = pd.DataFrame(
                    {f"{model_name}_IMPORTANCE": importance}, index=features
                )
                logger.info(f"Joining models results for {model_name}")
                self.feature_importance_dataframe = (
                    self.feature_importance_dataframe.merge(
                        right=importance_df,
                        left_index=True,
                        right_index=True,
                        how="left",
                    )
                )
        return self

    def _is_feature_significant(self):
        """
        Create a binary flag that indicates if a feature is significant or not.
        """
        logger.info("Creating feature significance flag")
        for model_name in self.feature_importance.keys():
            # Calculate Median Importance Value
            model_median_val = self.feature_importance_dataframe[
                f"{model_name}_IMPORTANCE"
            ].median()
            logger.info(f"Median value for {model_name} is {model_median_val}")

            # Create Significance Flag
            self.feature_importance_dataframe[f"{model_name}_IS_SIGNIFICANT"] = list(
                map(
                    lambda x: 1 if x >= model_median_val else 0,
                    self.feature_importance_dataframe[f"{model_name}_IMPORTANCE"],
                )
            )
        return self

    def _total_votes(self):
        """
        Calculate total significance votes.
        """
        significance_columns = [
            c
            for c in self.feature_importance_dataframe.columns
            if "IS_SIGNIFICANT" in c
        ]
        self.feature_importance_dataframe["TOTAL_VOTES"] = (
            self.feature_importance_dataframe[significance_columns].sum(axis=1).values
        )
        return self

    def _majority_vote(self):
        """
        Function to create a majority vote column.
        If N-1 models agree a feature is important then 1 else 0.
        """
        self.feature_importance_dataframe["IS_MAJORITY"] = list(
            map(
                lambda x: 1 if x >= (len(self.feature_importance.keys()) - 1) else 0,
                self.feature_importance_dataframe["TOTAL_VOTES"],
            )
        )
        return self

    def _add_feature_column(self):
        """
        Add feature column to dataframe.
        """
        logger.info("Adding feature column to dataframe")
        self.feature_importance_dataframe[
            "FEATURE"
        ] = self.feature_importance_dataframe.index
        return self

    def get_importance(self):
        logger.info("Executing feature importance class")
        self._get_sample()
        self._limit_columns()
        self._get_null_pct_by_feature()
        self._nan_fill()
        self._split_x_y()
        self._get_train_test_split()
        self._create_base_dataframe()
        self._decision_tree()
        self._random_forest()
        self._logistic_regression()
        self._xg_boost()
        self._build_feature_importance_dataframe()
        self._is_feature_significant()
        self._total_votes()
        self._majority_vote()
        self._add_feature_column()
        self._save_results(data=self.null_df, filename="feature_set_pct_null")
        self._save_results(
            data=self.feature_importance_dataframe, filename="feature_importance"
        )
        return self


class MergeImportance:
    """
    This function is deprecated.
    """

    def __init__(
        self,
        chi2_df: pd.DataFrame,
        model_df: pd.DataFrame,
        output_dir: str,
        merge_column_name: str = "FEATURE",
        model_names: list = [
            "CHI2",
            "DECISION_TREE",
            "LOGISTIC_REGRESSION",
            "RANDOM_FOREST",
            "XG_BOOST",
        ],
        chi2_column_mapping: dict = {
            "STATISTIC": "CHI2_IMPORTANCE",
            "RANK": "CHI2_IMPORTANCE_RANK",
        },
        save_results: bool = False,
        include_index: bool = False,
    ):
        self.chi2_df = chi2_df
        self.model_df = model_df
        self.output_dir = output_dir
        self.merge_column_name = merge_column_name
        self.model_names = model_names
        self.chi2_column_mapping = chi2_column_mapping
        self.save_results = save_results
        self.include_index = include_index
        self.importance_column_names = [f"{n}_IMPORTANCE" for n in self.model_names]
        self.rank_column_names = [f"{n}_IMPORTANCE_RANK" for n in self.model_names]
        self.merge_df = pd.DataFrame({})
        self.column_order = (
            [self.merge_column_name]
            + self.importance_column_names
            + self.rank_column_names
        )
        logger.info(f"Class object {__class__} instantiated successfully")

    def _rename_columns(self):
        logger.info("Renaming chi2 columns")
        self.chi2_df = self.chi2_df.rename(columns=self.chi2_column_mapping)
        return self

    def _limit_columns(self):
        logger.info("Limit chi2 columns")
        self.chi2_df = self.chi2_df[
            [self.merge_column_name, "CHI2_IMPORTANCE", "CHI2_IMPORTANCE_RANK"]
        ]
        return self

    def _merge_dataframes(self):
        logger.info("Merging chi2 and models dataframes")
        self.merge_df = pd.merge(
            left=self.model_df,
            right=self.chi2_df,
            how="left",
            left_on=self.merge_column_name,
            right_on=self.merge_column_name,
        )
        return self

    def _fill_na(self):
        """Left join of chi2 will create NaN values in importance and rank columns"""
        logger.info("Filling merged dataframe NaN values with 0")
        self.merge_df = self.merge_df.fillna(0)
        return self

    def _reorder_columns(self):
        logger.info("Reordering merged dataframe columns")
        self.merge_df = self.merge_df[self.column_order]
        return self

    def _calculate_rank_sum(self):
        logger.info("Calculating rank sum for all test & models rankings")
        self.merge_df["RANK_SUM"] = self.merge_df[self.rank_column_names].sum(axis=1)
        return self

    def _assign_rank_all(self):
        logger.info("Creating final rank order for rank-sum")
        self.merge_df = self.merge_df.sort_values(by="RANK_SUM", ascending=True)
        self.merge_df["RANK_ALL"] = [r for r in range(self.merge_df.shape[0])]
        return self

    def _save_results(self, data: pd.DataFrame, filename: str):
        if self.save_results:
            utils.write_dataframe(
                pd_df=data,
                directory=self.output_dir,
                filename=filename,
                index=self.include_index,
            )

    def transform(self):
        self._rename_columns()
        self._limit_columns()
        self._merge_dataframes()
        self._fill_na()
        self._reorder_columns()
        self._calculate_rank_sum()
        self._assign_rank_all()
        self._save_results(data=self.merge_df, filename="feature_importance_all")
        return self
