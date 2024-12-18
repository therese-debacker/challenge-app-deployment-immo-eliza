import pandas as pd

class FeatureEngineering:
    """
    Class to handle data cleaning for our real estate dataset
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialize the class with a dataframe
        :param df: dataframe containing the real estate dataset
        """
        self.df = df

    def remove_outliers(self) -> pd.DataFrame:
        """
        Function that removes the selected outliers that were spotted in the analysis
        :return: the new dataset without the outliers
        """
        self.df.drop(self.df[self.df["Price"] > 2500000].index, inplace=True)
        self.df.drop(
            self.df[self.df["Property type"] == "Other_Property"].index, inplace=True
        )
        self.df.drop(
            self.df[
                (self.df["Property type"] == "Mixed_Use_Building")
                & (self.df["Living area"] > 1200)
            ].index,
            inplace=True,
        )
        self.df.drop(
            self.df[
                (self.df["Property"] == "Apartment") & (self.df["Living area"] > 450)
            ].index,
            inplace=True,
        )
        self.df.drop(
            self.df[
                (self.df["Property"] == "Apartment") & (self.df["Price"] > 1000000)
            ].index,
            inplace=True,
        )
        self.df.drop(
            self.df[
                (self.df["Property"] == "House") & (self.df["Living area"] > 1200)
            ].index,
            inplace=True,
        )
        self.df.drop(
            self.df[
                (self.df["Property"] != "House") & (self.df["Property"] != "Apartment")
            ].index,
            inplace=True,
        )
        return self.df

    def remove_rows(self, column: str) -> pd.DataFrame:
        """
        Function that removes rows with missing values in specific columns
        :return: a dataframe with missing values in specific columns removed
        """
        self.df.dropna(subset=[column], inplace=True)
        return self.df

    def replace_navalues(self) -> pd.DataFrame:
        """
        Function that replaces missing values in specific columns using the median
        :return: the dataframe with missing values replaced
        """
        self.df["Surface of the plot"] = self.df.groupby("district")[
            "Surface of the plot"
        ].transform(lambda x: x.fillna(x.median()))
        self.df["median-price"] = self.df.groupby(["Province", "Property"])[
            "median-price"
        ].transform(lambda x: x.fillna(x.median()))
        return self.df

    def transform_columns(self) -> pd.DataFrame:
        """
        Function that transforms columns by removing irrelevant values (removing the rows)
        and replacing the strings by numerical values
        :return: the transformed dataframe
        """
        self.df.drop(
            self.df[self.df["Building condition"] == "Not mentioned"].index,
            inplace=True,
        )
        self.df["Building condition"] = self.df["Building condition"].replace(
            {
                "As new": 6,
                "Just renovated": 5,
                "Good": 4,
                "To be done up": 3,
                "To renovate": 2,
                "To restore": 1,
            }
        )
        return self.df

    # Transform categorical columns into new columns of 1/0
    def transform_categorical_values(
        self, categorical_columns: list[str]
    ) -> pd.DataFrame:
        """
        Function that transforms the categorical columns into dummy variables
        :param categorical_columns: list of the column names to transform into dummy variables
        :return: the dataframe with categorical columns transformed
        """
        self.df = pd.get_dummies(self.df, columns=categorical_columns, drop_first=True)
        return self.df