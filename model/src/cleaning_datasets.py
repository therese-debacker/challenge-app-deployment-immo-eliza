import pandas as pd
import numpy as np

class CleaningDatasets:
    """
    Class for cleaning and transforming datasets using CSV files
    """
    def __init__(self):
        """
        Function that initializes the class with the path to the data files
        """
        self.datapath = "./data/"

    # removing unnecessary columns
    def drop_columns(
        self, df: pd.DataFrame, columns_to_drop: list[str]
    ) -> pd.DataFrame:
        """
        Drop specific columns from a DataFrame.
        :param df: Dataframe with columns to remove
        :param columns_to_drop: columns to be removed
        :return: dataframe without the columns to drop
        """
        df = df.drop(columns=columns_to_drop)
        return df

    def drop_rows(self, df: pd.DataFrame, rows_to_drop: tuple) -> pd.DataFrame:
        """
        Drop rows based on a condition
        :param df: dataframe with rows to drop
        :param rows_to_drop: condition to select the rows to drop
        :return: dataframe with rows removed
        """
        df = df.drop(df[rows_to_drop].index)
        return df

    def rename_columns(self, df: pd.DataFrame, names: dict[str, str]) -> pd.DataFrame:
        """
        rename columns in a dataframe
        :param df: dataframe with columns to rename
        :param names: dictionnary with old and new names
        :return: dataframe with renamed columns
        """
        df = df.rename(columns=names)
        return df

    def new_columns_sum(
        self, df: pd.DataFrame, group: str, column: str, new_column: str
    ) -> pd.DataFrame:
        """
        Add a new column containing the sum of values in a specified column, grouped by another column.
        :param df: the input DataFrame.
        :param group: the column name to group by.
        :param column: the column name to compute the sum for.
        :param new_column: the name of the new column to store the summed values.
        :return: the updated DataFrame with the new column added.
        """
        sum_column = df.groupby(group)[column].sum()
        df[new_column] = df[group].map(sum_column)
        return df

    def new_columns_mean(
        self, df: pd.DataFrame, group: str, column: str, new_column: str
    ) -> pd.DataFrame:
        """
        Add a new column containing the mean of values in a specified column, grouped by another column.
        :param df: the input DataFrame.
        :param group: the column name to group by.
        :param column: the column name to compute the mean for.
        :param new_column: the name of the new column to store the mean values.
        :return: the updated DataFrame with the new column added.
        """
        median_column = df.groupby(group)[column].mean()
        df[new_column] = df[group].map(median_column)
        return df

    def new_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add new calculated columns to the DataFrame.
        - 'population-per-surface-district': population divided by surface area per district.
        - 'number-transactions': sum of house and apartment sales.
        :param df: the input DataFrame.
        :return: the updated DataFrame with the new columns added.
        """
        df["population-per-surface-district"] = (
            df["population-district"] / df["surface-area-total"]
        )
        df["number-transactions"] = (
            df["nb_transactions_house"] + df["nb_transactions_apartment"]
        )
        return df

    def new_columns_conditions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add a 'median-price' column based on property type (house or apartment).
        :param df: the input DataFrame.
        :return: the updated DataFrame with the new column added.
        """
        df["median-price"] = np.where(
            df["Property"] == "House",
            df["house-median-price"],
            df["apartment-median-price"],
        )
        return df

    def replace_elements(
        self, df: pd.DataFrame, column: str, element: str, replaced: str
    ) -> pd.DataFrame:
        """
        Replace specified elements in a column with a new value.
        :param df: the input DataFrame.
        :param column: the column to modify.
        :param element: the element to replace.
        :param replaced: the replacement value.
        :return: the updated DataFrame with the replacements applied.
        """
        df[column] = df[column].str.replace(element, replaced)
        return df

    def merging_dataset(
        self, df1: pd.DataFrame, df2: pd.DataFrame, column1: str, column2: str
    ) -> pd.DataFrame:
        """
        Merge two DataFrames on specified columns.
        :param df1: the first DataFrame.
        :param df2: the second DataFrame.
        :param column1: the column in the first DataFrame to join on.
        :param column2: the column in the second DataFrame to join on.
        :return: the merged DataFrame.
        """
        df = pd.merge(df1, df2, left_on=column1, right_on=column2, how="left")
        return df

    def change_type(self, df: pd.DataFrame, column: str, dtype: type) -> pd.DataFrame:
        """
        Change the data type of a specified column
        :param df: the input dataframe
        :param column: the column to change the type for
        :param: the desired data type
        :return: the updated dataframe with the column type changed
        """
        df[column] = df[column].astype(dtype)
        return df