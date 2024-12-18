# Import necessary libraries and classes
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from cleaning_datasets import CleaningDatasets

def create_final_dataset() -> pd.DataFrame:
    """
    Function that creates the dataset with all the data gathered
    :return: final dataframe that will be analysed
    """
    cleaner = CleaningDatasets()

    # Import all CSV files
    df = pd.read_csv(f"./data/precleaned-dataset-immoweb.csv")
    zip_code = pd.read_csv(f"./data/code-nis-zip-code.csv")
    income_median = pd.read_csv(f"./data/median-income-2022.csv")
    density_population = pd.read_csv(f"./data/density-population.csv")
    income_mean = pd.read_csv(f"./data/mean-income-2022.csv")
    surface_area = pd.read_csv(f"./data/surface-area-2024-district.csv", header=None)
    median_price = pd.read_csv(f"./data/sales-real-estates-belgium-district.csv")

    # Cleaning of median_price dataframe : remove rows, columns, rename columns, create new columns
    median_price = cleaner.drop_rows(median_price, (median_price["année"] != 2023))
    columns_drop_medianprice = [
        "localité",
        "année",
        "période",
        "prix premier quartile(€)-maison",
        "prix troisième quartile(€)-maison",
        "prix premier quartile(€)-appartement",
        "prix troisième quartile(€)-appartement",
        "Unnamed: 12",
        "Unnamed: 13",
        "Unnamed: 14",
        "0",
    ]
    median_price = cleaner.drop_columns(median_price, columns_drop_medianprice)
    new_names_median_price = {
        "nombre transactions - maison": "nb_transactions_house",
        "prix médian(€)-maison": "house-median-price",
        "nombre transactions-appartement": "nb_transactions_apartment",
        "prix médian(€)-appartement": "apartment-median-price",
    }
    median_price = cleaner.rename_columns(median_price, new_names_median_price)
    median_price = cleaner.new_columns_sum(
        median_price, "refnis", "nb_transactions_house", "nb_transactions_house"
    )
    median_price = cleaner.new_columns_sum(
        median_price, "refnis", "nb_transactions_apartment", "nb_transactions_apartment"
    )
    median_price = cleaner.new_columns_mean(
        median_price, "refnis", "house-median-price", "house-median-price"
    )
    median_price = cleaner.new_columns_mean(
        median_price, "refnis", "apartment-median-price", "apartment-median-price"
    )
    median_price = median_price.drop_duplicates()

    # Cleaning of surface_area dataframe : remove columns, drop rows, rename columns, create new columns
    surface_area.columns = [
        "refnis",
        "locality",
        "rubrique",
        "rubrique detail",
        "number-parcels",
        "surface-area-taxable",
        "surface-area-exonerate",
        "surface-area-total",
        "surface-area-promille",
    ]
    surface_area_columns = [
        "locality",
        "surface-area-taxable",
        "surface-area-exonerate",
        "surface-area-promille",
    ]
    surface_area = cleaner.drop_columns(surface_area, surface_area_columns)
    surface_area_total = cleaner.drop_rows(
        surface_area, (surface_area["rubrique"] != "6TOT")
    )
    surface_area_built = cleaner.drop_rows(
        surface_area, (surface_area["rubrique"] != "2TOT")
    )
    surface_area_land = cleaner.drop_rows(
        surface_area, (surface_area["rubrique"] != "1TOT")
    )
    new_names_surface_area = {"number-parcels": "number-parcels-total"}
    surface_area_total = cleaner.rename_columns(
        surface_area_total, new_names_surface_area
    )
    surface_area_built = cleaner.rename_columns(
        surface_area_built,
        {
            "number-parcels": "number-parcels-built",
            "surface-area-total": "surface-area-total-built",
        },
    )
    surface_area_land = cleaner.rename_columns(
        surface_area_land,
        {
            "number-parcels": "number-parcels-land",
            "surface-area-total": "surface-area-total-land",
        },
    )
    surface_area_total = cleaner.merging_dataset(
        surface_area_total, surface_area_built, "refnis", "refnis"
    )
    surface_area_total = cleaner.merging_dataset(
        surface_area_total, surface_area_land, "refnis", "refnis"
    )
    surface_area_new_columns = [
        "rubrique_x",
        "rubrique detail_x",
        "rubrique_y",
        "rubrique detail_y",
        "rubrique detail",
        "rubrique",
    ]
    surface_area_total = cleaner.drop_columns(
        surface_area_total, surface_area_new_columns
    )

    # Clean income_median CSV
    income_median = cleaner.drop_rows(income_median, (income_median["CD_YEAR"] != 2022))

    # Merging CSVs
    merged_df_income = cleaner.merging_dataset(
        zip_code, income_median, "Refnis code", "CD_MUNTY_REFNIS"
    )
    merged_df_population = cleaner.merging_dataset(
        merged_df_income, density_population, "CD_MUNTY_REFNIS", "code-ins"
    )
    merged_df_avgincome = cleaner.merging_dataset(
        merged_df_population, income_mean, "Nom commune", "Nom"
    )
    merged_df_surface_area = cleaner.merging_dataset(
        merged_df_avgincome, surface_area_total, "CD_DSTR_REFNIS", "refnis"
    )
    merged_df_median_price = cleaner.merging_dataset(
        merged_df_surface_area, median_price, "CD_DSTR_REFNIS", "refnis"
    )

    # Removing unuseful columns
    columns_to_drop = [
        "CD_RGN_REFNIS",
        "TX_RGN_DESCR_NL",
        "TX_RGN_DESCR_FR",
        "TX_RGN_DESCR_EN",
        "TX_RGN_DESCR_DE",
        "CD_PROV_REFNIS",
        "TX_PROV_DESCR_NL",
        "TX_PROV_DESCR_FR",
        "TX_PROV_DESCR_EN",
        "TX_PROV_DESCR_DE",
        "TX_DSTR_DESCR_NL",
        "TX_DSTR_DESCR_FR",
        "TX_DSTR_DESCR_EN",
        "TX_DSTR_DESCR_DE",
        "TX_MUNTY_DESCR_EN",
        "TX_MUNTY_DESCR_DE",
        "MS_Q1",
        "MS_Q3",
        "MS_NBR_ELIGIBLE",
        "MS_NBR_NOT_ELIGIBLE",
        "MS_PERC_NOT_ELIGIBLE",
        "MS_PERC_IOE_HH",
        "MS_INT_QUART_DIFF",
        "Commune",
        "CD_YEAR",
        "TX_MUNTY_DESCR_NL",
        "TX_MUNTY_DESCR_FR",
        "Gemeentenaam",
        "Nom commune",
        "CD_MUNTY_REFNIS",
        "Refnis code",
        "number-parcels-total",
        "refnis_y",
        "refnis_x",
        "men",
        "women",
        "code-ins",
    ]
    merged_df_median_price = cleaner.drop_columns(
        merged_df_median_price, columns_to_drop
    )

    # Rename some columns
    merged_median_price = {
        "total": "population",
        "MS_MEDIAN": "median-income",
        "Nom": "commune",
        "Revenu": "mean-income",
        "MS_ADMIN_AROP": "poverty-chance",
        "CD_DSTR_REFNIS": "district",
    }
    merged_df_median_price = cleaner.rename_columns(
        merged_df_median_price, merged_median_price
    )

    # Removing unnecessary elements
    merged_df_median_price = cleaner.replace_elements(
        merged_df_median_price, "poverty-chance", ",", "."
    )
    merged_df_median_price = cleaner.replace_elements(
        merged_df_median_price, "population", " ", ""
    )

    # Changing types
    merged_df_median_price = cleaner.change_type(
        merged_df_median_price, "poverty-chance", float
    )
    merged_df_median_price = cleaner.change_type(
        merged_df_median_price, "population", float
    )
    merged_df_median_price = cleaner.change_type(
        merged_df_median_price, "district", str
    )

    # Creating new columns
    merged_df_median_price = cleaner.new_columns_mean(
        merged_df_median_price, "district", "mean-income", "mean-income-district"
    )
    merged_df_median_price = cleaner.new_columns_mean(
        merged_df_median_price, "district", "median-income", "median-income-district"
    )
    merged_df_median_price = cleaner.new_columns_mean(
        merged_df_median_price, "district", "poverty-chance", "poverty-chance-district"
    )
    merged_df_median_price = cleaner.new_columns_sum(
        merged_df_median_price, "district", "population", "population-district"
    )
    merged_df_median_price = cleaner.new_columns(merged_df_median_price)

    # Merging with the immoweb dataset
    final_df = cleaner.merging_dataset(
        df, merged_df_median_price, "Zip code", "Postal code"
    )
    final_df = cleaner.drop_columns(final_df, ["Postal code"])
    final_df = cleaner.new_columns_conditions(final_df)

    return final_df


# Creation of the dataset to analyze
df = create_final_dataset()


# Getting a first look at the dataset to choose the features will use in our model and know what changes we have to make
def dataset_check_graphs_info(part: str):
    """
    Function that will give us an overview of the dataframe and create a correlation map (a general one and one per house and per apartment)
    :Parameter: a name to include in the heatmap name to be able to create the heatmap several times without erasing the previous ones
    """
    print(df.columns)
    print(df.describe())
    print(df.info())
    print(df.dtypes)
    print(df.isnull().sum())
    # Heatmaps
    plt.figure(figsize=(30, 20))
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.tight_layout()
    plt.savefig(f"./graphs/correlation-heatmap-{part}.png")
    plt.clf()
    df_houses = df.drop(df[df["Property"] != "House"].index)
    sns.heatmap(df_houses.corr(numeric_only=True), annot=True)
    plt.tight_layout()
    plt.savefig(f"./graphs/correlation-heatmap-houses-{part}.png")
    plt.clf()
    df_apartments = df.drop(df[df["Property"] != "Apartment"].index)
    sns.heatmap(df_apartments.corr(numeric_only=True), annot=True)
    plt.tight_layout()
    plt.savefig(f"./graphs/correlation-heatmap-apartment-{part}.png")
    plt.clf()
    # Boxplot of the price
    sns.boxplot(data=df.Price)
    step = 500000
    max_price = df["Price"].max()
    yticks = np.arange(0, max_price + step, step)
    plt.yticks(yticks, labels=[f"{int(y):,}" for y in yticks])
    plt.grid(alpha=0.5, linestyle="--")
    plt.savefig(f"./graphs/boxplot-price-{part}.png")


dataset_check_graphs_info("one")


# Creating graphs to have a better view of the dataset
def check_closer_corr(column: str):
    """
    Function that creates a scatterplot based on one column and the price and adding colours based on the property type
    :Parameter: the column we want to compare with the price
    """
    plt.clf()
    plt.figure(figsize=(15, 10))
    sns.scatterplot(x=column, y="Price", data=df, hue="Property")
    plt.ticklabel_format(style="plain", axis="y")
    step = 500000
    max_price = df["Price"].max()
    yticks = np.arange(0, max_price + step, step)
    plt.yticks(yticks, labels=[f"{int(y):,}" for y in yticks])
    plt.grid(alpha=0.5, linestyle="--")
    plt.savefig(f"./graphs/graph-{column}-price-property.png")
    plt.clf()
    sns.boxplot(data=df[column])
    plt.grid(alpha=0.5, linestyle="--")
    plt.savefig(f"./graphs/boxplot-{column}.png")


check_closer_corr("Garden surface")
check_closer_corr("Living area")
check_closer_corr("Surface of the plot")
check_closer_corr("Building condition")


# Creating boxplots to see outliers that may have to be removed
def check_boxplot(column: str):
    """
    Function that creates a boxplot based on one column and the price
    :Parameter: the column we want to compare with the price
    """
    plt.clf()
    plt.figure(figsize=(15, 15))
    sns.boxplot(x=column, y="Price", data=df)
    step = 500000
    max_price = df["Price"].max()  # Valeur maximum de Price
    yticks = np.arange(0, max_price + step, step)
    plt.yticks(yticks, labels=[f"{int(y):,}" for y in yticks])
    plt.xticks(rotation=90)
    plt.savefig(f"./graphs/boxplot-{column}-price.png")


check_boxplot("Building condition")
check_boxplot("Property type")
check_boxplot("Property")


# Removing the outliers
def remove_outliers():
    """
    Functiun that removes the outliers we chose to be able to get the new correlation and info about the dataset
    """
    df.drop(df[df["Price"] > 2500000].index, inplace=True)
    df.drop(df[df["Property type"] == "Other_Property"].index, inplace=True)
    df.drop(
        df[
            (df["Property type"] == "Mixed_Use_Building") & (df["Living area"] > 1200)
        ].index,
        inplace=True,
    )
    df.drop(
        df[(df["Property"] == "Apartment") & (df["Living area"] > 450)].index,
        inplace=True,
    )
    df.drop(
        df[(df["Property"] == "Apartment") & (df["Price"] > 1000000)].index,
        inplace=True,
    )
    df.drop(
        df[(df["Property"] == "House") & (df["Living area"] > 1200)].index, inplace=True
    )
    df.drop(
        df[(df["Property"] != "House") & (df["Property"] != "Apartment")].index,
        inplace=True,
    )


remove_outliers()

# Creating the new correlation graphs to see the difference once the outliers have been removed
dataset_check_graphs_info("two")