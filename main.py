import pandas as pd
from src.cleaning_datasets import CleaningDatasets
from src.cleaning_feature_engineering import FeatureEngineering
from src.linear_regression_model import LinearRegressionModel
from time import perf_counter
import pickle 


def main():
    """
    Main script to clean, preprocess, and train a linear regression model
    for predicting real estate prices.
    """
    start_time = perf_counter()
    cleaner = CleaningDatasets()
    
    # Import the necessary CSV files
    df = pd.read_csv(f'./data/precleaned-dataset-immoweb.csv')
    zip_code = pd.read_csv(f'./data/code-nis-zip-code.csv')
    income_median = pd.read_csv(f'./data/median-income-2022.csv')
    income_mean = pd.read_csv(f'./data/mean-income-2022.csv')
    median_price = pd.read_csv(f'./data/sales-real-estates-belgium-district.csv')

    # Cleaning of the median_price dataframe
    median_price = cleaner.drop_rows(median_price, (median_price['année'] != 2023))
    columns_drop_medianprice = ['localité', 'année', 'période','prix premier quartile(€)-maison', 'prix troisième quartile(€)-maison','prix premier quartile(€)-appartement',
       'prix troisième quartile(€)-appartement', 'Unnamed: 12', 'Unnamed: 13', 'nombre transactions-appartement', 'nombre transactions - maison',
       'Unnamed: 14', '0']
    median_price = cleaner.drop_columns(median_price, columns_drop_medianprice)
    new_names_median_price = {
        'prix médian(€)-maison': 'house-median-price',
        'prix médian(€)-appartement': 'apartment-median-price'
        }  
    median_price = cleaner.rename_columns(median_price, new_names_median_price)
    median_price = cleaner.new_columns_mean(median_price,'refnis', 'house-median-price', 'house-median-price')
    median_price = cleaner.new_columns_mean(median_price,'refnis', 'apartment-median-price','apartment-median-price')
    median_price = median_price.drop_duplicates()

    # Cleaning of the income_median CSV
    income_median = cleaner.drop_rows(income_median, (income_median['CD_YEAR'] != 2022))

    # Merging CSVs
    merged_df_income = cleaner.merging_dataset(zip_code, income_median,'Refnis code', 'CD_MUNTY_REFNIS')
    merged_df_avgincome = cleaner.merging_dataset(merged_df_income, income_mean,'Nom commune', 'Nom')
    merged_df_median_price = cleaner.merging_dataset(merged_df_avgincome, median_price,'CD_DSTR_REFNIS', 'refnis')

    # Removing unuseful columns & rename some others
    columns_to_drop = ['CD_RGN_REFNIS','TX_RGN_DESCR_NL','TX_RGN_DESCR_FR', 'TX_RGN_DESCR_EN', 'TX_RGN_DESCR_DE', 'CD_PROV_REFNIS', 'TX_PROV_DESCR_NL',
                             'TX_PROV_DESCR_FR', 'TX_PROV_DESCR_EN','TX_PROV_DESCR_DE','TX_DSTR_DESCR_NL','TX_DSTR_DESCR_FR','TX_DSTR_DESCR_EN','TX_DSTR_DESCR_DE',
                             'TX_MUNTY_DESCR_EN','TX_MUNTY_DESCR_DE','MS_Q1','MS_Q3','MS_NBR_ELIGIBLE','MS_NBR_NOT_ELIGIBLE','MS_PERC_NOT_ELIGIBLE','MS_PERC_IOE_HH',
                             'MS_INT_QUART_DIFF', 'Commune', 'CD_YEAR','TX_MUNTY_DESCR_NL','TX_MUNTY_DESCR_FR','Gemeentenaam', 'Nom commune','CD_MUNTY_REFNIS', 
                             'Refnis code']

    merged_df_median_price = cleaner.drop_columns(merged_df_median_price, columns_to_drop)
    merged_median_price = {'total': 'population','MS_MEDIAN': 'median-income','Nom': 'commune','Revenu': 'mean-income', 'MS_ADMIN_AROP': 'poverty-chance', 
                            'CD_DSTR_REFNIS':'district'}
    merged_df_median_price = cleaner.rename_columns(merged_df_median_price, merged_median_price)
    
    # Changing the data type in one column
    merged_df_median_price = cleaner.change_type(merged_df_median_price, 'district', str)

    # Merging with the immoweb dataset, removing one column and adding a new one
    final_df = cleaner.merging_dataset(df, merged_df_median_price,'Zip code', 'Postal code')
    final_df = cleaner.drop_columns(final_df, ['Postal code'])
    final_df = cleaner.new_columns_conditions(final_df)

    engineering = FeatureEngineering(final_df)

    # Transforming the final dataframe by removing outliers and some rows, replacing empty values and transforming columns for the model
    categorical_columns = ['Property type', 'district']
    final_df = engineering.remove_outliers()
    final_df = engineering.remove_rows('Living area')
    final_df = engineering.replace_navalues()
    final_df = engineering.transform_columns()
    final_df = engineering.transform_categorical_values(categorical_columns)
    
    # Removing additional columns that won't be necessary
    columns_to_drop =  ['Facades','Equipped kitchen', 'Furnished', 'Fireplace','Garden', 'Terrace', 'Terrace surface','Region', 'Bedrooms',
                    'Zip code', 'Locality', 'median-income', 'commune', 'Province','Garden surface','poverty-chance', 'Property', 
                    'house-median-price', 'apartment-median-price', 'refnis' ]
    final_df = cleaner.drop_columns(final_df, columns_to_drop)

    # Save the cleaned and preprocessed dataset
    final_df.to_csv(f'./data/dataset-preprocessed.csv', index=False)
    # Split the features and the target
    X = final_df.drop(columns=['Price'])
    y = final_df['Price']
    print(final_df.shape)
    # Train the linear regression model and getting the metrics
    model_trainer = LinearRegressionModel(final_df, X, y)
    model_trainer.create_linear_model()
    print(f"\nTime taken to from start to finish: {round(perf_counter()-start_time,3)} seconds.")

    # pickling the model 
    pickle_out = open("classifier.pkl", "wb")   
    pickle.dump(model_trainer, pickle_out) 
    pickle_out.close()


if __name__ == "__main__":
    main()