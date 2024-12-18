# Import librairies
import pandas as pd
import numpy as np

def dataframe_zip_code() -> pd.DataFrame:
    """
    Function that will clean the CSV file containing the zip code and the necessary data not provided by the user
    :return: the cleaned and complete dataframe
    """
    df = pd.read_csv('model/data/additional_data.csv')
    # Adding the missing district on the right line
    district = 37000
    df.loc[df['Postal code'] == 3717, 'district'] = district

    # Adding the provinces in the dataframe
    # Create a list of conditions
    conditions_province = [
        (df["Postal code"] <= 1299),
        (df["Postal code"] > 1299) & (df["Postal code"] <= 1499), 
        (df["Postal code"] > 1499) & (df["Postal code"] <= 1999) |
        (df["Postal code"] > 2999) & (df["Postal code"] <= 3499), 
        (df["Postal code"] > 1999) & (df["Postal code"] <= 2999),
        (df["Postal code"] > 3499) & (df["Postal code"] <= 3999),
        (df["Postal code"] > 3999) & (df["Postal code"] <= 4999),
        (df["Postal code"] > 4999) & (df["Postal code"] <= 5680),
        (df["Postal code"] > 5999) & (df["Postal code"] <= 6599) |
        (df["Postal code"] > 6999) & (df["Postal code"] <= 7999),
        (df["Postal code"] > 6599) & (df["Postal code"] <= 6999),
        (df["Postal code"] > 7999) & (df["Postal code"] <= 8999),
        (df["Postal code"] > 8999)
        ]

    # Create a list of the values we want to assign for each condition
    values_province = ["Brussels", "Brabant Wallon", "Vlaams-Brabant","Antwerp", 
                       "Limburg", "Liège", "Namur", "Hainaut", "Luxembourg", "West-Vlaanderen", 
                       "Oost-Vlaanderen"]

    # Create a new column and use np.select to assign values to it using our lists as arguments
    df["province"] = np.select(conditions_province, values_province, default="Unknown")
    # Filling in the empty values
    df['house-median-price'] = df.groupby(['district'])['house-median-price'].transform(lambda x: x.fillna(x.median()))
    df['apartment-median-price'] = df.groupby(['province'])['apartment-median-price'].transform(lambda x: x.fillna(x.median()))
    return df

# data used by the model -> columns used to predict a price
model_columns = ['Living area', 'Surface of the plot', 'Building condition',
    'Swimming pool', 'mean-income', 'median-price',
    'Property type_Bungalow', 'Property type_Castle',
    'Property type_Chalet', 'Property type_Country_Cottage',
    'Property type_Duplex', 'Property type_Exceptional_Property',
    'Property type_Farmhouse', 'Property type_Flat_Studio',
    'Property type_House', 'Property type_Kot', 'Property type_Loft',
    'Property type_Manor_House', 'Property type_Mansion',
    'Property type_Penthouse', 'Property type_Town_House',
    'Property type_Triplex', 'Property type_Villa', 'district_12000.0',
    'district_13000.0', 'district_21000.0', 'district_23000.0',
    'district_24000.0', 'district_25000.0', 'district_31000.0',
    'district_32000.0', 'district_33000.0', 'district_34000.0',
    'district_35000.0', 'district_36000.0', 'district_37000.0',
    'district_38000.0', 'district_41000.0', 'district_42000.0',
    'district_43000.0', 'district_44000.0', 'district_45000.0',
    'district_46000.0', 'district_51000.0', 'district_52000.0',
    'district_53000.0', 'district_55000.0', 'district_56000.0',
    'district_57000.0', 'district_58000.0', 'district_61000.0',
    'district_62000.0', 'district_63000.0', 'district_64000.0',
    'district_71000.0', 'district_72000.0', 'district_73000.0',
    'district_81000.0', 'district_82000.0', 'district_83000.0',
    'district_84000.0', 'district_85000.0', 'district_91000.0',
    'district_92000.0', 'district_93000.0']

def create_input_table(property_type, living_area, surface_plot, building_condition, swimming_pool, mean_income, median_price, district) -> pd.DataFrame:
    """
    Function that creates a dataframe containing the columns that the model needs and the data needed to predict a price
    :param: all the data needed to predict a price using the model
    :return: a dataframe containing the columns name and the data for one property
    """
    columns_data = pd.DataFrame(columns=model_columns)
    columns_data.loc[0] = 0
    columns_data['Living area'] = living_area
    columns_data['Surface of the plot'] = surface_plot
    columns_data['Building condition'] = building_condition
    columns_data['Swimming pool'] = swimming_pool
    columns_data['mean-income'] = mean_income
    columns_data['median-price'] = median_price
    if property_type != "Apartment":
        columns_data[f'Property type_{property_type}'] = 1
    if district != 11000.0:
        columns_data[f'district_{district}'] = 1
    return columns_data


def preprocess(property, property_type, zip_code, living_area, surface_plot, building_condition, swimming_pool) -> pd.DataFrame:
    """
    function that will process the input data to have all the info in the right format for prediction
    :param: input data 
    :return:the final dataframe with the information to make a prediction
    """
    # Property type: changing the name of the property type to match the columns from the model 
    if property_type == "Country cottage":
        property_type = "Country_Cottage"
    elif property_type == "Exceptional property":
        property_type = "Exceptional_Property"
    elif property_type == "Town house":
        property_type = "Town_House"
    elif property_type == "Manor house":
        property_type = "Manor_House"
    elif property_type == "Studio":
        property_type = "Flat_Studio"
    # from the zip code: getting the district code, the mean income and the median price
    zip_code_df = dataframe_zip_code()
    if zip_code in zip_code_df['Postal code'].values:
        zip_line = zip_code_df[zip_code_df['Postal code'] == zip_code]
        district = zip_line['district'].values[0]
        mean_income = zip_line['mean-income'].values[0]
        if property == "House":
            median_price = zip_line['house-median-price'].values[0]
        elif property == "Apartment":
            median_price = zip_line['apartment-median-price'].values[0]
    # Building condition: changing with numbers
    if building_condition == "As new":
        building_condition = 6
    elif building_condition == "Just renovated":
        building_condition = 5
    elif building_condition == "Good":
        building_condition = 4
    elif building_condition == "To be done up":
        building_condition = 3
    elif building_condition == "To renovate":
        building_condition = 2
    elif building_condition == "To restore":
        building_condition = 1
    # swimming pool: changing Yes/no by 1/0
    swimming_pool = 1 if swimming_pool == 'Yes' else 0		
    # Create the dataframe with all the info
    input_data = create_input_table(property_type, living_area, surface_plot, building_condition, swimming_pool, mean_income, median_price, district)
    return input_data



