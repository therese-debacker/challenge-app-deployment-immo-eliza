import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 


# Loading in the model to predict on the data 

def dataframe_zip_code():
	df = pd.read_csv('model/data/additional_data.csv')
	# empty_row = df.loc[df['Postal code'] == 3717]
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
	values_province = ["Brussels", "Brabant Wallon", "Vlaams-Brabant","Antwerp", "Limburg", "Liège", "Namur", "Hainaut", "Luxembourg", "West-Vlaanderen", "Oost-Vlaanderen"]

	# Create a new column and use np.select to assign values to it using our lists as arguments
	df["province"] = np.select(conditions_province, values_province, default="Unknown")
	# Filling in the empty values
	df['house-median-price'] = df.groupby(['district'])['house-median-price'].transform(lambda x: x.fillna(x.median()))
	df['apartment-median-price'] = df.groupby(['province'])['apartment-median-price'].transform(lambda x: x.fillna(x.median()))
	return df

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction(data): 
	pickle_in = open('predict/regression.pkl', 'rb') 
	regression = pickle.load(pickle_in) 
	prediction = regression.predict(data)[0] 
	return prediction 
	

# this is the main function in which we define our webpage 
def main(): 
	
	st.title("Real estate price prediction") 
	st.header("Fill in the form to get a prediction for your property")
	
	# Loading CSS style
	st.markdown('<style>' + open('style.css').read() + '</style>', unsafe_allow_html=True)


	
	
	house_types = ["House", "Bungalow", "Castle", "Chalet", "Country cottage", 
				   "Exceptional property","Farmhouse", 
				    "Manor house", "Mansion", "Town house","Villa"]
	apartment_types = ["Apartment", "Loft", "Penthouse", "Triplex", "Duplex", 
					"Studio", "Kot"]
	
	with st.container():
		property = st.radio("Is it a house or an apartment?", options=["House", "Apartment"], index=0, horizontal= True) 
		if property == "House": 
			property_type = st.selectbox("What's the type of house?", options=house_types)
			# Changing the name of the property type to match the columns from the model 
			if property_type == "Country cottage":
				property_type = "Country_Cottage"
			elif property_type == "Exceptional property":
				property_type = "Exceptional_Property"
			elif property_type == "Town house":
				property_type = "Town_House"
			elif property_type == "Manor house":
				property_type = "Manor_House"
		elif property == "Apartment":
			property_type = st.selectbox("What's the type of apartment?", options=apartment_types) 
			if property_type == "Studio":
				property_type = "Flat_Studio"

	# ZIP CODE
	with st.container():
		zip_code = st.number_input("Enter the zip code", placeholder="1000", min_value = 1000, max_value = 9999) 
		zip_code_df = dataframe_zip_code()
		if zip_code in zip_code_df['Postal code'].values:
			zip_line = zip_code_df[zip_code_df['Postal code'] == zip_code]
			district = zip_line['district'].values[0]
			mean_income = zip_line['mean-income'].values[0]
			if property == "House":
				median_price = zip_line['house-median-price'].values[0]
			elif property == "Apartment":
				median_price = zip_line['apartment-median-price'].values[0]
		else:
			st.warning("Enter a valid zip code")

	with st.container():
		st.write("Surface of spaces")
		col1, col2 = st.columns(2)
			# LIVING AREA
		living_area = col1.number_input("Living area in square meters", placeholder=100, min_value = 5) 
		if living_area < 5:
			col1.warning("Enter a valid answer")
		# SURFACE OF THE PLOT
		surface_plot = col2.number_input("Surface of the plot in square meters", placeholder=100, min_value = 5) 
		if surface_plot < 5:
			col2.warning("Enter a valid answer")
	with st.container():	
		# BUILDING CONDITION
		options_condition = ["As new", "Just renovated", "Good", "To be done up", "To renovate", "To restore"]
		building_condition = st.selectbox("What's the building condition?",options=options_condition,index=2)
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


	
	with st.container():
		# SWIMMING POOL
		swimming_pool = st.radio("Does the property have a swimming pool?", options=["Yes", "No"], index=1, horizontal= True) 
		swimming_pool = 1 if swimming_pool == 'Yes' else 0		
	
	preprocess(living_area, surface_plot, building_condition, swimming_pool, mean_income, median_price, property_type, district)
	
	result ="" 
    

	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction 
	# and store it in the variable result 
	if st.button("See the result"): 
		columns_data = create_input_table(living_area, surface_plot, building_condition, swimming_pool, mean_income, median_price, property_type, district)
		with open('predict/scaler.pkl', 'rb') as scaler_file:
			scaler = pickle.load(scaler_file)
		columns_data_scaled= scaler.transform(columns_data)
		result = prediction(columns_data_scaled) 
		result = int(result)
		st.markdown(f"<div style='text-align:center; border-radius:15px;witdh: 100%; background-color: #52ab98;color: #fff; font-size:25px;padding:20px;'>The predicted price is <br> <b>{(result)}</b> €</div>", unsafe_allow_html=True)

if __name__=='__main__': 
	main()