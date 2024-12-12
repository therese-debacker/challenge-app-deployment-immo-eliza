import pandas as pd 
import numpy as np 
import pickle 
import streamlit as st 
from PIL import Image 

# loading in the model to predict on the data 
pickle_in = open('classifier.pkl', 'rb') 
regression = pickle.load(pickle_in) 

def welcome(): 
	return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction(property_type, living_area, surface_plot, building_condition, swimming_pool, zip_code): 

	prediction = regression.predict( 
		[[property_type,living_area, surface_plot, building_condition, swimming_pool, zip_code]]) 
	print(prediction) 
	return prediction 
	

# this is the main function in which we define our webpage 
def main(): 
	# giving the webpage a title 
	st.title("House Price Prediction") 
	
	# here we define some of the front end elements of the web page like 
	# the font and background color, the padding and the text to be displayed 
	html_temp = """ 
	<div style ="background-color:yellow;padding:13px"> 
	<h1 style ="color:black;text-align:center;">House Price Prediction ML App </h1> 
	</div> 
	"""
	
	# this line allows us to display the front end aspects we have 
	# defined in the above code 
	st.markdown(html_temp, unsafe_allow_html = True) 
	
	# the following lines create text boxes in which the user can enter 
	# the data required to make the prediction 
	property_type = st.text_input("Property type", "Type Here") 
	living_area = st.text_input("Living area", "Type Here") 
	surface_plot = st.text_input("Surface of the plot", "Type Here") 
	building_condition = st.text_input("Building condition", "Type Here")
	swimming_pool = st.text_input("Swimming pool", "Type Here") 
	zip_code = st.text_input("Zip code", "Type Here") 
	result ="" 
	
	# the below line ensures that when the button called 'Predict' is clicked, 
	# the prediction function defined above is called to make the prediction 
	# and store it in the variable result 
	if st.button("Predict"): 
		result = prediction(property_type, living_area, surface_plot, building_condition, swimming_pool, zip_code) 
	st.success('The output is {}'.format(result)) 
	
if __name__=='__main__': 
	main() 
