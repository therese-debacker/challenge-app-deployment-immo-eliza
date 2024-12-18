# Import librairies and functions from other files
import pandas as pd
import numpy as np
import streamlit as st
from preprocessing.cleaning_data import dataframe_zip_code, preprocess
from predict.prediction import predict

def main():
    """
    Main script to launch the app and get input data from users
    for predicting real estate prices.
    """
    # Loading CSS style
    st.markdown(
        "<style>" + open("style.css").read() + "</style>", unsafe_allow_html=True
    )
    # Sidebar
    with st.sidebar:
        st.sidebar.header("How to use the app")
        st.sidebar.write(
            """
1. Fill in all the fields in the form.
2. Click on 'See the result'.
3. Your prediction will appear below the button.
"""
        )
        st.sidebar.header("The features")
        st.sidebar.write(
            "This app is based on a linear regression. Here are the features used to predict the price:"
        )
        st.sidebar.write(
            """
1. The property type
2. Mean income in the zip code
3. Median price in the district (based on the property)
4. The district
5. Living area
6. Surface of the plot
7. The building condition
8. Swimming pool                         
"""
        )
    # Title and subheader
    st.title("Real estate price prediction")
    # Space
    st.markdown(
        "<style>div.block-container{padding-top:4rem;}</style>", unsafe_allow_html=True
    )

    # Input data: property type
    house_types = [
        "House",
        "Bungalow",
        "Castle",
        "Chalet",
        "Country cottage",
        "Exceptional property",
        "Farmhouse",
        "Manor house",
        "Mansion",
        "Town house",
        "Villa",
    ]
    apartment_types = [
        "Apartment",
        "Loft",
        "Penthouse",
        "Triplex",
        "Duplex",
        "Studio",
        "Kot",
    ]

    with st.container():
        property = st.radio(
            "Is it a house or an apartment?",
            options=["House", "Apartment"],
            index=0,
            horizontal=True,
        )
        if property == "House":
            property_type = st.selectbox(
                "What's the type of house?", options=house_types
            )
        elif property == "Apartment":
            property_type = st.selectbox(
                "What's the type of apartment?", options=apartment_types
            )

    # Input data: ZIP CODE
    with st.container():
        zip_code = st.number_input(
            "Where is it located? Enter the zip code:",
            placeholder="1000",
            min_value=1000,
            max_value=9999,
        )
        zip_code_df = dataframe_zip_code()
        # Checking if the zip code exists
        if zip_code not in zip_code_df["Postal code"].values:
            st.warning("Enter a valid zip code")

    # Input data: surfaces
    with st.container():
        col1, col2 = st.columns(2)
        # LIVING AREA
        living_area = col1.number_input(
            "What's the living area (in square meters)?", placeholder=100, min_value=5
        )
        if living_area < 5:
            col1.warning("Enter a valid answer")
        # SURFACE OF THE PLOT
        surface_plot = col2.number_input(
            "What's the surface of the plot (in square meters)?",
            placeholder=100,
            min_value=5,
        )
        if surface_plot < 5:
            col2.warning("Please enter a valid answer")

    # Input data: BUILDING CONDITION
    with st.container():
        options_condition = [
            "As new",
            "Just renovated",
            "Good",
            "To be done up",
            "To renovate",
            "To restore",
        ]
        building_condition = st.selectbox(
            "What's the building condition?", options=options_condition, index=2
        )

    # Input data: SWIMMING POOL
    with st.container():
        swimming_pool = st.radio(
            "Does the property have a swimming pool?",
            options=["Yes", "No"],
            index=1,
            horizontal=True,
        )

    result = ""

    # Calling the functions to clean the input data and get a price prediction once the user click oon the button
    try:
        if st.button("See the result"):
            columns_data = preprocess(
                property,
                property_type,
                zip_code,
                living_area,
                surface_plot,
                building_condition,
                swimming_pool,
            )
            result = predict(columns_data)
            result = int(result)
            # Display the price prediction stored in the variable "result"
            st.markdown(
                f"<div style='text-align:center; border-radius:15px;witdh: 100%; background-color: #27334e;color: #fff; font-size:30px;padding:20px;'>The predicted price is <br> <b>{(result)}</b> €</div>",
                unsafe_allow_html=True,
            )
    except:
        print("Please fill in the form with correct info")


if __name__ == "__main__":
    main()
