# Real Estate Price Prediction App

## Overview
This Streamlit-based web application predicts real estate prices based on various property features and user inputs. It employs a linear regression model trained on property data to provide accurate price estimations.

## Features
- **User-friendly interface**: Enter property details via an interactive form.
- **Customizable options**: Choose property type, location, and features.
- **Real-time predictions**: Instantly get predicted prices after submitting the form.
- **Visualization**: Clean and responsive design for easy usage.

## How It Works
1. **Input data**: Users fill in property details such as type, location, size, and features.
2. **Data preprocessing**: Inputs are cleaned and processed to match the model’s requirements.
3. **Prediction**: A trained regression model predicts the property price based on the processed data.
4. **Result display**: The predicted price is displayed in a styled box on the app.

## Installation
1. Clone the repository:
    ```bash
    git clone <repository-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd real-estate-price-prediction
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Project Structure
```
.
├── app.py                  # Main script for launching the app
├── requirements.txt       # Python dependencies
├── style.css             # CSS for styling the app
├── .streamlit
│   └── config.toml   # Streamlit theme configuration
├── preprocessing
│   └── cleaning_data.py  # Functions for data cleaning and preprocessing
├── predict
│   ├── prediction.py     # Prediction logic and model loading
│   ├── regression.pkl    # Pre-trained regression model
│   └── scaler.pkl        # Scaler for data standardization
```

## How to Run the App
1. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
2. Open the local URL displayed in the terminal (e.g., `http://localhost:8501`).
3. Use the app by filling in the property details and clicking **See the result**.

## Input Features
The following features are used for prediction:
- **Property type**: Choose from houses (e.g., villa, mansion) or apartments (e.g., loft, studio).
- **ZIP code**: Enter the property’s location by postal code. Give access to district, median property prices and mean income in the area (derived automatically based on the ZIP code).
- **Living area**: Specify the size of the living area (in square meters).
- **Plot surface**: Specify the surface area of the plot (in square meters).
- **Building condition**: Select from conditions such as "As new," "Good," or "To renovate."
- **Swimming pool**: Indicate whether the property has a swimming pool.

## Preprocessing
- The `preprocessing/cleaning_data.py` file handles:
  - Cleaning ZIP code data.
  - Filling missing values for median prices and income.
  - Mapping categorical data to numerical values for modeling.
- The `predict/prediction.py` file:
  - Normalizes the input data using the scaler.
  - Loads the pre-trained regression model.
  - Outputs the predicted price.

