 Real Estate Price Prediction

This project predicts real estate prices using a linear regression model. It combines several datasets, cleans and preprocesses the data, and trains a machine learning model to provide accurate predictions. The project is modular and easy to extend, ensuring maintainability and scalability.

## **Project Structure**
├── data/ # Folder for raw and processed datasets ├── graphs/ # Folder for generated plots ├── main.py # Main script to execute the pipeline ├── cleaning_datasets.py # Module for data cleaning and merging ├── cleaning_feature_engineering.py # Module for feature engineering ├── linear_regression_model.py # Module for training and evaluating the model ├── requirements.txt # List of dependencies └── README.md # Project documentation

bash
Copier le code

## **How to Use**

### **1. Setup**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo-name/real-estate-price-prediction.git
   cd real-estate-price-prediction
Install the required dependencies:
bash
Copier le code
pip install -r requirements.txt
Ensure the raw datasets are in the data/ directory.
2. Run the Pipeline
Execute the pipeline by running:

bash
Copier le code
python main.py
This script performs the following steps:

Merges and cleans the datasets.
Preprocesses the data (handles outliers, missing values, and categorical variables).
Trains a linear regression model.
Saves cleaned data, metrics, and visualizations.
3. Outputs
Processed Dataset: data/dataset-preprocessed.csv
Model Evaluation: Metrics printed in the console.
Visualization: A scatter plot comparing predicted and real values saved in the graphs/ folder.
4. Example Dataset
Ensure your raw datasets have columns like:

Price, Property type, Living area, Building condition, Zip code, and more.
Refer to the project scripts for detailed dataset expectations.

Modules Overview
1. cleaning_datasets.py
Handles merging and cleaning of multiple datasets, renames columns, and ensures consistent formatting.

2. cleaning_feature_engineering.py
Includes functions for handling outliers, replacing missing values, and transforming categorical variables into numerical data.

3. linear_regression_model.py
Implements a linear regression model, evaluates it using metrics (MAE, RMSE, MAPE, R²), and generates visualizations.

Dependencies
Python 3.8+
pandas
numpy
scikit-learn
matplotlib
For detailed dependencies, see requirements.txt."# challenge-app-deployment-immo-eliza" 
