# Import necessary library
import pickle
import pandas as pd

def predict(data: pd.DataFrame) -> float:
    """
    Function to predict a price based on preprocessed input data
    :param: input data of the property
    :return: the predicted price
    """
    # Scaling the data
    with open("predict/scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    scaled_data = scaler.transform(data)
    # Loading the model to predict on the data
    pickle_in = open("predict/regression.pkl", "rb")
    regression = pickle.load(pickle_in)
    prediction = regression.predict(scaled_data)[0]
    return prediction