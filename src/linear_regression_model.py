# Import libraries & dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class LinearRegressionModel:
    """ 
    Class to create the linear regression model & compute the metrics
    """
    def __init__(self, df: pd.DataFrame, X, y):
        """
        Initialize the LinearRegression class with the dataset.
        :param df: dataframe containing features and the target variable.
        :param X: features set
        :param y: target values corresponding to the features
        """
        # Dataframe, features (X), target(y)
        self.df = df
        self.X = X
        self.y = y

    def create_linear_model(self):
        """
        Function that creates the train & test datasets, the linear regression model and get the metrics
        """
        # Split the dataset into training & testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        # Standaridization of the features
        X_train, X_test = self.standardization_values(X_train, X_test)
        # Create and train the model
        regression = LinearRegression()
        regression.fit(X_train, y_train)
        # Make predictions
        prediction_test = regression.predict(X_test)
        prediction_train = regression.predict(X_train)
        # Gettings the metrics & print them
        print('Training metrics:')
        self.score_model(regression, X_train, y_train)
        self.mae_metric(y_train, prediction_train)
        self.rmse_metric(y_train, prediction_train)
        self.mape_metric(y_train, prediction_train)
        print('Testing metrics:')
        self.score_model(regression, X_test, y_test)
        self.mae_metric(y_test, prediction_test)
        self.rmse_metric(y_test, prediction_test)
        self.mape_metric(y_test, prediction_test)
        # Save comparison of test into a CSV and a plot
        self.comparison_test_prediction(y_test, prediction_test)

    def standardization_values(self, X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        """
        Function that standardize the feature values in training and test sets
        :param X_train: training set for the model
        :param X_test: testing set for the model
        :return: standardized training and testing datasets
        """
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test

    def score_model(self, model: LinearRegression, features: np.ndarray, target: pd.Series):
        """
        Function that calculates and prints the R2 score of the model
        :param features: feature dataset
        :param target: target values corresponding to the features
        """
        score = round(model.score(features, target),2)
        print(f'Score: {score}') 
    
    def mae_metric(self, target: pd.Series, prediction: np.ndarray):
        """
        Function that calculates and prints the mean absolute error (MAE)
        :param target: target values corresponding to the features
        :param prediction: predicted values
        """
        mae = round(mean_absolute_error(target, prediction),2)
        print(f'MAE: {mae}')
      
    def rmse_metric(self, target: pd.Series, prediction: np.ndarray):
        """
        Function that calcultes and prints the mean squared error (MSE) and the root MSE (RSME)
        :param target: target values corresponding to the features
        :param prediction: predicted values
        """
        mse = mean_squared_error(target, prediction)
        rmse = round(np.sqrt(mse),2)
        print(f'RMSE: {rmse}')

    def mape_metric(self, target: pd.Series, prediction: np.ndarray):
        """
        Function that calcultes and prints the mean absolute percentage error (MAPE)
        :param target: target values corresponding to the features
        :param prediction: predicted values
        """
        mape = round(np.mean(np.abs((target - prediction)/target)),2)
        print(f'MAPE: {mape}')

    def comparison_test_prediction(self, y_test: pd.Series, prediction: np.ndarray):
        """
        Function that saves a comparison of the true and predicted values to a CSV 
        and generates a scatter plot
        :param y_test: actual target values from the testing test
        :param prediction: predicted values
        """
        comparison = pd.DataFrame({'Real values':y_test, 'Predicted values':prediction})
        comparison.to_csv('./data/comparison.csv')
        plt.figure(figsize=(15, 10))
        x = y_test
        y = prediction
        plt.scatter(x,y)
        m, b = np.polyfit(x, y, 1)
        plt.plot(x, m*x+b, color='red')
        plt.xlabel('Real values')
        plt.ylabel('Predicted values')
        plt.title('Predicted vs real values')
        plt.savefig(f'./graphs/prediction-vs-testdata.png')