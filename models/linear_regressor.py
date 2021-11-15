from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
import numpy as np


class LinearRegressor:

    def __init__(self, train, test):
        self.train = train
        self.test = test


    def predict(self):
        x_train = self.train.drop('Close', axis=1)
        y_train = self.train['Close']
        x_test = self.test.drop('Close', axis=1)
        regressor = LinearRegression()
        regressor.fit(x_train, y_train)
        pred = regressor.predict(x_test)
        return pred

    def Visualize(self, preds):
        self.test['Predictions'] = preds
        plt.figure(figsize=(16, 8))
        plt.plot(self.train['Close'])
        plt.plot(self.test[['Close', 'Predictions']])
