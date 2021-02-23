# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 10:59:45 2021

@author: Akshatha Shetty
"""
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

dataset = pd.read_csv('Emp Salary.csv')

dataset['experience'].fillna(0, inplace=True)
dataset['test_score'].fillna(dataset['test_score'].mean(), inplace = True)

X = dataset.iloc[:, :3]

def convert_to_int(exp):
    exp_dict = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7,
                'eight':8, 'nine':9, 'ten':10, 'eleven':11, 'twelve':12, 'zero':0, 0:0}
    return exp_dict[exp]

X['experience'] = X['experience'].apply(lambda x : convert_to_int(x))

Y = dataset.iloc[:, -1]

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

#Since we have a small dataset will train iur model with all available data
#Fitting the model with training data
regressor.fit(X,Y)

#Saving trained model to disk
pickle.dump(regressor, open('lin_model.pkl', 'wb'))

#Loading model to predict result
model = pickle.load(open('lin_model.pkl','rb'))
print(model.predict([[2, 9, 6]]))