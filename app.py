# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 11:00:32 2021

@author: Akshatha Shetty
"""

import numpy as np
from flask import Flask, request, render_template, jsonify
import pickle

app = Flask(__name__, template_folder='template')
model = pickle.load(open('lin_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering result on html GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    output = round(prediction[0],2)
    
    return render_template('index.html', prediction_text = 'Employee salary should be $ {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)