import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from  sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application


ridge_model = pickle.load(open('models/ridge.pkl','rb'))
StandardScaler = pickle.load(open('models/scaler.pkl','rb'))



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict_datapoint():
    if request.method == 'POST':
        temperature = float(request.form.get('temperature'))
        rh = float(request.form.get('rh'))
        ws = float(request.form.get('ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        classes = float(request.form.get('classes'))
        Region = float(request.form.get('Region'))

        new_scaled_data = StandardScaler.transform([[temperature, rh, ws, Rain, FFMC, DMC, ISI, classes, Region]])
        result = ridge_model.predict(new_scaled_data)
        return render_template("home.html", results = result[0])
    else:
        return render_template("home.html")


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)


