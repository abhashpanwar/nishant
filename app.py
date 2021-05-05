from flask import Flask, render_template, request

import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
AdaBoost_Regressor= AdaBoostRegressor(DecisionTreeRegressor())

import data as config

app = Flask(__name__)


@app.route("/")
def home():
    categories = {'manufacturer':[1,2,3],'condition':[1,2,3],
    'fuel':[1,2,3],'size':[1,2,3],'type':[1,2,3],'transmission':[1,2,3],'paint_color':[1,2,3]
    ,'type':[1,2,3],'drive':[1,2,3],'cylinders':[1,2,3]}

    categories = config.get_parameters()
    return render_template("index.html",type=0,data=categories)

@app.route("/predict",methods=['POST'])
def predict():
	
    import gzip
    #load model for scalling
    standardscaler= pickle.load(open(config.scaler_model, 'rb'))

    #load AdaBoost Regressor models
    mymodel = pickle.load(open(config.rf_model, 'rb'))

    #forms data
    year=int(request.form['year'])
    odometer=int(request.form['odometer'])
    year_odometer=pd.DataFrame(data=[[year,odometer]],columns=['year','odometer'])
    scaler_year_odometer = standardscaler.transform(year_odometer[['year','odometer']]).flatten()

    #define all features
    testcols=['year','manufacturer','condition','cylinders','fuel','odometer','title_status','transmission','drive','type','paint_color']

    #define features values
    testdata=[scaler_year_odometer[0],int(request.form['manufacturer']),int(request.form['condition'])
    ,int(request.form['cylinders'])
        ,int(request.form['fuel']),scaler_year_odometer[1],int(request.form['title_status']),int(request.form['transmission']),int(request.form['drive']),
        int(request.form['type']),int(request.form['paint_color'])]

    #creating daataframe
    test=pd.DataFrame(data=[testdata],columns=testcols)

    #predict value
    pred=mymodel.predict(test)[0]

    #price
    car_price = standardscaler.inverse_transform([[0],[0],[pred]]).flatten()[-1]

    categories = config.get_parameters()
    return render_template("index.html",type=1,price=car_price.round(2),data=categories)

if __name__ == "__main__":
    app.run(debug=True)
