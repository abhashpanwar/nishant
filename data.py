import pandas as pd
rf_model = 'models/AdaBoost_Regressor_Tuned_Model.pkl'
scaler_model = 'models/Scaler.pkl'

def get_parameters():
    clean_dataset = pd.read_csv("UsedCarsDataset.csv")

    cylinders_ordinal = {'3 cylinders':1,'4 cylinders':2,'5 cylinders':3,'6 cylinders':4,'8 cylinders':5,'10 cylinders':6,'12 cylinders':7,'Missing':8,'other':9}
    condition_ordinal = {'salvage':1,'fair':2,'good':3,'excellent':4,'like new':5,'new':6,'Missing':7}

    manufacturer_nominal = {val:i+1 for i,val in enumerate(sorted(clean_dataset['manufacturer'].unique()))}
    model_nominal = {val:i+1 for i,val in enumerate(sorted(clean_dataset['model'].unique()))}
    fuel_nominal = {val:i+1 for i,val in enumerate(sorted(clean_dataset['fuel'].unique()))}
    title_status_nominal = {val:i+1 for i,val in enumerate(sorted(clean_dataset['title_status'].unique()))}
    transmission_nominal = {val:i+1 for i,val in enumerate(sorted(clean_dataset['transmission'].unique()))}
    drive_nominal = {val:i+1 for i,val in enumerate(sorted(clean_dataset['drive'].unique()))}
    type_nominal = {val:i+1 for i,val in enumerate(sorted(clean_dataset['type'].unique()))}
    paint_color_nominal = {val:i+1 for i,val in enumerate(sorted(clean_dataset['paint_color'].unique()))}

    categories = {'manufacturer':manufacturer_nominal,'model':model_nominal,
    'condition':condition_ordinal,'cylinders':cylinders_ordinal,'fuel':fuel_nominal,
    'title_status':title_status_nominal,'transmission':transmission_nominal
    ,'drive':drive_nominal,'type':type_nominal,'paint_color':paint_color_nominal}
    return categories
