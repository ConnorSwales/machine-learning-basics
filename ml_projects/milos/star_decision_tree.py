# project: use big data ?? to find penis enlargening chemicals

# import numpy as np
# import scipy as sp
import sys
sys.path.append('modules')

# Core imports
import numpy as np
import scipy as sp
import pandas as pd

# Sklearn imports
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder # OHEncoding used for categorical non-ordinal variables
from sklearn.metrics import mean_absolute_error

# Functionality imports
import os.path as osp # useful for joining filepaths
import matplotlib.pyplot as plt # plotting
import basic_functions as funcs

# Declare data filepath and read into a variable using pandas DataFrame
star_data_fp = osp.join("resources", "datasets", "star_classification.csv")
star_data_raw = pd.read_csv(star_data_fp)

#funcs.print_info(star_data_raw)
star_data = star_data_raw.dropna(axis=0)
object_ids = star_data['obj_ID']
star_data = star_data[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z','class','redshift']]

#print(star_data.head(n=10))
print(star_data.isnull().sum()) #make sure there is no missing entries

s = (star_data.dtypes == 'object')
object_cols = list(s[s].index)
print("Categorical variables:")
print(object_cols) # check which column takes categorical variable entries

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 
OH_cols_star_data = pd.DataFrame(OH_encoder.fit_transform(star_data[object_cols]))
OH_cols_star_data.index = star_data.index

num_star_data = star_data.drop(object_cols, axis=1)
OH_star_data = pd.concat([num_star_data, OH_cols_star_data], axis=1)
#print(OH_star_data.head(n=10)) 
#here we have replaced categorical variables
# GALAXY -> 0, 
# QSO -> 1, 
# STAR -> 2

X = OH_star_data.drop(columns=[0,1,2])
print(X.head(n=10))
y = pd.DataFrame([OH_star_data[0], OH_star_data[1], OH_star_data[2]]).transpose() #need better way to do this
print(y.head(n=10))

X_train_with_id, X_val_with_id, y_train_with_id, y_val_with_id = train_test_split(
        X,
        y,
        random_state=0,
    )

star_model = DecisionTreeRegressor()
star_model.fit(X_train_with_id, y_train_with_id)
val_predictions = star_model.predict(X_val_with_id)
print(mean_absolute_error(y_val_with_id, val_predictions))