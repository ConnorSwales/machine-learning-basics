# Core imports
import numpy as np
import scipy as sp
import pandas as pd

# Sklearn imports
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
#from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import OneHotEncoder # OHEncoding used for categorical non-ordinal variables
from sklearn.metrics import mean_absolute_error

from tensorflow import keras
from keras import layers
from keras import callbacks
from keras import metrics
from keras import optimizers


# Functionality imports
import os.path as osp # useful for joining filepaths
import matplotlib.pyplot as plt # plotting
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')


# Link modules folder for functions and import functions
import sys
sys.path.append('modules')


# Declare data filepath and read into a variable using pandas DataFrame
vent_train_fp = osp.join("resources", "datasets", "timeseries", "ventilator-pressure-prediction", "train.csv")
vent_train_raw = pd.read_csv(vent_train_fp)

vent_test_fp = osp.join("resources", "datasets", "timeseries", "ventilator-pressure-prediction", "test.csv")
vent_test_raw = pd.read_csv(vent_test_fp)


"""
First step is to assess the dataset. We are looking for:
    - Overall shape of the dataset
    - Catagorical vs Numerical data
    - Useful features and the target feature
    - Missing datapoints
"""

# vent_train_raw
"""
There are 6036000 rows, 1-indexed, each with their own id
Input features are:
    - R : number 0-100 -> attribute of the lung, how restricted the airway is
    - C : number 0-100 -> attribute of the lung, how compliant the lung is

    - u_in : number 0-100, represents a tunable input var (percentage the inspiratory solenoid valve is open)
    - u_out : binary 0 or 1, represents a tunable input var (whether the exploratory valve is open (1) or closed (0))

    - pressure : number > 0 -> the target, airway pressure
"""

# vent_train_raw[vent_train_raw["breath_id"] == 1]
# vent_train_raw.isna().any().any()
"""
These revealed that:
    - for each unique breath, there are 80 rows, with a time_step from 0 to ~3s
    - there is no missing data (NaNs)
    - there are some 'breath_id's missing though (e.g: breath 8 is not present)
"""


"""
Intuition from this is:

Since breath_id is not a feature, seems like the train data should be dealt with in chunks for each breath
"""

def engineer_features(df):

    # Firstly, only the inhalation period is of interest
    df = df[df.u_out == 0].drop(columns=["u_out"])

    # Norm features to 0-1


    return df

vent_train = engineer_features(vent_train_raw)
vent_test = engineer_features(vent_test_raw)

# vent_train.drop(columns=["id", "breath_id"], inplace=True)
# vent_test.drop(columns=["id", "breath_id"], inplace=True)





# Need to work with an extremely reduced version of the problem first I think, going to split out purely the first breath
vent_first_train = vent_train[vent_train["breath_id"] == 1]
vent_first_test = vent_test[vent_test["breath_id"] == 1]


press_first_train = vent_first_train["pressure"]
vent_first_train.drop(columns=["id", "breath_id", "pressure"], inplace=True)

press_first_test = vent_first_test["pressure"]
vent_first_test.drop(columns=["id", "breath_id", "pressure"], inplace=True)


model = keras.Sequential([
                    layers.LSTM(80, input_shape=(vent_first_train.shape[1:]), return_sequences=True),
                    layers.Dropout(0.2),
                    layers.LSTM(128),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(0.2),
                    layers.Dense(80, activation='softmax')
])


model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer=keras.optimizers.Adam(lr=0.001, decay=1e-6),
    metrics=['accuracy']
)

#Fitting the data to the model
model.fit(
    vent_first_train,
    press_first_train,
    epochs=1,
    validation_data=(vent_first_test, press_first_test)
)

a=1