# project: use big data ?? to find penis enlargening chemicals

# import numpy as np
# import scipy as sp
import sys
sys.path.append('modules')

# Core imports
import numpy as np
import scipy as sp
import pandas as pd

a = 1
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

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

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

temp_X = OH_star_data.drop(columns=[0,1,2])
scaler = MinMaxScaler(feature_range=(-1,1))

scaler.fit(temp_X)
X = pd.DataFrame(scaler.fit_transform(temp_X))

print(X.head(n=10))
y = pd.DataFrame([OH_star_data[0], OH_star_data[1], OH_star_data[2]]).transpose() #need better way to do this
print(y.head(n=10))

# X_train_with_id, X_val_with_id, y_train_with_id, y_val_with_id = train_test_split(
#         X,
#         y,
#         random_state=0,
#     )


a =1
input_shape = [X.shape[1]]
model_DL = keras.Sequential([
    layers.BatchNormalization(input_shape = input_shape),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation = 'softmax'),
])

model_DL.compile(
    optimizer='adam',
    loss='categorical_hinge',
    metrics = ['categorical_accuracy']
)
early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=5, # how many epochs to wait before stopping
    restore_best_weights=True,
)
EPOCHS = 30
history = model_DL.fit(
    X, y,
    validation_split = 0.2,
    batch_size=32,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1,
)
a = 1

history_df = pd.DataFrame(history.history)
# Start the plot at epoch 5
history_df.loc[1:, ['categorical_accuracy', 'val_categorical_accuracy']].plot(xlabel='epoch', ylabel='categorical accuracy')
history_df.loc[0:, ['loss', 'val_loss']].plot(xlabel='epoch', ylabel='loss')
plt.show 
print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_categorical_accuracy'].max()))