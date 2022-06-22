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


wine_qual_fp = osp.join("resources", "datasets", "winequality-red.csv")
wine_data_raw = pd.read_csv(wine_qual_fp)

wine_data = wine_data_raw.dropna(axis=0)

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False) 
OH_cols_wine_data = pd.DataFrame(OH_encoder.fit_transform(wine_data[['quality']]))
OH_cols_wine_data.index = wine_data.index 
y = OH_cols_wine_data
X_prescale = wine_data.drop('quality', axis=1)

# X.min() to check if there are any negative values - there aren't, so we can use
scaler = MinMaxScaler(feature_range=(0,1))
scaler.fit(X_prescale)

X = pd.DataFrame(scaler.fit_transform(X_prescale))


# begin the model
input_shape = [X.shape[1]]
model_DL = keras.Sequential([
    layers.BatchNormalization(input_shape = input_shape),
    layers.Dense(32, activation='relu'),
    #layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(32, activation='relu'),
    #layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(6, activation = 'softmax'),
])
Adam_2 = keras.optimizers.Adam(learning_rate=1e-4)

model_DL.compile(
    optimizer=Adam_2,
    loss='categorical_crossentropy',
    metrics = ['categorical_accuracy']
)

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=500, # how many epochs to wait before stopping
    restore_best_weights=True,
)
EPOCHS = 500
history = model_DL.fit(
    X, y,
    validation_split = 0.2,
    batch_size=1,
    epochs=EPOCHS,
    callbacks=[early_stopping],
    verbose=1,
    shuffle = True
)
history_df = pd.DataFrame(history.history)
a = 1
# Start the plot at epoch 5
history_df.loc[0:, ['categorical_accuracy', 'val_categorical_accuracy']].plot(xlabel='epoch', ylabel='categorical accuracy')
history_df.loc[0:, ['loss', 'val_loss']].plot(xlabel='epoch', ylabel='loss')
print(("Best Validation Loss: {:0.4f}" +\
      "\nBest Validation Accuracy: {:0.4f}")\
      .format(history_df['val_loss'].min(), 
              history_df['val_categorical_accuracy'].max()))
