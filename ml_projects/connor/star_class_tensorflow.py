# Core imports
import numpy as np
# import scipy as sp
import pandas as pd

# Functionality imports
import os.path as osp # useful for joining filepaths
import matplotlib.pyplot as plt # plotting

from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import OneHotEncoder # OHEncoding used for categorical non-ordinal variables

from tensorflow import keras
from keras import layers
from keras import callbacks

# Link modules folder for functions and import functions
import sys
sys.path.append('modules')



# Declare data filepath and read into a variable using pandas DataFrame
star_data_fp = osp.join("resources", "datasets", "star_classification.csv")
star_data_raw = pd.read_csv(star_data_fp)

# -------------------------------------------------------------------------------------------
#                           ----------  PREPROCESSING  ----------
# -------------------------------------------------------------------------------------------
"""
Preprocessing step will go something like this:

1. Look for blatant anomalies
2. Decide what to do with missing data
3. Look at encoding, scaling etc

"""
star_data = star_data_raw.dropna(axis=0)

object_ids = star_data['obj_ID']
star_data = star_data[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z','class','redshift']]

star_data_minus_class = star_data.drop(columns = ['class'])

outlier_finder = LocalOutlierFactor()
y_pred = outlier_finder.fit_predict(star_data_minus_class)

# Create the anomaly_score
anomaly_score = outlier_finder.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = anomaly_score

# Determine the threshold that classifies an anomalous datapoint
# NOTE: 10 was chosen by inspection of the data, I have no idea what is appropriate generally
threshold = 10

# Apply the filter to the outlier scores
filter_func = abs(outlier_score["score"]) > threshold
outlier_indecies = outlier_score[filter_func].index.tolist()

print("\nElements that have been determined to be anomalous have indecies:")
print(outlier_indecies)

# Loop through indecies and drop the offensing datapoints
for ind in outlier_indecies:
    star_data = star_data.drop(star_data.index[ind], axis=0)


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









# Simple initial deep learning: Single hidden neuron, 128 inputs

input_shape = X_train_with_id.shape[1]

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=[input_shape]),
    layers.Dense(1)
])

model.compile(
    loss = "mae"
)

something = model.fit(
    X_train_with_id, y_train_with_id,
    batch_size=256,
    epochs=10
)

loss_data = something.history['loss']

# Data for plotting
epochs = np.arange(len(loss_data))

fig, ax = plt.subplots()
ax.plot(epochs, loss_data)

ax.set(xlabel='epochs', ylabel='loss',
       title='1 Hidden Neuron, Simple')
ax.grid()

fig.savefig("test.png")




# More complex deep learning: Multiple hidden neurons, 128 inputs

input_shape = [X_train_with_id.shape[1]]

early_stopping = callbacks.EarlyStopping(
    min_delta=0.001, # minimium amount of change to count as an improvement
    patience=20, # how many epochs to wait before stopping
    restore_best_weights=True,
)



model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=input_shape),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(1),
])

model.compile(
    optimizer = 'adam',
    loss = 'categorical_crossentropy',
    metrics=['catagorical_accuracy'],
)

something = model.fit(
    X_train_with_id, y_train_with_id,
    batch_size=256,
    epochs=100,
    callbacks=[early_stopping]
)

loss_data = something.history['loss']

# Data for plotting
epochs = np.arange(len(loss_data))

fig, ax = plt.subplots()
ax.plot(epochs, loss_data)

ax.set(xlabel='epochs', ylabel='loss',
       title='Multiple Hidden Neuron, Adam')
ax.grid()

fig.savefig("test_2.png")



