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
star_data_fp = osp.join("resources", "datasets", "star_classification.csv")
star_data_raw = pd.read_csv(star_data_fp)

# -------------------------------------------------------------------------------------------
#                           ----------  PREPROCESSING  ----------
# -------------------------------------------------------------------------------------------


"""
First step is to assess the dataset. We are looking for:
    - Overall shape of the dataset
    - Catagorical vs Numerical data
    - Useful features and the target feature
    - Missing datapoints
"""

# star_data_raw.shape = (100000, 18) -> 100000 datapoints, 18 columns

# star_data_raw.dtypes = All interesting columns are floats
#   Some columns are int but they are all unimportant to the analysis (info on which equipment was used etc)
#   The target (class) is catagorical

# star_data_raw['class'].unique() = ['GALAXY', 'QSO', 'STAR'] -> Target feature has 3 discrete values present

# star_data_raw.isna().any().any() = False -> There are no missing datapoints at all

"""
From that quick analysis I conclude that:
    - No datapoints need be dropped
    - All float type features should be included
    - The target feature should be OneHotEncoded
    - All float values should be scaled
"""

# First, filter down to interesting feature columns only
star_data = star_data_raw[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z','class','redshift']]

# Break off the class column temporarily
star_data_features = star_data.drop(columns=['class'])

# Next, search for blatant anomalous datapoints
outlier_finder = LocalOutlierFactor()
y_pred = outlier_finder.fit_predict(star_data_features)

anomaly_score = outlier_finder.negative_outlier_factor_
outlier_score = pd.DataFrame()
outlier_score["score"] = anomaly_score

# Determine the threshold that classifies an anomalous datapoint
threshold = 10

# Apply the filter to the outlier scores
filter_func = abs(outlier_score["score"]) > threshold
outlier_indecies = outlier_score[filter_func].index.tolist()

print("\nElements that have been determined to be anomalous have indecies:")
print(outlier_indecies)

# Loop through indecies and drop the offensing datapoints
for ind in outlier_indecies:
    star_data = star_data.drop(star_data.index[ind], axis=0)


# Next, encode the catagorical target feature
ohe = OneHotEncoder()

encoded_quality = ohe.fit_transform(star_data[['class']])
new_encoded_col_names = ohe.categories_[0]

star_data[new_encoded_col_names] = encoded_quality.toarray()
star_data_enc = star_data.drop(columns=["class"])


# Scale all columns 0->1
max_ = star_data_enc.max(axis=0)
min_ = star_data_enc.min(axis=0)
star_enc_scaled = (star_data_enc - min_) / (max_ - min_)


# Split off a random selection of the data for validation, this is used when all is said and done to check the model
star_training = star_enc_scaled.sample(frac=0.7, random_state=0)
star_valid = star_enc_scaled.drop(star_training.index)


# Seperate target from features
# Here, X referes to features, y refers to target
X_train = star_training.drop(columns=new_encoded_col_names)
y_train = star_training[new_encoded_col_names]

X_valid = star_valid.drop(columns=new_encoded_col_names)
y_valid = star_valid[new_encoded_col_names]


# Time to develop a first attempt model as such:
# 2 hidden layers with 8 neurons and relu activation
# mae loss func
# adam optimizer
# An early stopping callback


early_stopping = callbacks.EarlyStopping(
    min_delta=0.001,
    patience=50,
    restore_best_weights=True,
)

input_shape = X_train.shape[1]


layer_depths = [32, 64, 128]
batch_sizes = [128, 256]
l_rates = [0.01, 0.001]

output_storage = []

for layer_depth in layer_depths:
    for l_rate in l_rates:

        model = keras.Sequential([
            layers.BatchNormalization(input_shape = [input_shape]),
            layers.Dense(layer_depth, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(layer_depth, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(layer_depth, activation='relu'),
            layers.Dropout(0.3),
            layers.BatchNormalization(),
            layers.Dense(3, activation = 'softmax'),
        ])

        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=l_rate),
            loss = 'categorical_crossentropy',
            metrics = ['categorical_accuracy']
        )

        for batch_size in batch_sizes:

            print("\n Batch Size: {} \n Layer depth: {}".format(batch_size, layer_depth))

            output = model.fit(
                X_train, y_train,
                validation_data=(X_valid, y_valid),
                batch_size=batch_size,
                epochs=500,
                callbacks=[early_stopping],
                verbose=0
            )

            ignore_first = 5

            output_df = pd.DataFrame(output.history)

            output_storage.append(dict(
                batch = batch_size,
                depth = layer_depth,
                rate = l_rate,
                acc = output_df['val_categorical_accuracy'].max(),
                loss = output_df['val_loss'].min()
            ))

best_5_combinations = sorted(output_storage, key=lambda x: x["acc"])[-5:]

print('Best 5 combinations are:')

for combo in best_5_combinations:
    print("Accuracy of {:0.4f} with batch of {}, {} layers and learning rate of {}\n". format(combo["acc"],combo["batch"],combo["depth"], combo["rate"]))









