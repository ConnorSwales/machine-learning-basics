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
wine_data_fp = osp.join("resources", "datasets", "winequality-red.csv")
wine_data_raw = pd.read_csv(wine_data_fp)

"""
First step is to assess the dataset. We are looking for:
    - Overall shape of the dataset
    - Catagorical vs Numerical data
    - Useful features and the target feature
    - Missing datapoints
"""

# wine_data_raw.shape = (1599,12) -> 1599 datapoints, 12 columns

# wine_data_raw.dtypes = All columns floats except quality (the target) which is int

# sorted(wine_data_raw.quality.unique()) = [3,4,5,6,7,8] -> Target feature has 6 discrete values present
# does this mean we encode just these or all available numbers 1-10?

# wine_data_raw.isna().any().any() = False -> There are no missing datapoints at all

"""
From that quick analysis I conclude that:
    - No datapoints need be dropped
    - All 11 features should be included
    - The target feature should be OneHotEncoded
    - All float values should be scaled
"""

# Initialize OHE, fit it to the target feature, add the new features to the dataset and remove the old one
ohe = OneHotEncoder()

encoded_quality = ohe.fit_transform(wine_data_raw[['quality']])
new_encoded_col_names = ohe.categories_[0]

wine_data_raw[new_encoded_col_names] = encoded_quality.toarray()
wine_data_enc = wine_data_raw.drop(columns=["quality"])
# NOTE: We have only encoded the possibility of 3,4,5,6,7,8, not 1,2,9,10, this may have reprecussions


# Scale all columns 0->1
max_ = wine_data_enc.max(axis=0)
min_ = wine_data_enc.min(axis=0)
wine_enc_scaled = (wine_data_enc - min_) / (max_ - min_)


# Split off a random selection of the data for validation, this is used when all is said and done to check the model
wine_training = wine_enc_scaled.sample(frac=0.7, random_state=0)
wine_valid = wine_enc_scaled.drop(wine_training.index)


# Seperate target from features
# Here, X referes to features, y refers to target
X_train = wine_training.drop(columns=new_encoded_col_names)
y_train = wine_training[new_encoded_col_names]

X_valid = wine_valid.drop(columns=new_encoded_col_names)
y_valid = wine_valid[new_encoded_col_names]


# Time to develop a first attempt model as such:
# 2 hidden layers with 8 neurons and relu activation
# mae loss func
# adam optimizer
# An early stopping callback


early_stopping = callbacks.EarlyStopping(
    min_delta=0.005,
    patience=10,
    restore_best_weights=True,
)

input_shape = X_train.shape[1]


layer_depths = [32,64,128, 256]
batch_sizes = [4 ,8, 16]
l_rates = [0.01, 0.001, 0.0001]

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
            layers.Dense(6, activation = 'softmax'),
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
                epochs=100,
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














# output_df.loc[ignore_first:, ['categorical_accuracy', 'val_categorical_accuracy']].plot(xlabel='epoch', ylabel='categorical accuracy')
# output_df.loc[ignore_first:, ['loss', 'val_loss']].plot(xlabel='epoch', ylabel='loss')
# plt.show

# print("Best Validation Loss: {:0.4f}".format(output_df['val_loss'].min()))
# print("Best Validation Accuracy: {:0.4f}".format(output_df['val_categorical_accuracy'].max()))


# loss_data = output.history['loss'][ignore_first:]
# val_loss = output.history['val_loss'][ignore_first:]

# # Data for plotting
# epochs = np.arange(len(loss_data)) + ignore_first

# fig, ax = plt.subplots()
# ax.plot(epochs, loss_data)
# ax.plot(epochs, val_loss)

# ax.set(xlabel='epochs', ylabel='loss',
#     title='')
# ax.grid()

# fig.savefig("ml_projects/connor/images/test_{}.png".format(batch_size))
