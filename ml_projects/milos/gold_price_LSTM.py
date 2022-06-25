# Perform all imports and read-ins

import numpy as np
import scipy as sp
import pandas as pd

from tensorflow import keras
from keras import layers
from keras import callbacks
from keras import metrics
from keras import optimizers

import os.path as osp # useful for joining filepaths
import matplotlib.pyplot as plt # plotting
plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('animation', html='html5')

gold_raw = pd.read_csv('monthly_csv.csv', index_col = 'Date')
gold_raw.head(n = 5)

#Define how we will split our sequence of gold prices into multiple samples to feed into the model:
n_steps1 = 100
predNum = 1
n_steps = n_steps1 - predNum

def sampling(sequence, n_steps):
  X, Y = list(), list()
  for i in range(len(sequence)):
    sam = i+n_steps
    if sam > len(sequence)-predNum:
      break 

    x, y = sequence[i:sam-predNum+1], sequence[sam]#sequence[sam-predNum+2:sam+2]
    X.append(x)
    Y.append(y)
  return np.array(X), np.array(Y)


X, Y = sampling(gold_raw['Price'].tolist(), n_steps)

for i in [1,2,3,4,5,6,7,8,9,10]:
  print(X[i], Y[i])
  print(i)

endPoint = 700
print('Range is', range(len(X)))
X_train = X[:endPoint]
# print(X_train.shape)
X_test = X[endPoint:]
# print(X_test.shape)
y_train = Y[:endPoint]
y_test = Y[endPoint:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

#Create the model and define the layers. Using TimeDistributed Conv1D LSTM here
n_features = 1
sub_seq = 2
# model = keras.Sequential([
#     layers.LSTM(50, activation = 'relu', input_shape = (n_steps, n_features)),
#     layers.Dense(1)                    
# ])

model = keras.Sequential([
                    layers.TimeDistributed(layers.Conv1D(filters = 64, kernel_size = 1, activation = 'relu'), input_shape = (None, n_steps, n_features)),
                    layers.TimeDistributed(layers.MaxPooling1D(pool_size=2)),
                    layers.TimeDistributed(layers.Flatten()),
                    layers.LSTM(25, activation = 'relu'),
                    layers.Dense(1)
])
Adam2 = keras.optimizers.Adam(learning_rate = 1e-4)
model.compile(optimizer = Adam2, loss = 'huber_loss', metrics = 'poisson')

#X needs to be reshaped to enter our Conv1D layer
sub_seq = 1
X_train = X_train.reshape((X_train.shape[0], sub_seq, X_train.shape[1], 1))
# model.summary()
X_train.shape

#Define early stopping, even though it's not actually used in time series?
early_stopping = callbacks.EarlyStopping(
    min_delta = 0.001,
    patience = 100, 
    restore_best_weights = True,
)

history = model.fit(
       X_train,y_train, 
       validation_split = 0.2, 
       batch_size = 8, 
       epochs=200, 
       verbose = 1, 
       callbacks = [early_stopping], 
       shuffle='True')

#plot results of loss and metric
history_df = pd.DataFrame(history.history)
history_df.loc[0:, ['poisson', 'val_poisson']].plot(xlabel='epoch', ylabel='mae')
history_df.loc[0:, ['loss', 'val_loss']].plot(xlabel='epoch', ylabel='loss')

#Predict further steps:
X_test2 = X_test.reshape((X_test.shape[0], sub_seq, X_test.shape[1], 1))
X_testSave = X_test2
X_test2 = X_testSave.copy()
print(X_test2[1,0,:,0])


# dummy_predictions = np.ones(X_test2.shape[0])
predictions = model.predict(X_test2, verbose=0)
number_of_steps = 50
predictOne = np.zeros(number_of_steps)
predictOne[0] = predictions[0]

predictFifty = np.zeros(number_of_steps)

# for each analysis step
for ii in range(number_of_steps):

  # for each sample (there are 97 of them)
  for i in range(X_test2.shape[0]):
    temp_arr = X_test2[i, 0, :, 0]
    # create a temporary sample, add the pred and drop the first
    temp_arr = np.append(temp_arr, predictions[i])
    temp_arr = np.delete(temp_arr, [0])
    # Place that temp arr back into the whole datastruct
    X_test2[i, 0, :, 0] = temp_arr
    


  # now that each of the 97 samples has had it's start chopped and pred added...
  new_predictions = model.predict(X_test2, verbose=0) 

  predictions = new_predictions
  predictOne[ii] = predictions[0]
  predictFifty[ii] = predictions[50]
print(X_test2[1,0,:,0])

gPrice = gold_raw['Price'].tolist()
gCheck = gPrice[endPoint+40:endPoint+50]
# rangeX = np.arange(number_of_steps,len(predictions)+number_of_steps)
predictComb = np.append(predictOne, predictFifty)
predictOne_df = pd.DataFrame(predictOne)
predictComb_df = pd.DataFrame(predictComb)
# predictions_df = pd.DataFrame(predictions)
# predictions_df = predictions_df.set_index(rangeX)
y_df = pd.DataFrame(y_test)
# y_df.plot()
# plt.plot(predictions_df, 'blue')
# plt.plot(predictOne_df, 'blue')
plt.plot(predictComb_df, 'blue')
plt.plot(y_df, 'orange')
plt.plot(np.arange(-len(gCheck),0),gCheck,'green')
plt.show
print(X_test2[1,0,:,0])