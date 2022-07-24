# Core imports
import numpy as np
import scipy as sp
import pandas as pd


# Functionality imports
import os.path as osp # useful for joining filepaths
import matplotlib.pyplot as plt # plotting

from sklearn.metrics import mean_squared_error

# load dataset
def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')

shampoo_fp = osp.join("resources", "datasets", "timeseries", "shampoo_uni.csv")
series = pd.read_csv(shampoo_fp, header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)

SUPPRESS_PLOTS = True

# summarize first few rows
print(series.head())

if not SUPPRESS_PLOTS:
    # line plot
    series.plot()
    plt.show()


# split data into train and test
X = series.values
train, test = X[0:-12], X[-12:]

# walk-forward validation
history = [x for x in train]
predictions = list()
for i in range(len(test)):
    # make prediction
    predictions.append(history[-1])
    # observation
    history.append(test[i])
# report performance
rmse = np.sqrt(mean_squared_error(test, predictions))
print('RMSE: %.3f' % rmse)
if not SUPPRESS_PLOTS:
    # line plot of observed vs predicted
    plt.plot(test)
    plt.plot(predictions)
    plt.show()


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df

supervised = timeseries_to_supervised(X)


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pd.Series(diff)


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


# transform to be stationary
differenced = difference(series, 1)
print(differenced.head())
# invert transform
inverted = list()
for i in range(len(differenced)):
    value = inverse_difference(series, differenced[i], len(series)-i)
    inverted.append(value)
inverted = pd.Series(inverted)
print(inverted.head())

a=1