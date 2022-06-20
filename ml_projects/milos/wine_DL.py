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
a = 1
wine_data = wine_data_raw.dropna(axis=0)
object_ids = wine_data['obj_ID']