# Core imports
import numpy as np
import scipy as sp
import pandas as pd

# Functionality imports
import os.path as osp # useful for joining filepaths
import matplotlib.pyplot as plt # plotting

from sklearn.neighbors import LocalOutlierFactor


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

class_mapping = {
    "GALAXY": 0,
    "QSO": 1,
    "STAR": 2
}

class_mapping_back = {k:v for v,k in class_mapping.items()}

star_data["class"] = [
    class_mapping["GALAXY"] if i == "GALAXY"
    else class_mapping["QSO"]  if i == "QSO"
    else class_mapping["STAR"]
    for i in star_data["class"]
]

outlier_finder = LocalOutlierFactor()
y_pred = outlier_finder.fit_predict(star_data)

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



star_data