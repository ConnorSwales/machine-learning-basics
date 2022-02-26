# Core imports
import numpy as np
#import scipy as sp
import pandas as pd

# Sklearn imports
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree

# Functionality imports
import os.path as osp # useful for joining filepaths
import matplotlib.pyplot as plt # plotting


# Link modules folder for functions and import functions
import sys
sys.path.append('modules')

# Declare data filepath and read into a variable using pandas DataFrame
star_data_fp = osp.join("resources", "datasets", "star_classification.csv")
star_data_raw = pd.read_csv(star_data_fp)

# -------------------------------------------------------------------------------------------
#                    ---------- Extracting Simple Information ----------
# -------------------------------------------------------------------------------------------

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


# -------------------------------------------------------------------------------------------
#                      ---------- Simple ML Treatment ----------
# -------------------------------------------------------------------------------------------


star_data["obj_ID"] = object_ids
star_data = star_data.drop_duplicates(subset="obj_ID")

X = star_data.drop(columns=["class"])
y = star_data["class"]


def decision_tree_tester(X, y, *, test_size=None, max_leaf_nodes=None):

    X_train_with_id, X_val_with_id, y_train_with_id, y_val_with_id = train_test_split(
        X,
        y,
        random_state=0,
        test_size=test_size,
    )

    X_train = X_train_with_id.drop(columns=["obj_ID"])
    X_val = X_val_with_id.drop(columns=["obj_ID"])
    y_train = y_train_with_id.drop(columns=["obj_ID"])

    # Define model. Specify a number for random_state to ensure same results each run
    _model = DecisionTreeClassifier(random_state=0, max_leaf_nodes=max_leaf_nodes)

    # Fit model
    _model = _model.fit(X_train, y_train)

    # Take the actual classes column from the original data...
    classes = star_data["class"][star_data.obj_ID.isin(X_val_with_id["obj_ID"])].to_frame()
    classes["predicted_class"] = _model.predict(X_val)
    correct_guesses = classes[classes["class"] == classes["predicted_class"]]
    percentage_correct = correct_guesses.shape[0]/classes.shape[0] * 100

    return percentage_correct


# correctness = {
#     "size": [],
#     "result": []
# }

# sizes = np.arange(0.001, 0.1, 0.001)
# for size in sizes:
#     _result = decision_tree_tester(X, y, test_size=size)
#     correctness["size"].append(size)
#     correctness["result"].append(_result)


# max_index = np.argmax(correctness["result"])
# min_index = np.argmin(correctness["result"])
# print("\nTest Size Analysis:")
# print(f"\nThe best result was {correctness['result'][max_index]}%, at a test size of {correctness['size'][max_index]}")
# print(f"\nThe worst result was {correctness['result'][min_index]}%, at a test size of {correctness['size'][min_index]}")

# best_size = correctness['size'][max_index]

best_size = 0.019

# correctness = {
#     "nodes": [],
#     "result": []
# }

# for node in [2, 3, 4, 5, 10, 50, 200, 500, 1000]:
#     _result = decision_tree_tester(X, y, test_size=best_size, max_leaf_nodes=node)
#     correctness["nodes"].append(node)
#     correctness["result"].append(_result)


# max_index = np.argmax(correctness["result"])
# min_index = np.argmin(correctness["result"])
# print("\nLeaf Nodes Analysis:")
# print(f"\nThe best result was {correctness['result'][max_index]}%, at node number {correctness['nodes'][max_index]}")
# print(f"\nThe worst result was {correctness['result'][min_index]}%, at node number {correctness['nodes'][min_index]}")

best_nodes = 2


# plt.figure()
# plot_tree(
#     _model,
#     filled=True,
#     feature_names=X.feature_names,
#     class_names=X.target_names
# )
# plt.show()