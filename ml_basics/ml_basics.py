# Core imports
#import numpy as np
#import scipy as sp
import pandas as pd
#import sklearn as sk
import sklearn.neighbors as sk_nb

# Functionality imports
import matplotlib.pyplot as plt # plotting
import seaborn as sns # more statistical plotting
import os.path as osp # useful for joining filepaths

# Link modules folder for functions and import functions
import sys
sys.path.append('modules')
import basic_functions as funcs


"""
This file contains a walkthrough of the basics of ML using sklearn
Plenty of comments and log outputs
For use as reference for building scripts elsewhere in the code
"""

# Declare data filepath and read into a variable using pandas DataFrame
star_data_fp = osp.join("resources", "datasets", "star_classification.csv")
star_data_raw = pd.read_csv(star_data_fp)


# -------------------------------------------------------------------------------------------
#                           ---------- View raw data ----------
# -------------------------------------------------------------------------------------------


# Print overview of data
print("\n Star Data Overview: \n")
print(star_data_raw.describe())
"""
data.describe() -> returns an overview of the data including headings and other basic info

Headings: The collected data
Count:    The number of none NaN datapoints for that heading
...
"""

# Print all of the column headings
print("\n Star Data Columns: \n")
print(star_data_raw.columns)

# Or more log-friendly way ...
print("\n Star Data Columns (clearer): \n")
for column in star_data_raw.columns:
    print(column)
print("\n")



# OR, do all of the above with a premade function
# funcs.print_info(star_data_raw)



# -------------------------------------------------------------------------------------------
#                    ---------- Extracting Simple Information ----------
# -------------------------------------------------------------------------------------------


# Intention here is to pull out some key info about the dataset and maybe make some plots.
# No machine learning tactics are to be employed until the next section

# Clean out the dataset by dropping entries with NaN entries
# axis=0 -> drops rows with missing columns (removes the whole entry if one datapoint missing)
# axis=1 -> drops columns with missing rows (removes entire datapoint if there's a single missing entry)
star_data = star_data_raw.dropna(axis=0)

# The star_data has many columns that describe meta-data things like run_ID, camera information, id numbers etc
# Remove all of these to leave interesting columns
star_data = star_data[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z','class','redshift']]
funcs.print_info(star_data)

# star_data has 3 possible values for the column 'class'
    # - GALAXY
    # - QSO (Quasar)
    # - STAR
# This appears to be the best choice for our target for the ML algorithms down the line
# i.e: is this object a star, quasar or galaxy? (based on *some* data)


# Make a seperate DataFrame for each object class
galaxy_df = star_data[star_data["class"] == "GALAXY"]
galaxy_df = galaxy_df.drop(columns=["class"])

print("\nGalaxies:")
print("\n    Info:")
print(galaxy_df.describe())
print("\n    Head:")
print(galaxy_df.head(5))

qso_df = star_data[star_data["class"] == "QSO"]
qso_df = qso_df.drop(columns=["class"])
print("\nQuasars: ")
print("\n    Info:")
print(qso_df.describe())
print("\n    Head:")
print(qso_df.head(5))


star_df = star_data[star_data["class"] == "STAR"]
star_df = star_df.drop(columns=["class"])
print("\nStars: ")
print("\n    Info:")
print(star_df.describe())
print("\n    Head:")
print(star_df.head(5))


# Function to make a set of scatter diagrams
def each_class_subplot(galaxy_df, qso_df, star_df, *, title=None, x, y, color, filename):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    if title:
        fig.suptitle(title)

    sns.scatterplot(
        data=galaxy_df,
        ax=axes[0],
        x=x,
        y=y,
        color=color
    )
    axes[0].set_title("Galaxies")

    sns.scatterplot(
        data=qso_df,
        ax=axes[1],
        x=x,
        y=y,
        color=color
    )
    axes[1].set_title("Quasars")

    sns.scatterplot(
        data=star_df,
        ax=axes[2],
        x=x,
        y=y,
        color=color
    )
    axes[2].set_title("Stars")

    plt.savefig(osp.join("ml_basics", "outputs", f"{filename}.png"))


each_class_subplot(
    galaxy_df,
    qso_df,
    star_df,
    title="Alpha vs. Redshift",
    x="alpha",
    y="redshift",
    color="r",
    filename="alpha_red"
)

each_class_subplot(
    galaxy_df,
    qso_df,
    star_df,
    title="Green vs. Red Filter",
    x="g",
    y="r",
    color="g",
    filename="green_red_filt_anom"
)





# -------------------------------------------------------------------------------------------
#                      ---------- Scrub Anomalous Data ----------
# -------------------------------------------------------------------------------------------


# If you look at the image 'green_red_filt', it is clear that there is an anomalous datapoint
# in the 'stars' panel. Let's try get rid of it

# Sklearn package has a submodule named 'neighbors', all to do with applying nearest neighbors algorithms
# LocalOutlierFactor() assigns each datapoint a 'score' to do with how well it fits in with the other nearby points
# NOTE: this func takes an optional arg 'n_neighbors'. Worth playing around with that to see how it effects things

outlier_finder = sk_nb.LocalOutlierFactor()
y_pred = outlier_finder.fit_predict(star_df)

# Create the anomaly_score
anomaly_score = outlier_finder.negative_outlier_factor_

# Create an empty DataFrame and put the anomaly_score array into it
outlier_score = pd.DataFrame()
outlier_score["score"] = anomaly_score

# Determine the threshold that classifies an anomalous datapoint
# NOTE: 10 was chosen by inspection of the data, I have no idea what is appropriate generally
threshold = 10

# filter_func makes a 'filter' of True/False depending on the right hand side
filter_func = abs(outlier_score["score"]) > threshold

# Apply the filter to the outlier scores
outlier_indecies = outlier_score[filter_func].index.tolist()

print("Elements that have been determined to be anomalous have indecies:")
print(outlier_indecies)

# Loop through indecies and drop the offensing datapoints
for ind in outlier_indecies:
    star_df = star_df.drop(star_df.index[ind], axis=0)

# Make another plot to check the anomalies have gone
each_class_subplot(
    galaxy_df,
    qso_df,
    star_df,
    title="Green vs. Red Filter",
    x="g",
    y="r",
    color="g",
    filename="green_red_filt_fixed"
)
