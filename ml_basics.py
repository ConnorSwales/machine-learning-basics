"""
This file contains a walkthrough of the basics of ML using sklearn
Plenty of comments and log outputs
For use as reference for building scripts elsewhere in the code
"""

#import numpy as np
#import scipy as sp
import pandas as pd
#import sklearn as sk


# os.join joins together a filepath depending on operating system (different slashes for win/mac)
import os.path as osp


# Declare data filepath and read into a variable using pandas dataframe
star_data_fp = osp.join("resources", "datasets", "star_classification.csv")
star_data_raw = pd.read_csv(star_data_fp)


print(star_data_raw.describe())
"""
data.describe() -> returns an overview of the data including headings and other basic info

Headings: The collected data
Count:    The number of none NaN datapoints for that heading
...
The rest are intuitive
"""


print(star_data_raw.columns)
print("\n")

# Or more log-friendly way ...
for column in star_data_raw.columns:
    print(column)
print("\n")
"""
Prints the data headings, see basic_functions.print_info()
"""
