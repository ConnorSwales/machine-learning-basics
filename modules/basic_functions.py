"""
Here we store basic functions for use in our code

To access them, simply import this file into your script:
    - import basic_functions as funcs

then to call the functions:
    - funcs.a_stored_function(my_data)
"""

# import numpy as np
# import scipy as sp
# import pandas as pd
# import sklearn as sk


def print_info(data):

    print(data.describe())
    print("\n")
    print("Data Columns:")
    print("\n")

    for column in data.columns:
        print(column)