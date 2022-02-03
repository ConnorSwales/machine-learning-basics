# project: use big data ?? to find penis enlargening chemicals

# import numpy as np
# import scipy as sp
import sys
sys.path.insert(1, 'modules')

import pandas as pd
import sklearn as sk

import os.path as osp

import basic_functions as funcs

star_data_fp = osp.join("resources", "datasets", "star_classification.csv")
star_data_raw = pd.read_csv(star_data_fp)

funcs.print_info(star_data_raw)
