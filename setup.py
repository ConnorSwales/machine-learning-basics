import numpy as np
import scipy as sp
import pandas as pd
import sklearn as sk

# main.py -> Simply loads the modules and prints a log of the current module versions installed

log_str = "{} -> version {}"

print("\nModules: \n")
print(log_str.format(np.__name__, np.__version__))
print(log_str.format(sp.__name__, sp.__version__))
print(log_str.format(pd.__name__, pd.__version__))
print(log_str.format(sk.__name__, sk.__version__))
print("\n")

