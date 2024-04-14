import numpy as np
import pandas as pd
from scipy.stats import boxcox


def boxcox_3p(time_series: pd.Series):
   boxcox_time_series, lmbda = boxcox(time_series + 3)
   return boxcox_time_series, lmbda


def invboxcox(y, lmbda):
   """Inverse boxcox transformation by given lmbda"""
   if lmbda == 0:
      return np.exp(y)
   else:
      return np.exp(np.log(lmbda * y + 3) / lmbda)
