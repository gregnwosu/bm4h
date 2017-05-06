from  matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize
import numpy as np

figsize(12.5, 3.5)
np.set_printoptions(precision=3, suppress=True)
data = np.genfromtxt(
  "/Users/greg/dev/python/bm4h/data/stuff.csv",
  skip_header=1,
  usecols=[1, 2],
  missing_values="NA",
  delimiter=",")

def main():
    print data
