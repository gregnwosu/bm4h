import pymc as pm
import numpy as np
from matplotlib import pyplot as plt
from IPython.core.pylabtools import figsize

count_data = np.loadtxt("/Users/greg/dev/python/bm4h/data/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color='#348ABD')
plt.xlabel('Time(days)')
plt.ylabel('Text messages recieved')
plt.title("Did the users texting habits change over time?")
plt.xlim(0, n_count_data)
plt.show()
