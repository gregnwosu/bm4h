from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pymc as pm
import numpy as np
import matplotlib

figsize(2.5, 0.7)
matplotlib.rcParams.update({'font.size': 3})
tau = pm.rdiscrete_uniform(0, 80)
print tau
alpha = 1. / 20.
lambda_1, lambda_2 = pm.rexponential(alpha, 2)
print lambda_1, lambda_2

# for days before t lambda = lambda1 for days after t lambda =lambda_3
lambda_ = np.r_[lambda_1 * np.ones(tau), lambda_2 * np.ones(80-tau)]

print lambda_
data = pm.rpoisson(lambda_)
print data

# plot the artificial dataset

plt.bar(np.arange(80), data, color="#348ABD")
plt.bar(tau - 1,
        data[tau - 1],
        color='r',
        label="user behaviour changed")

plt.xlabel("Time days")
plt.ylabel("Text messages received")
plt.title("Artificial data-set fro simulating the model")
plt.xlim(0, 80)
plt.legend()
plt.show()
