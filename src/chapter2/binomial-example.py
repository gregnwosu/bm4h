import pymc as pm
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats as stats
import numpy as np


# The Binomial Distribution
# P(X = k) = (N k) p^(k)(1 - p)^(N -k)
# if X is a binomial variable with parameters p and N denoted X ~ Bin(N,p).
# Then is is the number of events that occurred in the N trials 0<=X<=N
# The larger p is(while still remaining between 0 and 1),
# the more events are likely to occur.
# The expected value of a binomial is equal to Np
def plotBinomial(params, _color):
    N, p = params
    _x = np.arange(N + 1)
    plt.bar(
      _x - 0.5, binomial.pmf(_x, N, p), color=_color,
      edgecolor=_color, alpha=0.6,
      label="$N$: %d, $p$: %.1f" % (N, p),
      linewidth=3)


figsize(4.5, 3.5)
matplotlib.rcParams.update({'font.size': 8})
binomial = stats.binom
parameters = [(10, 0.4), (10, 0.9)]
colors = ["#348ABD", "#A60628"]
for i in range(2):
    plotBinomial(parameters[i], colors[i])
plt.legend(loc="upper left")
plt.xlim(0, 10.5)
plt.xlabel("$k$")
plt.ylabel("$P(X = k)$")
plt.title("Probability mass distributions of binomial random variables")
plt.show()
