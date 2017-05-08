import pymc as pm
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib
import numpy as np

# We need a function of temperature ( p(t)) that is
#   1) bounded between 0 and 1 (so as to model a probability
#   2) changes from 1 to 0 as we increase temperature
# there are many such functions but the most popular choice
# the logistic function
# p(t) = 1 / (1 + e^(Bt))
# In this model B is the variable we are uncertain about
# This creates smooth structures that tend upwards or downwards
# at a rate depending on beta
# but we only see changes near 0 so we add a bias term
# p(t) = 1 / (1 + e^(Bt+a))


def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))


figsize(8, 3.5)
matplotlib.rcParams.update({'font.size': 8})
np.set_printoptions(precision=3, suppress=True)
x = np.linspace(-4, 4, 100)
plt.plot(x, logistic(x, 1),  label="$\beta = 1$", ls="--", lw=1)
plt.plot(x, logistic(x, 3),  label="$\beta = 3$", ls="--", lw=1)
plt.plot(x, logistic(x, -5), label="$\beta = -5$", ls="--", lw=1)
plt.plot(
  x, logistic(x, 1, 1),
  label="$\beta=1, \alpha = 1$",
  lw=1, color="#348ABD")
plt.plot(
  x, logistic(x, 3, -2),
  label="$\beta = 3, \alpha = -2$",
  lw=1, color="#A60628")
plt.plot(
  x, logistic(x, -5, 7),
  label="$\beta = -5, \alpha = 7$",
  lw=1, color="#7A68A6")
plt.xlabel("$x$")
plt.ylabel("Logistic function at $x$")
plt.title("Logistic function for different $\beta$ and $\alpha$ values")
plt.legend()
plt.show()
