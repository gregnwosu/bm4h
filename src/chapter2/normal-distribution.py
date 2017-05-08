#     Normal distribution
# X ~ N(u, 1/t) has a distribution with two parameters:
# the mean u and the precision t.
# the smaller the t the wider the distribution
# the larger the t the tighter the distribution ( we are more certain)
# f (x | u, t) = sqrt((t / 2pi) exp (-(t/2)((x-u)^ 2)))
# A normal random variable can take on any real number
# the variable is very likely to be relatively close to mu.
# The expected value of a normal distribution is mu
# E[X| mu, tau] = mu
# the variance is equal to inverse tau
# Var(X|mu, tau) = 1/tau
import pymc as pm
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import scipy.stats as stats

figsize(6.5, 3.5)
matplotlib.rcParams.update({'font.size': 8})
np.set_printoptions(precision=3, suppress=True)

nor = stats.norm
x = np.linspace(-8, 7, 150)
mu = (-2, 0, 3)
tau = (.7, 1, 2.8)
colors = ["#348ABD", "#A60628", "#7A68A6"]
parameters = zip(mu, tau, colors)
for _mu, _tau, _color in parameters:
    plt.plot(
      x,
      nor.pdf(x, _mu, scale=1./_tau),
      label="$\mu = %d, \tau = %.1f$" % (_mu, _tau),
      color=_color)
    plt.fill_between(
      x,
      nor.pdf(x, _mu, scale=1./_tau), color=_color, alpha=.33)
    plt.legend(loc="upper right")
    plt.xlabel("$x$")
    plt.ylabel("Density function at $x$")
    plt.title(
      "Probability distribution of 3 different normal random variables")
plt.show()
