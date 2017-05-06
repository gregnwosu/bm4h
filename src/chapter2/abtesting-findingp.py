import pymc as pm
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib

p_true = 0.05  # remember this is unknown in real life
# setting up a Bayesian model
# p is the probability that a *individual*
# user converts its unknown so we make it uniform
p = pm.Uniform('p', lower=0, upper=1)
# n is the number of users shown site A
N = 1500
# Sample N Bernoulli random variables from Ber(0.05)
# Each random variable has a 0.05 chance of being a 1
# This is a data-generation step
occurences = pm.rbernoulli(p_true, N)

print occurences  # Remember : Python treats True == 1 and False == 0
print occurences.sum()
print "What is the observed frequency in Group A %.4f" % occurences.mean()
obsFreqEqTrueFreq = occurences.mean() == p_true
print "Does observed frequency equal true frequency? %s" % obsFreqEqTrueFreq

# Having found that the observed frequency from our sample
# doesn't equal the frequency from which its generated, we refine our model

# Include observations which are Bernoulli
# We use a uniform distribution for p as we have no idea what it should be

obs = pm.Bernoulli("obs", p, value=occurences, observed=True)

mcmc = pm.MCMC([p, obs])
mcmc.sample(20000, 1000)
# I prefer this figsize and font size for working on a retina mac
figsize(4.5, 3.5)
matplotlib.rcParams.update({'font.size': 8})
plt.title("Posterior distribution of $p_A$\n the true effectiveness of site A")
plt.vlines(p_true, 0, 90, linestyle="-", label="true $p_A$ (unknown)")
plt.hist(mcmc.trace("p")[:], bins=35, histtype="stepfilled", normed=True)
plt.xlabel("Value of $p_A$")
plt.ylabel("Density")
plt.legend()
plt.show()
