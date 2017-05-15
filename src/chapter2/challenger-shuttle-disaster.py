import pymc as pm
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy.stats.mstats import mquantiles
from separation_plot import separation_plot

# separation_plot:
# a data visualisation approach to allow graphical comparison of alternate models against each other

figsize(6.5, 3.5)
matplotlib.rcParams.update({'font.size': 8})
np.set_printoptions(precision=3, suppress=True)
challenger_data = np.genfromtxt(
  "/Users/greg/dev/python/bm4h/data/challenger_data.csv",
  skip_header=1,
  usecols=[1, 2],
  missing_values="NA",
  delimiter=",")

# Drop NaN values
challenger_data = challenger_data[~np.isnan(challenger_data[:, 1])]
print "Temp (F) O-ring failure"
print challenger_data

# Plot o-ring failure (col 1 ) against temperature (col 2)
plt.scatter(
    challenger_data[:, 0],
    challenger_data[:, 1],
    s=75,
    color="k",
    alpha=0.5)
plt.yticks([0, 1])
plt.ylabel("Damage incident?")
plt.xlabel("Outside temperature (Fahrenheit)")
plt.title("Defects of the space shuttle o-rings versus temperature")
plt.show()

# We need a function of temperature call it p(t) that is
# bounded between 0 and 1. (see logistic-function.py)

# We model parameters alpha and beta from the logistic function.
# The alpha and beta parameters are not:
#   1) only positive
#   2) bounded
#   3) relatively large
# so we choose
#     Normal distribution
# X ~ N(u, 1/t) has a distribution with two parameters:
# the mean u and the precision t.
# the smaller the t the wider the distribution
# the larger the t the tighter the distribution ( we are more certain)
# f (x | u, t) = sqrt((t / 2pi) exp (-(t/2)((x-u)^ 2)))
# So we are using the normal distribution
# to estimate parameters for the logistic function p(t)

temperature = challenger_data[:, 0]
D = challenger_data[:, 1]  # defect or not?
beta = pm.Normal("beta", 0, 0.001, value=0)
alpha = pm.Normal("alpha", 0, 0.001, value=0)


def logistic(x, beta, alpha=0):
    return 1.0 / (1.0 + np.exp(np.dot(beta, x) + alpha))


@pm.deterministic
def p(t=temperature, alpha=alpha, beta=beta):
    return logistic(t,  beta, alpha)

# We now connect our probabilities to our observed data
# We use a Bernoulli variable
# Defect Incident D_i ~ Ber(p(t_i)), i = 1..N
# Notice that we set the values of beta and alpha to 0.
# The reason for this is that if beta and alpha are very large
# they make p equal to 1 or 0.
# Unfortunately pm.Bernoulli does not like probabilities of exactly 0 or 1
# setting coefficients to 0 we indirectly set p to a reasonable starting value


observed = pm.Bernoulli("bernoulli_obs", p, value=D, observed=True)
model = pm.Model([observed, beta, alpha])
map_ = pm.MAP(model)
map_.fit()
mcmc = pm.MCMC(model)
mcmc.sample(120000, 100000, 2)

alpha_samples = mcmc.trace('alpha')[:, None]  # best to make them 1D
beta_samples = mcmc.trace('beta')[:, None]

# histogram of the samples

plt.subplot(211)
plt.title(r"Posterior distribution of the model parameters $\alpha, \beta$")
plt.hist(
  beta_samples,
  histtype='stepfilled',
  bins=35,
  alpha=0.85,
  label=r"posterior of $\beta$",
  color="#7A68A6",
  normed=True)
plt.legend()
plt.subplot(212)
plt.hist(
  alpha_samples,
  histtype='stepfilled',
  bins=35,
  alpha=0.85,
  label=r"posterior of $\alpha$",
  color="#A60628",
  normed=True)
plt.xlabel("Value of parameter")
plt.ylabel("Density")
plt.legend()
plt.show()
t = np.linspace(
  temperature.min() - 5,
  temperature.max() + 5,
  50)[:, None]
p_t = logistic(t.T, beta_samples, alpha_samples)
# p_t are probabilities skewed through the logistic function
mean_prob_t = p_t.mean(axis=0)
plt.plot(
  t,
  mean_prob_t,
  lw=3,
  label="average posterior \n probability of defect")
plt.plot(t, p_t[0, :],  ls="--", label="realization from posterior")
plt.plot(t, p_t[-2, :], ls="--", label="realization from posterior")
plt.show()
# vectorized bottom and top 2.5% quantiles for "credible interval"

qs = mquantiles(p_t, [0.025, 0.975], axis=0)
plt.fill_between(t[:, 0], *qs, alpha=0.7, color="#7A68A6")
plt.plot(
  t,
  mean_prob_t,
  lw=1,
  ls="--",
  color="k",
  label="average posterior \n probability of defect")
plt.xlim(t.min(), t.max())
plt.ylim(-0.02, 1.02)
# between 0 and 1 with a little extra
plt.legend(loc="lower left")
plt.scatter(temperature, D, color="k", s=50, alpha=0.5)
plt.xlabel("Temperature, $t$")
plt.ylabel("Probability estimate")
plt.title("Posterior probability of estimate, given temperature $t$")
plt.show()

prob_31 = logistic(31, beta_samples, alpha_samples)
plt.xlim(0.995, 1)
plt.hist(prob_31, bins=1000, normed=True, histtype='stepfilled')
plt.title("Posterior distribution pf probability of defect given $t = 31$")
plt.ylabel("Density")
plt.xlabel("Probability of defect occurring in o-ring")
plt.show()

simulated_data = pm.Bernoulli("simulation_data", p)
# suppose we think we have a good model of the data with parameters"
# p, alpha and beta
# we only get to compare data generated from our model with observed data
# here is our model

simulated = pm.Bernoulli("bernoulli_sim", p)
N = 10000

mcmc = pm.MCMC([simulated, alpha, beta, observed])
# lets sample
mcmc.sample(N)
# when we sample we get N x 23 , as its a Bernoulli we get a Boolean but we cast it to an int
simulations = mcmc.trace("bernoulli_sim")[:].astype(int)
print "shape of simulations array: ", simulations.shape
plt.title("Simulated datasets using posterior parameters")
for i in range(4):
    ax = plt.subplot(4, 1, i+1)
    plt.scatter(
        temperature,
        simulations[1000*i, :],
        color="k",
        s=50,
        alpha=0.6)
plt.show()

# We wish to asses how good our model is. We do this graphically.
# The alternative is to use Bayesian P values which are a statistical
# summary of the model. However its still subjective where
# to cut of the parameters

# Separation plots allow the user to graphically compare a suite of
# models against each other.
# For each model calculate the proportion of times the posterior
# simulation proposed a value of 1 for a particular temperature - that
# is estimate
# P (Defect = 1 | t). This gives the posteriror probability of a defect
# at ach data point (temperature) in our dataset

# first lets generate our data,
# we will try to test if the model is a good fit from this generated data.
# calculate the probability from the model (posterior probability)
# count the number of observations across all samples for each temperature
# we are doing a column-wise calculation by taking the mean
# we now have the probability for each temperature
posterior_probability = simulations.mean(axis=0)
print "Obs. | Array of Simulated Defects \
            | Posterior Probability of Defect | Realised Defect"
for i in range(len(D)):
    print "%s | %s | %.2f                        |   %d" %\
        (str(i).zfill(2), str(simulations[:10, i])[:-1] +
         "...]".ljust(12), posterior_probability[i], D[i])

# we now sort each column by posterior probability
# here we generate an index of element locations
# for sorting posterior probability
ix = np.argsort(posterior_probability)
# and print out whether the defect was observed against this probability
print "Posterior probability of defect | Realised defect"
# we apply i indexing values to posterior probability and observations
for i in range(len(D)):
    print "%.2f                            |  %d" %\
          (posterior_probability[ix[i]], D[ix[i]])
separation_plot(posterior_probability, D)
plt.show()
