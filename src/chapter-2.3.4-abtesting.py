import pymc as pm

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
