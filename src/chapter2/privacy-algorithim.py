import pymc as pm
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats as stats

figsize(4.5, 3.5)
matplotlib.rcParams.update({'font.size': 8})

N = 100
p = pm.Uniform("freq_cheating", 0, 1)

true_answers = pm.Bernoulli("truths", p, size=N)
first_coin_flips = pm.Bernoulli("first_flips", 0.5, size=N)
second_coin_flips = pm.Bernoulli("second_flips", 0.5, size=N)


@pm.deterministic
def observed_proportion(t_a=true_answers,
                        fc=first_coin_flips,
                        sc=second_coin_flips):
        observed = fc * t_a + (1 - fc) * sc
# this line contains the heart of the privacy algorithm.
# An element in this array is one if and only if
# 1) the first toss is heads and the student cheated (t) or;
# 2) the first toss is tails  and the second toss is heads
        return observed.sum() / float(N)
# this line takes the number of 1's in the line
# and turns it into a probability of receiving a 1.

# The researchers received 35 "Yes" responses
# Therefore: The researchers observe a binomial random variable with
# N = 100
# p = observed proportion
# value = 35
# n.b. we dont know what p is so initially we create p from a uniform distribution
# however we do know that p is skewed so we must skew our uniform p
# before we add it to our binomial equation


X = 35
# we always build our model from our unknown stochastic vars

observations = pm.Binomial(
  "obs",
  N,
  observed_proportion,
  observed=True,
  value=X)

model = pm.Model(
  [p,
   true_answers,
   first_coin_flips,
   second_coin_flips,
   observed_proportion,
   observations])

mcmc = pm.MCMC(model)
mcmc.sample(40000, 15000)
p_trace = mcmc.trace("freq_cheating")[:]
plt.hist(
  p_trace,
  histtype="stepfilled",
  normed="True",
  alpha=0.85,
  bins=30,
  label="posterior distribution",
  color="#348ABD"
)
plt.show()
