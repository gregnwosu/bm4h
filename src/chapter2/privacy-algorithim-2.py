import pymc as pm
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib

figsize(4.5, 3.5)
matplotlib.rcParams.update({'font.size': 8})


# We can move more of the coinflip calculations into our deterministic function

# If we pretend to know p (probability of a cheater):
# we can find the probability that a student will answer yes
# P("Yes") =
#    P(heads on first coin) x p(cheater) +
#    P(tails on first coin) x P(heads on second coin)
# = 0.5p+ (0.5 * 0.5)
# = 0.5p + 0.25
# if we know the probability of respondends saying yes is p_skeed
N = 100
X = 35
p = pm.Uniform("freq_cheating", 0, 1)


@pm.deterministic
def p_skewed(p=p):
    return 0.5*p + 0.25


# we always build our model from our unknown stochastic vars

yes_responses = pm.Binomial(
  "cheaters",
  N,
  p_skewed,
  observed=True,
  value=X)

model = pm.Model(
  [p,
   yes_responses,
   p_skewed])

mcmc = pm.MCMC(model)
mcmc.sample(25000, 2500)
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
plt.vlines([.05, .35], [0, 0], [5, 5], alpha=0.2)
plt.xlim(0, 1)
plt.xlabel("Value of $p$")
plt.ylabel("Density")
plt.title("Posterior distribution of parameter $p$ , from alternate model")
plt.show()
