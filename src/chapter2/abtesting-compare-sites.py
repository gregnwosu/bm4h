import pymc as pm
from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import matplotlib

figsize(4.5, 3.5)
matplotlib.rcParams.update({'font.size': 8})
# These two quantities are unknown to us
true_p_A = 0.05
true_p_B = 0.04
# Notice the unequal sample sizes-no problem in Bayesian analysis
N_A = 1500
N_B = 750
# Generate some observations
observations_A = pm.rbernoulli(true_p_A, N_A)
observations_B = pm.rbernoulli(true_p_B, N_B)
print "Obs from site A: ", observations_A[:30].astype(int), "..."
print "Obs from site B: ", observations_B[:30].astype(int), "..."
print observations_A.mean()
print observations_B.mean()
p_A = pm.Uniform("p_A", 0, 1)
p_B = pm.Uniform("p_B", 0, 1)
# Define the deterministic delta function


@pm.deterministic
def delta(p_A=p_A, p_B=p_B):
    return p_A - p_B

# so we reckon our (generated) observed data is Bernoulli
# we have no idea of p_a p_b so we create a uniform stochastic var for each
# we plug in our observations to get a stochastic model


obs_A = pm.Bernoulli("obs_A", p_A, value=observations_A, observed=True)
obs_B = pm.Bernoulli("obs_B", p_B, value=observations_B, observed=True)
mcmc = pm.MCMC([p_A, p_B, delta, obs_A, obs_B])
mcmc.sample(25000, 5000)
p_A_samples = mcmc.trace("p_A")[:]
p_B_samples = mcmc.trace("p_B")[:]
delta_samples = mcmc.trace("delta")[:]
plt.xlim(0, 0.1)
plt.ylim(0, 80)
plt.legend(loc="upper right")
plt.title("Posterior distributions of $p_A$ , $p_B$ and delta unknowns")
plt.subplot(311)
plt.hist(p_A_samples, histtype='stepfilled', bins=30,
         alpha=0.85, label="posterior of $p_A$", color="#A60628", normed=True)
plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_A$ (unknown)")
plt.subplot(312)
plt.hist(p_B_samples, histtype='stepfilled', bins=30,
         alpha=0.85, label="posterior of $p_B$", color="#467821", normed=True)
plt.vlines(true_p_A, 0, 80, linestyle="--", label="true $p_B$ (unknown)")
plt.subplot(313)
plt.hist(delta_samples, histtype='stepfilled', bins=30,
         alpha=0.85, label="posterior of delta", color="#7A68A6", normed=True)
plt.vlines(true_p_A - true_p_B, 0, 80, linestyle="--",
           label="true delta (unknown)")
plt.vlines(0, 0, 60, color="black", alpha=0.2)
plt.xlabel("Value")
plt.ylabel("Density")
plt.show()

# Lets plot the posterior distribution side by side so we can see how valuable

# histogram of posteriors

plt.xlim(0, 0.1)
plt.hist(
  p_A_samples, histtype='stepfilled', bins=30,  alpha=0.8,
  label='posterior of $p_A$', color='#A60628', normed=True)
plt.hist(
  p_B_samples, histtype='stepfilled', bins=30, alpha=0.8,
  label='posterior of $p_B$', color='#467821', normed=True)
plt.legend(loc="upper right")
plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Posterior distribution of $p_A$ and $p_B$")
plt.ylim(0, 80)
plt.show()
