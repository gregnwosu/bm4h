from IPython.core.pylabtools import figsize
from matplotlib import pyplot as plt
import pymc as pm

figsize(12.5, 4)
lambda_ = pm.Exponential("poission_param", 1)
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["figure.dpi"] = 300


# samples =
data_generator = pm.Poisson("data_generator", lambda_)
data_plus_one = data_generator + 1

print ("Children of lambda :", lambda_.children)
print ("data_generator.value:", data_generator.value)
print ("data_plus_one.value", data_plus_one.value)

#  Calling random

lambda_1 = pm.Exponential("lambda_1", 1)
lambda_2 = pm.Exponential("lambda_2", 1)
tau = pm.DiscreteUniform("tau", lower=1, upper=10)

print "lambda_1.value: %.3f" % lambda_1.value
print "lambda_2.value: %.3f" % lambda_2.value
print "tau.value: %.3f" % tau.value

print

lambda_1.random()
lambda_2.random()
tau.random()

print " After calling random() on our variables"

print "lambda_1.value: %.3f" % lambda_1.value
print "lambda_2.value: %.3f" % lambda_2.value
print "tau.value: %.3f" % tau.value

samples = [lambda_1.random() for i in range(20000)]
plt.hist(samples, bins=70, normed=True, histtype="stepfilled")
plt.title("Prior distribution for $\lambda_1$")
plt.xlabel("Value")
plt.ylabel("Density")
plt.xlim(0, 8)
plt.show()
