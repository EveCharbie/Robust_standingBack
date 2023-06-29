import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

# --- Graph 1 : Guassienne normal article Diane --- #

mu = 0
variance = 1
sigma = math.sqrt(variance)
x = np.linspace(mu - 0.5 * sigma, mu + 0.5 * sigma, 500)
plt.plot(x, stats.norm.pdf(x, mu, sigma), color="red", label="µ: 0, σ: 1")
plt.show()

# --- Graph 2 : Gaussienne qu'on obtient --- #

mu = 0
variance = 27.5
sigma = math.sqrt(variance)
x = np.linspace(mu - 0.5 * sigma, mu + 0.5 * sigma, 500)
plt.plot(x, stats.norm.pdf(x, mu, sigma), color="blue", label="µ: 0, σ: 27.5")
plt.show()

mu = 0
variance = 28.6
sigma = math.sqrt(variance)
x = np.linspace(mu - 0.5 * sigma, mu + 0.5 * sigma, 500)
plt.plot(x, stats.norm.pdf(x, mu, sigma), color="blue", label="µ: 0, σ: 28.6")
plt.show()
