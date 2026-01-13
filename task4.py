#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABC-SMC parameter inference example.

Source: pyabc documentation example (https://pyabc.readthedocs.io/en/latest/examples/parameter_inference.html)
Adapted by: Kai Budde (2019-08-06)
Modified by: Pia Wilsdorf (2026-01-07)
"""

import pyabc

import scipy as sp
import numpy as np
import tempfile
import os
import matplotlib.pyplot as plt

#%% Model

# Assume a Gaussian model N(mean,0.5^2) with the single parameter mean and
# variance 0.5^2.
def model(parameter):
    return {"data": parameter["mean"] + 0.5 * np.random.randn()}

#%% What it does:

# Get help:
print(np.random.randn.__doc__)

dummy = np.random.randn(1000) *3
n, bins, patches = plt.hist(dummy, 50)

#%% Data
# Here, we just assume, that the measured data was 2.5.

observation = 6

#%% Prior for the mean

# We then define the prior for the mean to be uniform over the
# interval [0, 0+5].
prior = pyabc.Distribution(mean=pyabc.RV("uniform", 0, 5))

#%% Distance function

# Specify when we consider data to be close in form of a distance funtion.
# We just take the absolute value of the difference here.

def distance(x, y):
    return abs(x["data"] - y["data"])


#%% Creat ABCSMC Object

abc = pyabc.ABCSMC(model, prior, distance)

# Specify where to log the ABC-SMC runs.
# We can later query the database with the help of the History class.
# Usually you would now have some measure data which you want to know the
# posterior of.

db_path = ("sqlite:///" +
           os.path.join(tempfile.gettempdir(), "test.db"))

# The new method returned an integer. This is the id of the ABC-SMC run.
# This id is only important if more than one ABC-SMC run is stored in the
# same database.
abc.new(db_path, {"data": observation})

#%% Start sampling
# We’ll sample until the acceptance threshold epsilon drops below 0.1.
# We also specify that we want a maximum number of 10 populations.

history = abc.run(minimum_epsilon=.1, max_nr_populations=10)

# The History object returned by ABCSMC.run can be used to query the database.
# This object is also available via abc.history.

#%% Visualizing the results

# We visualize the probability density functions.
# The vertical line indicates the location of the observation.

fig, ax = plt.subplots()

for t in range(history.max_t+1):
    df, w = history.get_distribution(m=0, t=t)
    pyabc.visualization.plot_kde_1d(
            df, w,
            xmin=0, xmax=5,
            x="mean", ax=ax,
            label="PDF t={}".format(t))

ax.axvline(observation, color="k", linestyle="dashed")
ax.legend()

#%% Visualizing more results

pyabc.visualization.plot_sample_numbers(history)

pyabc.visualization.plot_epsilons(history)

pyabc.visualization.plot_credible_intervals(
        history, levels=[0.95, 0.9, 0.5], ts=[0, 1, 2, 3, 4],
        show_mean=True, show_kde_max_1d=True,
        refval={'mean': 2.5})

pyabc.visualization.plot_effective_sample_sizes(history)


# Display all figures
plt.show()

