# Common imports
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.style.use('ggplot')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 3, 'axes.labelsize': fontsize + 2,
             'lines.markersize': 7, 'figure.figsize': [15, 10],
             'ytick.labelsize': fontsize, 'figure.autolayout': True,
             'xtick.labelsize': fontsize, 'legend.loc': 'best',
             'legend.fontsize': fontsize + 2, 'figure.titlesize': fontsize + 5}
plt.rcParams.update(newparams)

# src: https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/Programs/BoltzmannMachines/VMC/python/block.py

# Where to save the figures and data files
DATA_ID = ""


def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)


infile = open(data_path("Energies.dat"), 'r')

from numpy import log2, zeros, mean, var, sum, loadtxt, arange, array, cumsum, dot, transpose, diagonal, sqrt
from numpy.linalg import inv
import numpy as np


def block(x):
    # preliminaries
    n = len(x)
    d = int(log2(n))
    s, gamma = zeros(d), zeros(d)
    mu = np.mean(x)

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in arange(0, d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1) * sum((x[0:(n - 1)] - mu) * (x[1:n] - mu))
        # estimate variance of x
        s[i] = var(x)
        # perform blocking transformation
        # print(x, len(x))
        x = 0.5 * (x[0:-1:2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (cumsum(((gamma / s)**2 * 2**arange(1, d + 1)[::-1])[::-1]))[::-1]

    # we need a list of magic numbers
    q = array([6.634897, 9.210340, 11.344867, 13.276704, 15.086272, 16.811894, 18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967, 27.688250, 29.141238, 30.577914,
               31.999927, 33.408664, 34.805306, 36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820, 44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in arange(0, d):
        if(M[k] < q[k]):
            break
    if (k >= d - 1):
        print("Warning: Use more data")
    return mu, s[k] / 2**(d - k)


import pandas as pd


xinpu = loadtxt(infile)

(mean0, var0) = block(xinpu)
std0 = sqrt(var0)
# mean.append(mean0)
# std.append(std0)

data = {'Mean': [mean0], 'STDev': [std0]}
frame_full = pd.DataFrame(data, index=['Values'])


c = int(len(xinpu) / 8999)
xx = zeros((c, 8999))
mean = zeros(c)
std = zeros(c)
for i in range(c):
    xx[i] = xinpu[i * 8999:(i + 1) * 8999]

for i in range(c):
    (mean0, var0) = block(xx[i])
    std0 = sqrt(var0)
    mean[i] = mean0
    std[i] = std0

pd.set_option('max_columns', 6)

data = {'Mean': mean, 'STDev': std}
# frame = pd.DataFrame(data,index=['Values'])
frame = pd.DataFrame(data)
print(frame)
print(frame_full)

np.savetxt("block_res.dat", (mean, std))
print("Lowest mean energy was {} at iteration {}.".format(
    min(mean), np.where(mean == min(mean))[0][0]))
print("Lowest std was {} at iteration {}.".format(
    min(std), np.where(std == min(std))[0][0]))


plt.plot(range(c), mean)
plt.xlabel("Iteration")
plt.ylabel(r"$\langle E \rangle$")
plt.grid(1)
plt.show()

plt.plot(range(c), std)
plt.xlabel("Iteration")
plt.ylabel(r"$\sigma (E)$")
plt.grid(1)
plt.show()
