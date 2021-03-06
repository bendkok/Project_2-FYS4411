# Common imports
import os
from numpy import log2, zeros, var, sum, loadtxt, arange, array, cumsum, sqrt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#src: https://github.com/CompPhysics/ComputationalPhysics2/blob/gh-pages/doc/Programs/BoltzmannMachines/VMC/python/block.py

# Where to save the figures and data files
DATA_ID = ""

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)


interaction = True
#save and load files based upon wheter we use interaction or not
infile = open(data_path("res/Energies_" + str(interaction) + ".dat"), 'r')

def block(x):
    # preliminaries
    n = len(x)
    d = int(log2(n))
    s, gamma = zeros(d), zeros(d)
    mu = np.mean(x)

    # estimate the auto-covariance and variances
    # for each blocking transformation
    for i in arange(0,d):
        n = len(x)
        # estimate autocovariance of x
        gamma[i] = (n)**(-1)*sum( (x[0:(n-1)]-mu)*(x[1:n]-mu) )
        # estimate variance of x
        s[i] = var(x)
        # perform blocking transformation
        x = 0.5*(x[0:-1:2] + x[1::2])

    # generate the test observator M_k from the theorem
    M = (cumsum( ((gamma/s)**2*2**arange(1,d+1)[::-1])[::-1] )  )[::-1]

    # we need a list of magic numbers
    q =array([6.634897,9.210340, 11.344867, 13.276704, 15.086272, 16.811894, 18.475307, 20.090235, 21.665994, 23.209251, 24.724970, 26.216967, 27.688250, 29.141238, 30.577914, 31.999927, 33.408664, 34.805306, 36.190869, 37.566235, 38.932173, 40.289360, 41.638398, 42.979820, 44.314105, 45.641683, 46.962942, 48.278236, 49.587884, 50.892181])

    # use magic to determine when we should have stopped blocking
    for k in arange(0, d):
        if(M[k] < q[k]):
            break
    if (k >= d - 1):
        print("Warning: Use more data")
    return mu, s[k] / 2**(d - k)


xinpu = loadtxt(infile)

(mean0, var0) = block(xinpu)
std0 = sqrt(var0)

data = {'Mean': [mean0], 'STDev': [std0]}
frame_full = pd.DataFrame(data, index=['Values'])


c = int(len(xinpu) / 8999)
xx = zeros((c, 8999))
mean = zeros(c)
std = zeros(c)
for i in range(c):
    xx[i] = xinpu[i * 8999: (i + 1) * 8999]

for i in range(c):
    (mean0, var0) = block(xx[i])
    std0 = sqrt(var0)
    mean[i] = mean0
    std[i] = std0


if interaction:
    print("Interaction results:")
else:
    print("No interaction results:")


pd.set_option('max_columns', 6)

data = {'Mean': mean, 'STDev': std}
frame = pd.DataFrame(data)
print(frame)
print(frame_full)

np.savetxt("res/block_res_" + str(interaction) + ".dat", (mean, std))
print("Lowest mean energy was {} at iteration {}.".format(min(mean), np.where(mean == min(mean))[0][0]))
print("Lowest std was {} at iteration {}.".format(min(std), np.where(std == min(std))[0][0])) 


