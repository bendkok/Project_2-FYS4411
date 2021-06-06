import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import glob

mpl.style.use('ggplot')
fontsize = 20
newparams = {'axes.titlesize': fontsize + 4, 'axes.labelsize': fontsize,
             'lines.linewidth': 2, 'lines.markersize': 8,
             'ytick.labelsize': fontsize + 2,
             'xtick.labelsize': fontsize + 2,
             'legend.fontsize': fontsize + 2}
plt.rcParams.update(newparams)

CURRENT_DIR = os.path.dirname(__file__)

def plot_standard(iter = 100):

    x = np.arange(0, iter)
    files = ["/res/energy_True.dat", "/res/energy_False.dat"]
    for i in files:
        energies = np.loadtxt(CURRENT_DIR + i)
        min_energy = np.amin(energies)
        min_index = np.argmin(energies)
        plt.plot(x, energies)
        plt.plot(min_index, min_energy, 'ko', label=r"$E_{\mathrm{min}}$")
        plt.tight_layout()
        plt.grid(True)
        plt.xlabel("Iterations")
        plt.ylabel(r"$E$")
        plt.legend()
        plt.title("System energy w.r.t iterations")
        plt.show()
        print(min_energy)
        print(x[min_index])

#plot_standard()

def plot_iteration_comp():

    files = glob.glob(CURRENT_DIR + "/res/iterations/*.dat")
    #print(files)
    current_order = [1, 10, 100, 1000, 200, 300, 400, 50, 500, 700]
    correct_order = np.argsort(current_order)
    x = np.zeros(len(files))
    y = np.zeros(len(files))

    for i in range(0, len(current_order)):
        energy = np.loadtxt(files[correct_order[i]])
        energy = np.min(energy)
        #print(energy)
        #print(current_order[correct_order[i]])
        x[i] = current_order[correct_order[i]]
        y[i] = energy

        #label="%d " % (current_order[correct_order[i]])
        plt.plot(current_order[correct_order[i]], energy, 'bo')

    plt.plot(x, y, 'r:', label=r"$\overline{E}_i$")
    plt.xlabel("Iterations")
    plt.ylabel(r"$E [a.u]$")
    plt.legend()
    plt.show()

#plot_iteration_comp()

def plot_blocking_results():
    files = ["/res/block_res_True.dat", "/res/block_res_False.dat"]
    x = np.arange(0, 100)

    for i in files:
        mean, std = np.loadtxt(CURRENT_DIR + i)
        min_std = np.amin(std)
        min_std_index = np.argmin(std)
        print("Min std: ")
        print(x[min_std_index], " ", min_std)
        print("Corresponding mean: ", mean[min_std_index])

        min_mean = np.amin(mean)
        min_mean_index = np.argmin(mean)
        print("Min mean energy: ")
        print(x[min_mean_index], " ", min_mean)
        print("Corresponding std: ", std[min_mean_index])

        plt.plot(x, mean)
        plt.plot(x[min_mean_index], min_mean, 'ko', label=r"$\mathrm{min} (\langle  E \rangle)$")
        plt.plot(x[min_std_index], mean[min_std_index], 'bo', label=r"$\mathrm{min} (\sigma (E))$")
        plt.xlabel("Iterations")
        plt.ylabel(r'$\langle  E \rangle$')
        plt.grid(True)
        plt.legend()
        plt.show()

        plt.plot(x, std)
        plt.yscale("log")
        plt.xlabel("Iterations")
        plt.ylabel(r"$\sigma (E)$")
        plt.plot(x[min_mean_index], std[min_mean_index], 'ko', label=r"$\mathrm{min} (\langle  E \rangle)$")
        plt.plot(x[min_std_index], min_std, 'bo', label=r"$\mathrm{min} (\sigma (E))$")
        plt.grid(True)
        plt.legend()
        plt.show()

#plot_blocking_results()

def plot_gradient():
    files = glob.glob(CURRENT_DIR + "/res/learning_rate/*.txt")
    current_order = [0.001, 0.005, 0.01, 0.05, 0.1]
    x = range(100)

    for i in range(0, len(files)):
        energy = np.loadtxt(files[i])

        plt.plot(x, energy, label=r"$\eta = %.3f $" % current_order[i])

    plt.legend()
    plt.title("Comparing learning rate performance")
    plt.ylabel(r"$E$")
    plt.xlabel("Iterations")
    plt.show()


#plot_gradient()
