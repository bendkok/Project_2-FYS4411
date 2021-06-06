# -*- coding: utf-8 -*-
"""
Created on Wed May 12 21:11:11 2021

@author: benda
"""

# %matplotlib inline

# 2-electron VMC code for 2dim quantum dot with importance sampling
# Using gaussian rng for new positions and Metropolis- Hastings 
# Added restricted boltzmann machine method for dealing with the wavefunction
# RBM code based heavily off of:
# https://github.com/CompPhysics/ComputationalPhysics2/tree/gh-pages/doc/Programs/BoltzmannMachines/MLcpp/src/CppCode/ob
from random import random, seed, normalvariate
import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
import glob
import os

np.random.seed(2)

# Trial wave function for the 2-electron quantum dot in two dims
@nb.njit
def WaveFunction(p, d, h, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    Psi1 = 0.0
    Psi2 = 1.0
    Q = Qfac(h, r, b, w)

    for iq in range(p):
        for ix in range(d):
            Psi1 += (r[iq,ix]-a[iq,ix])**2

    for ih in range(h):
        Psi2 *= (1.0 + np.exp(Q[ih]))

    Psi1 = np.exp(-Psi1/2)

    return Psi1*Psi2

# Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
@nb.njit
def LocalEnergy(p, d, h, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray, interaction) -> float:
    locenergy = 0.0

    Q = Qfac(h, r, b, w)

    for iq in range(p):
        for ix in range(d):
            sum1 = 0.0
            sum2 = 0.0
            for ih in range(h):
                sum1 += w[iq,ix,ih]/(1+np.exp(-Q[ih]))
                sum2 += w[iq,ix,ih]**2 * np.exp(-Q[ih]) / (1.0 + np.exp(-Q[ih]))**2 #minustegn?

            dlnpsi1 = -(r[iq,ix] - a[iq,ix]) + sum1
            dlnpsi2 = -1 + sum2
            locenergy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + r[iq,ix]**2)

    if(interaction is True):
        for iq1 in range(p):
            for iq2 in range(iq1):
                distance = 0.0
                for ix in range(d):
                    distance += (r[iq1,ix] - r[iq2,ix])**2

                locenergy += 1/np.sqrt(distance)

    return locenergy

# Derivate of wave function ansatz as function of variational parameters
@nb.njit
def DerivativeWFansatz(h, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple:
    Q = Qfac(h, r, b, w)

    WfDer_a = r - a
    nev = 1 + np.exp(-Q) 
    WfDer_b = 1 / nev
    WfDer_w = w / nev

    return  WfDer_a, WfDer_b, WfDer_w

# Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
@nb.njit
def QuantumForce(p, d, h, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    qforce = np.zeros((p, d), np.double)
    sum1 = np.zeros((p, d), np.double)

    Q = Qfac(h, r, b, w)

    for ih in range(h):
        sum1 += w[:,:,ih]/(1+np.exp(-Q[ih]))

    qforce = 2*(-(r-a) + sum1)

    return qforce

@nb.njit
def Qfac(h: int, r: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    Q = np.zeros((h), np.double)
    temp = np.zeros((h), np.double)

    for ih in range(h):
        temp[ih] = (r*w[:,:,ih]).sum()

    Q = b + temp

    return Q

# Computing the derivative of the energy and the energy 
# @nb.njit
def EnergyMinimization(p, d, h, mc_cycles, a: np.ndarray, b: np.ndarray, w: np.ndarray, outfile, equ_frac, TimeStep = 0.05, interaction=True, Printout=True) -> tuple:
    # Parameters in the Fokker-Planck simulation of the quantum force
    D = 0.5
    TimeStep = 0.05
    # positions
    PositionOld = np.zeros((p, d), np.double)
    PositionNew = np.zeros((p, d), np.double)
    # Quantum force
    QuantumForceOld = np.zeros((p, d), np.double)
    QuantumForceNew = np.zeros((p, d), np.double)

    energy = 0.0
    DeltaE = 0.0

    EnergyDer_a, EnergyDer_b, EnergyDer_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)
    DeltaPsi_a, DeltaPsi_b, DeltaPsi_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)
    DerivativePsiE_a, DerivativePsiE_b, DerivativePsiE_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)


    #Initial position
    for i in range(p):
        for j in range(d):
            PositionOld[i, j] = normalvariate(0.0, 1.0) * np.sqrt(TimeStep)

    wfold = WaveFunction(p, d, h, PositionOld, a, b, w)
    QuantumForceOld = QuantumForce(p, d, h, PositionOld, a, b, w)

    #Loop over MC MCcycles
    for MCcycle in range(mc_cycles):
        #Trial position moving one particle at the time
        for i in range(p):
            for j in range(d):
                PositionNew[i,j] = PositionOld[i,j]+normalvariate(0.0,1.0)*np.sqrt(TimeStep)+\
                                       QuantumForceOld[i,j]*TimeStep*D

            wfnew = WaveFunction(p, d, h, PositionNew, a, b, w)
            QuantumForceNew = QuantumForce(p, d, h, PositionNew, a, b, w)

            GreensFunction = 0.0
            for j in range(d):
                GreensFunction += 0.5*(QuantumForceOld[i,j]+QuantumForceNew[i,j])*\
                                      (D*TimeStep*0.5*(QuantumForceOld[i,j]-QuantumForceNew[i,j])-\
                                      PositionNew[i,j]+PositionOld[i,j])

            GreensFunction = np.exp(GreensFunction)
            ProbabilityRatio = GreensFunction*wfnew**2/wfold**2
            #Metropolis-Hastings test to see whether we accept the move
            np.random.seed(5)
            if np.random.uniform(0.0, 1.0) <= ProbabilityRatio:
                for j in range(d):
                    PositionOld[i,j] = PositionNew[i,j]
                    QuantumForceOld[i,j] = QuantumForceNew[i,j]

                wfold = wfnew

        DeltaE = LocalEnergy(p, d, h, PositionOld, a, b, w, interaction)
        DerPsi = DerivativeWFansatz(h, PositionOld, a, b, w)

        if MCcycle > mc_cycles*equ_frac:
            DeltaPsi_a += DerPsi[0]
            DeltaPsi_b += DerPsi[1]
            DeltaPsi_w += DerPsi[2]

            energy += DeltaE

            if Printout: 
                outfile.write('%f\n' %(energy/(MCcycle-int(mc_cycles*equ_frac) + 1.0)))

            DerivativePsiE_a += DerPsi[0]*DeltaE
            DerivativePsiE_b += DerPsi[1]*DeltaE
            DerivativePsiE_w += DerPsi[2]*DeltaE

    # We calculate mean values
    fraq = mc_cycles - (mc_cycles * equ_frac)
    energy /= fraq

    DerivativePsiE_a /= fraq
    DerivativePsiE_b /= fraq
    DerivativePsiE_w /= fraq

    DeltaPsi_a /= fraq
    DeltaPsi_b /= fraq
    DeltaPsi_w /= fraq

    EnergyDer_a  = 2*(DerivativePsiE_a-DeltaPsi_a*energy)
    EnergyDer_b  = 2*(DerivativePsiE_b-DeltaPsi_b*energy)
    EnergyDer_w  = 2*(DerivativePsiE_w-DeltaPsi_w*energy)

    return energy, [EnergyDer_a, EnergyDer_b, EnergyDer_w]


def nodes_and_weights(p, d, h):
    a = np.random.normal(loc=0.0, scale=.5, size=(p, d))
    b = np.random.normal(loc=0.0, scale=.5, size=(h))
    w = np.random.normal(loc=0.0, scale=.5, size=(p, d, h))
    return a, b, w

def run_simulation(eta, MaxIterations, NumberMCcycles, info, filename="res/Energies_", interaction=True, Printout=True):

    tot_time = time.time()

    NumberParticles = 2
    Dimension = 2
    NumberHidden = 2
    equ_frac = 0.1
    gamma = 0.9

    # guess for parameters
    a, b, w = nodes_and_weights(NumberParticles, Dimension, NumberHidden)

    #savefile is based upon whether we use interaction or not
    outfile = open(filename + info + ".dat", 'w')

    # Set up iteration using stochastic gradient method
    Energy = 0
    EDerivative = [np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)]

    np.seterr(invalid='raise')
    Energies = np.zeros(MaxIterations)
    da = np.zeros(MaxIterations)
    times = np.zeros(MaxIterations)

    momentum_a = np.zeros_like(a)
    momentum_b = np.zeros_like(b)
    momentum_w = np.zeros_like(w)

    perc = -1
    message = 'PROGRESS:'

    for iteration in range(MaxIterations):
        if int(100 * iteration / MaxIterations) > perc:
            perc = int(100 * iteration / MaxIterations)
            print(f'\r{message}{perc: 3.0f} %', end = '')

        timing = time.time()

        Energy, EDerivative = EnergyMinimization(NumberParticles, Dimension, NumberHidden, NumberMCcycles, a-momentum_a*gamma,b-momentum_b*gamma,w-momentum_w*gamma, outfile, equ_frac, interaction=interaction, Printout=Printout)

        momentum_a = momentum_a * gamma + eta * EDerivative[0]
        momentum_b = momentum_b * gamma + eta * EDerivative[1]
        momentum_w = momentum_w * gamma + eta * EDerivative[2]

        a -= momentum_a
        b -= momentum_b
        w -= momentum_w

        Energies[iteration] = Energy
        da[iteration] = np.sum(np.abs(momentum_a))
        times[iteration] = time.time() - timing

    print(f'\r{message} {100:3.0f}%')
    #nice printout with Pandas
    pd.set_option('max_columns', 6)
    data = {'Energy': Energies, 'da': da, 'Time': times}


    if interaction:
        print("Interaction results:")
    else:
        print("No interaction results:")

    frame = pd.DataFrame(data)
    print(frame)
    print("Average energy: {}. Lowest: {}".format(np.mean(Energies), np.min(Energies)))
    print("Total elapsed time: {}s".format(time.time() - tot_time))

    np.savetxt(filename + info + ".txt", Energies)

    outfile.close()

    return Energy, da, times



#_, _, _ = run_simulation(0.05, 100, 10000, info="False", interaction=False, Printout=True)


def test_learning_rate():
    etas = [0.1, 0.05, 0.01, 0.005, 0.001]
    for i in etas:
        Energy, da, times = run_simulation(i, 100, 10000, str(i), filename="res/learning_rate/Energies_", interaction=True, Printout=False)

test_learning_rate()

def test_iterations():
    iterations = [1, 10, 50, 100, 500, 1000]

    for i in iterations:
        Energy, da, times = run_simulation(0.05, i, 10000, str(i), filename="res/iterations/Energies_", Printout=False)

#test_iterations()