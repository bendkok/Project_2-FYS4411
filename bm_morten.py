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
# from math import exp, sqrt
from random import random, seed, normalvariate
import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame


interaction=True
#savefile is based upon wheter we use interaction or not
outfile = open("res/Energies_"+str(interaction)+".dat",'w')

# Trial wave function for the 2-electron quantum dot in two dims
@nb.njit
def WaveFunction(r:np.ndarray,a:np.ndarray,b:np.ndarray,w:np.ndarray) -> float:
    # sigma=1.0
    # sig2 = 1 #sigma**2
    Psi1 = 0.0
    Psi2 = 1.0
    Q = Qfac(r,b,w)
    
    for iq in range(NumberParticles):
        for ix in range(Dimension):
            Psi1 += (r[iq,ix]-a[iq,ix])**2
            
    for ih in range(NumberHidden):
        Psi2 *= (1.0 + np.exp(Q[ih]))
        
    # Psi1 = np.exp(-Psi1/(2*sig2))
    Psi1 = np.exp(-Psi1/2)

    return Psi1*Psi2

# Local energy  for the 2-electron quantum dot in two dims, using analytical local energy
@nb.njit
def LocalEnergy(r:np.ndarray,a:np.ndarray,b:np.ndarray,w:np.ndarray) -> float:
    # sigma=1.0
    # sig2 = sigma**2
    locenergy = 0.0
    
    Q = Qfac(r,b,w)

    for iq in range(NumberParticles):
        for ix in range(Dimension):
            sum1 = 0.0
            sum2 = 0.0
            for ih in range(NumberHidden):
                sum1 += w[iq,ix,ih]/(1+np.exp(-Q[ih]))
                sum2 += w[iq,ix,ih]**2 * np.exp(-Q[ih]) / (1.0 + np.exp(-Q[ih]))**2 #minustegn?
    
            dlnpsi1 = -(r[iq,ix] - a[iq,ix]) + sum1
            dlnpsi2 = -1 + sum2
            locenergy += 0.5*(-dlnpsi1*dlnpsi1 - dlnpsi2 + r[iq,ix]**2)
            
    if(interaction==True):
        for iq1 in range(NumberParticles):
            for iq2 in range(iq1):
                distance = 0.0
                for ix in range(Dimension):
                    distance += (r[iq1,ix] - r[iq2,ix])**2
                    
                locenergy += 1/np.sqrt(distance)
                
    return locenergy

# Derivate of wave function ansatz as function of variational parameters
@nb.njit
def DerivativeWFansatz(r:np.ndarray,a:np.ndarray,b:np.ndarray,w:np.ndarray) -> tuple:
    
    # sigma=1.0
    # sig2 = sigma**2
    
    Q = Qfac(r,b,w)
    
    WfDer_a = r-a
    nev = 1 + np.exp(-Q) 
    WfDer_b = 1 / nev
    WfDer_w = w / nev

    return  WfDer_a, WfDer_b, WfDer_w

# Setting up the quantum force for the two-electron quantum dot, recall that it is a vector
@nb.njit
def QuantumForce(r:np.ndarray,a:np.ndarray,b:np.ndarray,w:np.ndarray) -> np.ndarray:

    # sigma=1.0
    # sig2 = sigma**2
    
    qforce = np.zeros((NumberParticles,Dimension), np.double)
    sum1 = np.zeros((NumberParticles,Dimension), np.double)
    
    Q = Qfac(r,b,w)
    
    for ih in range(NumberHidden):
        sum1 += w[:,:,ih]/(1+np.exp(-Q[ih]))
    
    # qforce = 2*(-(r-a)/sig2 + sum1/sig2)
    qforce = 2*(-(r-a) + sum1)
    
    return qforce

@nb.njit
def Qfac(r:np.ndarray,b:np.ndarray,w:np.ndarray) -> np.ndarray:
    Q = np.zeros((NumberHidden), np.double)
    temp = np.zeros((NumberHidden), np.double)
    
    for ih in range(NumberHidden):
        temp[ih] = (r*w[:,:,ih]).sum()
        
    Q = b + temp
    
    return Q
    
# Computing the derivative of the energy and the energy 
# @nb.njit
def EnergyMinimization(a:np.ndarray,b:np.ndarray,w:np.ndarray) -> tuple:
    # Parameters in the Fokker-Planck simulation of the quantum force
    D = 0.5
    TimeStep = 0.05
    # positions
    PositionOld = np.zeros((NumberParticles,Dimension), np.double)
    PositionNew = np.zeros((NumberParticles,Dimension), np.double)
    # Quantum force
    QuantumForceOld = np.zeros((NumberParticles,Dimension), np.double)
    QuantumForceNew = np.zeros((NumberParticles,Dimension), np.double)

    # seed for rng generator 
    # seed()
    energy = 0.0
    DeltaE = 0.0

    EnergyDer_a,EnergyDer_b,EnergyDer_w = np.zeros_like(a),np.zeros_like(b),np.zeros_like(w)
    DeltaPsi_a,DeltaPsi_b,DeltaPsi_w = np.zeros_like(a),np.zeros_like(b),np.zeros_like(w)
    DerivativePsiE_a,DerivativePsiE_b,DerivativePsiE_w = np.zeros_like(a),np.zeros_like(b),np.zeros_like(w)

    
    #Initial position
    for i in range(NumberParticles):
        for j in range(Dimension):
            PositionOld[i,j] = normalvariate(0.0,1.0)*np.sqrt(TimeStep)
    wfold = WaveFunction(PositionOld,a,b,w)
    QuantumForceOld = QuantumForce(PositionOld,a,b,w)

    #Loop over MC MCcycles
    for MCcycle in range(NumberMCcycles):
        #Trial position moving one particle at the time
        for i in range(NumberParticles):
            for j in range(Dimension):
                PositionNew[i,j] = PositionOld[i,j]+normalvariate(0.0,1.0)*np.sqrt(TimeStep)+\
                                       QuantumForceOld[i,j]*TimeStep*D
            wfnew = WaveFunction(PositionNew,a,b,w)
            QuantumForceNew = QuantumForce(PositionNew,a,b,w)
            
            GreensFunction = 0.0
            for j in range(Dimension):
                GreensFunction += 0.5*(QuantumForceOld[i,j]+QuantumForceNew[i,j])*\
                                      (D*TimeStep*0.5*(QuantumForceOld[i,j]-QuantumForceNew[i,j])-\
                                      PositionNew[i,j]+PositionOld[i,j])
      
            GreensFunction = np.exp(GreensFunction)
            ProbabilityRatio = GreensFunction*wfnew**2/wfold**2
            #Metropolis-Hastings test to see whether we accept the move
            if random() <= ProbabilityRatio:
                for j in range(Dimension):
                    PositionOld[i,j] = PositionNew[i,j]
                    QuantumForceOld[i,j] = QuantumForceNew[i,j]
                wfold = wfnew

        
        DeltaE = LocalEnergy(PositionOld,a,b,w)
        DerPsi = DerivativeWFansatz(PositionOld,a,b,w)
        
        if MCcycle > NumberMCcycles*equ_frac:
            DeltaPsi_a += DerPsi[0]
            DeltaPsi_b += DerPsi[1]
            DeltaPsi_w += DerPsi[2]
            
            energy += DeltaE
            if Printout: 
                outfile.write('%f\n' %(energy/(MCcycle-int(NumberMCcycles*equ_frac)+1.0)))
    
            DerivativePsiE_a += DerPsi[0]*DeltaE
            DerivativePsiE_b += DerPsi[1]*DeltaE
            DerivativePsiE_w += DerPsi[2]*DeltaE
            
    # We calculate mean values
    fraq = NumberMCcycles-NumberMCcycles*equ_frac
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
    return energy, [EnergyDer_a,EnergyDer_b,EnergyDer_w]


tot_time = time.time()
#Here starts the main program with variable declarations
np.random.seed(2)
seed(2)
NumberParticles = 2
Dimension = 2
NumberHidden = 2
equ_frac = 0.1


# guess for parameters
# a=np.random.normal(loc=0.0, scale=0.001, size=(NumberParticles,Dimension))
# b=np.random.normal(loc=0.0, scale=0.001, size=(NumberHidden))
# w=np.random.normal(loc=0.0, scale=0.001, size=(NumberParticles,Dimension,NumberHidden))
a=np.random.normal(loc=0.0, scale=.5, size=(NumberParticles,Dimension))
b=np.random.normal(loc=0.0, scale=.5, size=(NumberHidden))
w=np.random.normal(loc=0.0, scale=.5, size=(NumberParticles,Dimension,NumberHidden))

# Set up iteration using stochastic gradient method
Energy = 0
EDerivative = [np.zeros_like(a),np.zeros_like(b),np.zeros_like(w)]

# Learning rate eta, max iterations
eta = 0.05
MaxIterations = 100
NumberMCcycles = 10000
Printout = True
np.seterr(invalid='raise')
Energies = np.zeros(MaxIterations)
da = np.zeros(MaxIterations)
times = np.zeros(MaxIterations)
EnergyDerivatives1 = np.zeros(MaxIterations)
EnergyDerivatives2 = np.zeros(MaxIterations)
gamma = 0.9
momentum_a = np.zeros_like(a)
momentum_b = np.zeros_like(b)
momentum_w = np.zeros_like(w)

perc = -1
message = 'PROGRESS:'

for iteration in range(MaxIterations):
    if int(100*iteration/MaxIterations) > perc:
        perc = int(100*iteration/MaxIterations)
        print(f'\r{message} {perc:3.0f}%', end = '')
    timing = time.time()
    
    Energy, EDerivative = EnergyMinimization(a-momentum_a*gamma,b-momentum_b*gamma,w-momentum_w*gamma)
    # Energy, EDerivative = EnergyMinimization(a,b,w)

    momentum_a = momentum_a*gamma + eta*EDerivative[0]
    momentum_b = momentum_b*gamma + eta*EDerivative[1]
    momentum_w = momentum_w*gamma + eta*EDerivative[2]
    a -= momentum_a
    b -= momentum_b
    w -= momentum_w

    Energies[iteration] = Energy
    da[iteration] = np.sum(np.abs(momentum_a))
    times[iteration] = time.time() - timing
    


print(f'\r{message} {100:3.0f}%')

#nice printout with Pandas
pd.set_option('max_columns', 6)
data ={'Energy':Energies, 'da':da, 'Time':times}#,'A Derivative':EnergyDerivatives1,'B Derivative':EnergyDerivatives2,'Weights Derivative':EnergyDerivatives3}


if interaction:
    print("Interaction results:")
else:
    print("No interaction results:")

frame = pd.DataFrame(data)
print(frame)
print("Average energy: {}. Lowest: {} at {}.".format(np.mean(Energies), np.min(Energies), np.where(np.min(Energies)==Energies)[0][0]))
print("Total elapsed time: {}s".format(time.time() - tot_time))

outfile.close()

plt.plot(range(0,MaxIterations), Energies)
plt.plot(np.where(np.min(Energies)==Energies)[0][0], np.min(Energies), "o", label="Minimum Energy")
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Energy")
# plt.legend(["Energy", "Minimum Energy"])
plt.legend()
if Printout:
    plt.savefig("res/energy_"+str(interaction)+".pdf")
plt.show()



