import numpy as np
import numba as nb
from random import random, seed, normalvariate
import time
import pandas as pd
from pandas import DataFrame
from dataclasses import dataclass
import numerical as num

np.random.seed(2)

class Boltzmann_Machine:

    def __init__(self, filename="class_test.dat"):
        self.particles = 2
        self.hidden = 2
        self.dimension = 2

        self.system = np.array([self.particles, self.dimension, self.hidden], dtype=np.int64)
        self.interaction = True
        self.print_out = True
        self.equilibration_fraction = 0.1
        self.mc_cycles = 10000
        self.outfile = open(filename, 'w')

        self.a = np.random.normal(loc=0.0, scale=1, size=(self.particles, self.dimension))
        self.b = np.random.normal(loc=0.0, scale=1, size=(self.hidden))
        self.w = np.random.normal(loc=0.0, scale=1, size=(self.particles, self.dimension, self.hidden))

        self.energies = 0
        self.elapsed = 0

    def set_interaction(self, x: bool) -> None:
        self.interaction = x

    def set_print(self, x: bool) -> None:
        self.print_out = x

    def Qfac(self, r: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        Q = num.QFac_helper(r, b, w, self.hidden) + b

        return Q

    def wave_function(self, r:np.ndarray, a:np.ndarray, b:np.ndarray, w:np.ndarray) -> float:
        Q = self.Qfac(r, b, w)

        return num.wave_function_helper(self.system, r, a, b, w, Q)

    def local_energy(self, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
        Q = self.Qfac(r, b, w)

        return num.local_energy_helper(self.system, r, a, b, w, Q, self.interaction)

    def wave_function_derivatives(self, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple:
        Q = self.Qfac(r, b, w)

        WfDer_a = r - a
        nev = 1 + np.exp(-Q)  #todo: test if correct
        WfDer_b = 1 / nev
        WfDer_w = w / nev

        return WfDer_a, WfDer_b, WfDer_w

    def quantum_force(self, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        Q = self.Qfac(r, b, w)

        return num.quantum_force_helper(self.system, r, a, b, w, Q)


    def energy_minimization(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple:
        # Parameters in the Fokker-Planck simulation of the quantum force
        D = 0.5
        TimeStep = 0.05
        # positions
        PositionOld = np.zeros((self.particles, self.dimension), np.double)
        # Quantum force
        QuantumForceOld = np.zeros((self.particles, self.dimension), np.double)

        EnergyDer_a, EnergyDer_b, EnergyDer_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)

        #Initial position
        PositionOld = num.minimisation_loop1(self.system, PositionOld, TimeStep)

        wfold = self.wave_function(PositionOld, a, b, w)
        QuantumForceOld = self.quantum_force(PositionOld, a, b, w)

        DerivativePsiE_a, DerivativePsiE_b, DerivativePsiE_w, DeltaPsi_a, DeltaPsi_b, DeltaPsi_w, energy = num.minimisation_loop2(self.system, a, b, w, self.mc_cycles, PositionOld, QuantumForceOld, wfold, TimeStep, self.equilibration_fraction, self.interaction, self.outfile, self.print_out)

        """
        DerivativePsiE_a = derivatives[0]
        DerivativePsiE_b = derivatives[1]
        DerivativePsiE_w = derivatives[2]

        DeltaPsi_a = deltas[0]
        DeltaPsi_b = deltas[1]
        DeltaPsi_w = deltas[2]"""

        # We calculate mean values
        fraq = self.mc_cycles - (self.mc_cycles * self.equilibration_fraction)
        energy /= fraq
        DerivativePsiE_a /= fraq
        DerivativePsiE_b /= fraq
        DerivativePsiE_w /= fraq
        DeltaPsi_a /= fraq
        DeltaPsi_b /= fraq
        DeltaPsi_w /= fraq
        EnergyDer_a = 2 * (DerivativePsiE_a - DeltaPsi_a * energy)
        EnergyDer_b = 2 * (DerivativePsiE_b - DeltaPsi_b * energy)
        EnergyDer_w = 2 * (DerivativePsiE_w - DeltaPsi_w * energy)

        return energy, [EnergyDer_a, EnergyDer_b, EnergyDer_w]

    def write_out(self, da, times):
        message = 'PROGRESS:'
        print(f'\r{message} {100:3.0f}%')

        #nice printout with Pandas
        pd.set_option('max_columns', 6)
        data = {'Energy': self.energies, 'da': da, 'Time': times} 

        frame = pd.DataFrame(data)
        print(frame)
        print("Average energy: {}. Lowest: {}".format(np.mean(self.energies), np.min(self.energies)))
        print("Total elapsed time: {}s".format(self.elapsed))

        self.outfile.close()


    def run(self):
        tot_time = time.time()
        #Here starts the main program with variable declarations
        # guess for parameters
        # Set up iteration using stochastic gradient method
        Energy = 0
        EDerivative = [np.zeros_like(self.a), np.zeros_like(self.b), np.zeros_like(self.w)]

        # Learning rate eta, max iterations, need to change to adaptive learning rate
        eta = 0.05
        MaxIterations = 100
        np.seterr(invalid='raise')
        Energies = np.zeros(MaxIterations)
        da = np.zeros(MaxIterations)
        times = np.zeros(MaxIterations)

        gamma = 0.9
        momentum_a = np.zeros_like(self.a)
        momentum_b = np.zeros_like(self.b)
        momentum_w = np.zeros_like(self.w)

        perc = -1
        message = 'PROGRESS:'

        for iteration in range(MaxIterations):
            if int(100 * iteration / MaxIterations) > perc:
                perc = int(100 * iteration / MaxIterations)
                print(f'\r{message} {perc:3.0f}%', end = '')

            timing = time.time()

            Energy, EDerivative = self.energy_minimization(self.a - (momentum_a * gamma), self.b - (momentum_b * gamma), self.w - (momentum_w * gamma))

            momentum_a = momentum_a * gamma + eta * EDerivative[0]
            momentum_b = momentum_b * gamma + eta * EDerivative[1]
            momentum_w = momentum_w * gamma + eta * EDerivative[2]
            self.a -= momentum_a
            self.b -= momentum_b
            self.w -= momentum_w

            Energies[iteration] = Energy
            da[iteration] = np.sum(np.abs(momentum_a))
            times[iteration] = time.time() - timing

        elapsed = time.time() - tot_time

        self.energies = Energies
        self.elapsed = elapsed

        self.write_out(da, times)
        return Energies, elapsed


if __name__ == "__main__":
    test = Boltzmann_Machine()
    test.run()
