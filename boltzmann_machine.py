import numpy as np
import numba as nb
from random import random, seed, normalvariate
import time
import pandas as pd
from pandas import DataFrame

np.random.seed(2)
@nb.jit
class Boltzmann_Machine:

    def __init__(self):
        self.particles = 2
        self.hidden = 2
        self.dimension = 2
        self.interaction = True
        self.print_out = True
        self.equilibration_fraction = 0.1
        self.mc_cycles = 10000

        self.a = np.random.normal(loc=0.0, scale=1, size=(self.particles, self.dimension))
        self.b = np.random.normal(loc=0.0, scale=1, size=(self.hidden))
        self.w = np.random.normal(loc=0.0, scale=1, size=(self.particles, self.dimension, self.hidden))

        self.energies = 0
        self.elapsed = 0

    def set_interaction(self, x: bool) -> None:
        self.interaction = x

    def set_print(self, x: bool) -> None:
        self.print_out = x


    @nb.njit
    def Qfac(self, r: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        hidden = self.hidden

        Q = np.zeros((hidden), np.double)
        temp = np.zeros((hidden), np.double)

        for ih in range(hidden):
            temp[ih] = (r * w[:, :, ih]).sum()

        Q = b + temp

        return Q

    @nb.njit
    def wave_function(self, r:np.ndarray, a:np.ndarray, b:np.ndarray, w:np.ndarray) -> float:
        Psi1 = 0.0
        Psi2 = 1.0
        Q = self.Qfac(r, b, w)

        for iq in range(self.particles):
            for ix in range(self.dimension):
                Psi1 += (r[iq, ix] - a[iq, ix])**2

        for ih in range(self.hidden):
            Psi2 *= (1.0 + np.exp(Q[ih]))

        Psi1 = np.exp(-Psi1 / 2)

        return Psi1 * Psi2

    @nb.njit
    def local_energy(self, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
        # sigma=1.0
        # sig2 = sigma**2
        locenergy = 0.0

        Q = self.Qfac(r, b, w)

        for iq in range(self.particles):
            for ix in range(self.dimension):
                sum1 = 0.0
                sum2 = 0.0
                for ih in range(self.hidden):
                    sum1 += w[iq, ix, ih] / (1 + np.exp(-Q[ih]))
                    sum2 += w[iq, ix, ih]**2 * np.exp(-Q[ih]) / (1.0 + np.exp(-Q[ih]))**2

                dlnpsi1 = -(r[iq, ix] - a[iq, ix]) + sum1
                dlnpsi2 = -1 + sum2
                locenergy += 0.5 * (- pow(dlnpsi1, 2) - dlnpsi2 + r[iq, ix]**2)

        if(self.interaction is True):
            for iq1 in range(self.particles):
                for iq2 in range(iq1):
                    distance = 0.0
                    for ix in range(self.dimension):
                        distance += (r[iq1, ix] - r[iq2, ix])**2

                    locenergy += (1 / np.sqrt(distance))

        return locenergy


    @nb.njit
    def wave_function_derivatives(self, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple:
        Q = self.Qfac(r, b, w)

        WfDer_a = r - a
        nev = 1 + np.exp(-Q)  #todo: test if correct
        WfDer_b = 1 / nev
        WfDer_w = w / nev

        return WfDer_a, WfDer_b, WfDer_w


    @nb.njit
    def quantum_force(self, r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
        qforce = np.zeros((self.particles, self.dimension), np.double)
        sum1 = np.zeros((self.particles, self.dimension), np.double)

        Q = self.Qfac(r, b, w)

        for ih in range(self.hidden):
            sum1 += w[:, :, ih] / (1 + np.exp(-Q[ih]))

        qforce = 2 * (-(r - a) + sum1)

        return qforce



    def energy_minimization(self, a: np.ndarray, b: np.ndarray, w: np.ndarray) -> tuple:
        # Parameters in the Fokker-Planck simulation of the quantum force
        D = 0.5
        TimeStep = 0.05
        # positions
        PositionOld = np.zeros((self.particles, self.dimension), np.double)
        PositionNew = np.zeros((self.particles, self.dimension), np.double)
        # Quantum force
        QuantumForceOld = np.zeros((self.particles, self.dimension), np.double)
        QuantumForceNew = np.zeros((self.particles, self.dimension), np.double)

        energy = 0.0
        DeltaE = 0.0

        EnergyDer_a, EnergyDer_b, EnergyDer_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)
        DeltaPsi_a, DeltaPsi_b, DeltaPsi_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)
        DerivativePsiE_a, DerivativePsiE_b, DerivativePsiE_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)

        #Initial position
        for i in range(self.particles):
            for j in range(self.dimension):
                PositionOld[i, j] = normalvariate(0.0, 1.0) * np.sqrt(TimeStep)
        wfold = self.wave_function(PositionOld, a, b, w)
        QuantumForceOld = self.quantum_force(PositionOld, a, b, w)

        #Loop over MC MCcycles
        for MCcycle in range(self.mc_cycles):
            #Trial position moving one particle at the time
            for i in range(self.particles):
                for j in range(self.dimension):
                    PositionNew[i, j] = PositionOld[i,j] + normalvariate(0.0, 1.0) * np.sqrt(TimeStep) + QuantumForceOld[i, j] * TimeStep * D
                wfnew = self.wave_function(PositionNew, a, b, w)
                QuantumForceNew = self.quantum_force(PositionNew, a, b, w)

                GreensFunction = 0.0
                for j in range(Dimension):
                    GreensFunction += 0.5*(QuantumForceOld[i, j] + QuantumForceNew[i, j])*\
                                          (D*TimeStep*0.5*(QuantumForceOld[i,j]-QuantumForceNew[i,j])-\
                                          PositionNew[i,j]+PositionOld[i,j])

                GreensFunction = np.exp(GreensFunction)
                ProbabilityRatio = GreensFunction*wfnew**2/wfold**2
                #Metropolis-Hastings test to see whether we accept the move
                if random() <= ProbabilityRatio:
                    for j in range(self.dimension):
                        PositionOld[i, j] = PositionNew[i, j]
                        QuantumForceOld[i, j] = QuantumForceNew[i, j]
                    wfold = wfnew

            DeltaE = self.local_energy(PositionOld, a, b, w)
            DerPsi = self.wave_function_derivatives(PositionOld, a, b, w)

            if MCcycle > (self.mc_cycles * self.equilibration_fraction):
                DeltaPsi_a += DerPsi[0]
                DeltaPsi_b += DerPsi[1]
                DeltaPsi_w += DerPsi[2]

                energy += DeltaE
                if self.print_out:
                    outfile.write('%f\n' %(energy / (MCcycle - int(self.mc_cycles * self.equilibration_fraction) + 1.0)))

                DerivativePsiE_a += DerPsi[0] * DeltaE
                DerivativePsiE_b += DerPsi[1] * DeltaE
                DerivativePsiE_w += DerPsi[2] * DeltaE

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


    def write_out(self, filename = "class_test.dat"):
        outfile = open(filename, 'w')
        print(f'\r{message} {100:3.0f}%')

        #nice printout with Pandas
        pd.set_option('max_columns', 6)
        data ={'Energy': self.energies, 'da': da, 'Time': times}  #,'A Derivative':EnergyDerivatives1,'B Derivative':EnergyDerivatives2,'Weights Derivative':EnergyDerivatives3}

        frame = pd.DataFrame(data)
        print(frame)
        print("Average energy: {}. Lowest: {}".format(np.mean(self.energies), np.min(self.energies)))
        print("Total elapsed time: {}s".format(self.elapsed))

        outfile.close()


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

        self.write_out()
        return Energies, elapsed


if __name__ == "__main__":
    test = Boltzmann_Machine()
    test.run()
