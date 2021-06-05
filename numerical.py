#jit functions for Bolzmann class

import numpy as np
import numba as nb
from random import random, normalvariate
np.random.seed(2)

@nb.njit
def QFac_helper(r, b, w, hidden):
    temp = np.zeros((hidden), np.double)

    for ih in range(hidden):
        temp[ih] = (r * w[:, :, ih]).sum()

    return temp

@nb.njit
def wave_function_helper(system, r, a, b, w, Q):
    Psi1 = 0.0
    Psi2 = 1.0

    for iq in range(system[0]):
        for ix in range(system[1]):
            Psi1 += (r[iq, ix] - a[iq, ix])**2

    for ih in range(system[2]):
        Psi2 *= (1.0 + np.exp(Q[ih]))

    return (np.exp(-Psi1 / 2) * Psi2)

@nb.njit
def local_energy_helper(system, r, a, b, w, Q, interaction):
    locenergy = 0.0
    p = system[0]
    d = system[1]
    h = system[2]

    for iq in range(p):
        for ix in range(d):
            sum1 = 0.0
            sum2 = 0.0
            for ih in range(h):
                sum1 += w[iq, ix, ih] / (1 + np.exp(-Q[ih]))
                sum2 += w[iq, ix, ih]**2 * np.exp(-Q[ih]) / (1.0 + np.exp(-Q[ih]))**2

            dlnpsi1 = -(r[iq, ix] - a[iq, ix]) + sum1
            dlnpsi2 = -1 + sum2
            locenergy += 0.5 * (- pow(dlnpsi1, 2) - dlnpsi2 + r[iq, ix]**2)

    if(interaction is True):
        for iq1 in range(p):
            for iq2 in range(iq1):
                distance = 0.0
                for ix in range(d):
                    distance += (r[iq1, ix] - r[iq2, ix])**2

                locenergy += (1 / np.sqrt(distance))

    return locenergy


@nb.njit
def quantum_force_helper(system, r, a, b, w, Q):
    p = system[0]
    d = system[1]
    h = system[2]

    sum1 = np.zeros((p, d), np.double)

    for ih in range(h):
        sum1 += w[:, :, ih] / (1 + np.exp(-Q[ih]))

    return (2 * (-(r - a) + sum1))

@nb.njit
def minimisation_loop1(system, pos_old, time_step):
    p = system[0]
    d = system[1]

    for i in range(p):
        for j in range(d):
            pos_old[i, j] = normalvariate(0.0, 1.0) * np.sqrt(time_step)

    return pos_old

@nb.njit
def new_pos_loop(i, system, pos_old, qf_old, time_step):
    p = system[0]
    d = system[1]
    D = 0.5
    pos_new = np.zeros((p, d), np.double)

    #Trial position moving one particle at the time
    for j in range(d):
        pos_new[i, j] = pos_old[i, j] + normalvariate(0.0, 1.0) * np.sqrt(time_step) + qf_old[i, j] * time_step * D

    return pos_new


@nb.njit
def importance_sampling(i, d, qf_new, qf_old, pos_new, pos_old, wf_new, wf_old, time_step):
    D = 0.5
    GreensFunction = 0.0

    for j in range(d):
        term1 = 0.5 * (qf_old[i, j] + qf_new[i, j])
        qf_diff = qf_old[i, j] - qf_new[i, j]
        pos_add = pos_new[i, j] + pos_old[i, j]
        GreensFunction += term1 * (D * time_step * 0.5 * (qf_diff) - pos_add)

    GreensFunction = np.exp(GreensFunction)
    ProbabilityRatio = GreensFunction * wf_new**2 / wf_old**2

    #Metropolis-Hastings test to see whether we accept the move
    if np.random.uniform(0.0, 1.0) <= ProbabilityRatio:
        for j in range(d):
            pos_old[i, j] = pos_new[i, j]
            qf_old[i, j] = qf_new[i, j]
        wf_old = wf_new

    return pos_old, qf_old, wf_old

def wave_function_derivatives_clone(r: np.ndarray, a: np.ndarray, b: np.ndarray, w: np.ndarray, Q) -> tuple:
    WfDer_a = r - a
    nev = 1 + np.exp(-Q)  #todo: test if correct
    WfDer_b = 1 / nev
    WfDer_w = w / nev
    return WfDer_a, WfDer_b, WfDer_w

def minimisation_loop2(system, a, b, w, cycles, pos_old, qf_old, wf_old, time_step, equ_frac, interaction, outfile, print_out):

    energy = np.float(0.0)
    DeltaE = np.float(0.0)

    DeltaPsi_a, DeltaPsi_b, DeltaPsi_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)
    DerivativePsiE_a, DerivativePsiE_b, DerivativePsiE_w = np.zeros_like(a), np.zeros_like(b), np.zeros_like(w)

    p = system[0]
    d = system[1]

    #Loop over MC MCcycles
    for MCcycle in range(cycles):

        #Trial position moving one particle at the time
        for i in range(p):
            pos_new = new_pos_loop(i, system, pos_old, qf_old, time_step)

            Q = QFac_helper(pos_new, b, w, system[2]) + b

            wf_new = wave_function_helper(system, pos_new, a, b, w, Q)
            qf_new = quantum_force_helper(system, pos_new, a, b, w, Q)

            pos_old, qf_old, wf_old = importance_sampling(i, d, qf_new, qf_old, pos_new, pos_old, wf_new, wf_old, time_step)

        Q = QFac_helper(pos_new, b, w, system[2]) + b
        DeltaE = local_energy_helper(system, pos_old, a, b, w, Q, interaction)
        DerPsi = wave_function_derivatives_clone(pos_old, a, b, w, Q)

        if MCcycle > (cycles * equ_frac):
            DeltaPsi_a += DerPsi[0]
            DeltaPsi_b += DerPsi[1]
            DeltaPsi_w += DerPsi[2]
            energy += DeltaE

            if print_out:
                outfile.write('%f\n' %(energy / (MCcycle - int(cycles * equ_frac) + 1.0)))

            DerivativePsiE_a += DerPsi[0] * DeltaE
            DerivativePsiE_b += DerPsi[1] * DeltaE
            DerivativePsiE_w += DerPsi[2] * DeltaE

    #return np.array([DerivativePsiE_a, DerivativePsiE_b, DerivativePsiE_w]), np.array([DeltaPsi_a, DeltaPsi_b, DeltaPsi_w])
    return DerivativePsiE_a, DerivativePsiE_b, DerivativePsiE_w, DeltaPsi_a, DeltaPsi_b, DeltaPsi_w, energy


