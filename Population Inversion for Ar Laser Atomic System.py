import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def Calc(x, t):
    # constants
    c = 3e8  # speed of light
    h = 6.62608e-34  # Planck constant

    # parameters
    P_max = 27e-3  # pumping energy

    lambda_43 = 0.6328e-6  # wavelength of emitted light during transition between levels 4-3
    lambda_14 = 0.3888e-6  # wavelength of emitted light during transition between levels 1-4
    lambda_13 = 63e-9  # wavelength of emitted light during transition between levels 1-3
    lambda_12 = 63.5e-9  # wavelength of emitted light during transition between levels 1-2

    nu_43 = c/lambda_43  # frequency of the during transition between levels 4-3
    nu_14 = c/lambda_14  # frequency of the during transition between levels 1-4
    nu_13 = c/lambda_13  # frequency of the during transition between levels 1-3
    nu_12 = c/lambda_12  # frequency of the during transition between levels 1-2

    delta_nu_1 = 0.03e10  # broadering of the atomic level 1
    delta_nu_4 = 150e10  # broadering of the atomic level 4

    tau_1 = 2e-3  # lifetime of the the atomic level 1
    tau_4 = 4e-3  # lifetime of the the atomic level 4

    tau_21 = 1.53e-3  # lifetime of the transition between levels 2-1
    tau_32 = 30e-9  # lifetime of the transition between levels 3-2
    tau_41 = 4e-3  # lifetime of the transition between levels 4-1

    g3 = 4  # degeneration of the level 3
    g4 = 2  # degeneration of the level 4

    # effective cross section of the transition between levels 1-2
    sigma_12 = ((c**2)/(4 * math.pi**2 * nu_12**2 * tau_1 * delta_nu_1))*10**4
    # effective cross section of the transition between levels 1-3
    sigma_13 = ((c**2)/(4 * math.pi**2 * nu_13**2 * tau_1 * delta_nu_1))*10**4
    # effective cross section of the transition between levels 1-4
    sigma_14 = ((c**2)/(4 * math.pi**2 * nu_14**2 * tau_1 * delta_nu_1))*10**4
    # effective cross section of the transition between levels 4-3
    sigma_43 = ((c**2)/(4 * math.pi**2 * nu_43**2 * tau_4 * delta_nu_4))*10**4

    # probability of the transition betweel levels 1-2
    w12 = (lambda_12/h) * sigma_12 * P_max
    # probability of the transition betweel levels 1-3
    w13 = (lambda_13/h) * sigma_13 * P_max
    # probability of the transition betweel levels 1-4
    w14 = (lambda_14/h) * sigma_14 * P_max
    # probability of the transition betweel levels 4-3
    w43 = (lambda_43/h) * sigma_43 * P_max

    # initial conditions
    n1 = x[0]  # number of the atoms on level 1
    n2 = x[1]  # number of the atoms on level 2
    n3 = x[2]  # number of the atoms on level 3
    n4 = x[3]  # number of the atoms on level 4

    # system of the rate equations for four-level Argon laser system
    dn1dt = -w14*(n1-n4) + n4/tau_41 - w12*(n1-n2) - w13*(n1-n3) + n2/tau_21
    dn2dt = w12*(n1-n2) + n3/tau_32 - n2/tau_21
    dn3dt = w13*(n1-n3) - n3/tau_32 + w43*(n4-n3*(g4/g3))
    dn4dt = w14*(n1-n4) - n4/tau_41 - w43*(n4-n3*(g4/g3))
    return [dn1dt, dn2dt, dn3dt, dn4dt]


#  graphical solution of the system of the rate equations for levels 2 and 3
T = 6e-7
x0 = [8.22078e16, 0, 0, 0]
t = np.linspace(2e-3*T, T, 1000)
x = odeint(Calc, x0, t)

n1 = x[:, 0]
n2 = x[:, 1]
n3 = x[:, 2]
n4 = x[:, 3]

plt.plot(t, n2, label='atomic level 2 (n2)')
plt.plot(t, n3, label='atomic level 3 (n3)')
plt.xlabel('Time')
plt.ylabel('Atom Population')
plt.title("Population Inversion for Ar Laser Atomic System")
plt.legend()
plt.show()
