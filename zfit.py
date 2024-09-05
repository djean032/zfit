import numpy as np
from scipy.integrate import odeint, solve_ivp
from scipy import constants
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from time import time
 
def rates(t, state, I, omega, sigma_01, sigma_12, sigma_34, t10, t13, t21, t30, t43):
    Ns0, Ns1, Ns2, Nt1, Nt2 = state
    dNs0 = -(sigma_01 * Ns0 * I) / (constants.hbar * omega) + Ns1 / t10 + Nt1 / t30
    dNs1 = (sigma_01 * Ns0 * I) / (constants.hbar * omega) - Ns1 / t10 - sigma_12 * Ns1 * I / (constants.hbar * omega) + Ns2 / t21 - Ns1 / t13
    dNs2 = (sigma_12 * Ns1 * I) / (constants.hbar * omega) - Ns2 / t21
    dNt1 = -(sigma_34 * Nt1 * I) / (constants.hbar * omega) + Nt2 / t43 + Ns1 / t13 - Nt1 / t30
    dNt2 = (sigma_34 * Nt1 * I) / (constants.hbar * omega) - Nt2 / t43
    return [dNs0, dNs1, dNs2, dNt1, dNt2]
 
I = 1.0
omega = 1.0
sigma_01 = 10.0
sigma_12 = 10.0
sigma_34 = 10.0
t10 = 1.0
t13 = 1.0
t21 = 1.0
t30 = 1.0
t43 = 1.0
 
p = (I, omega, sigma_01, sigma_12, sigma_34, t10, t13, t21, t30, t43)
 
y0 = [1.0, 0.0, 0.0, 0.0, 0.0]  # Initial state of the system


t_span = (0.0, 8.0)
t = np.arange(0.0, 8.0, 0.01)
 
begin_odeint = time()
result_odeint = odeint(rates, y0, t, p, tfirst=True)
end_odeint = time()
begin_ivp = time()
result_solve_ivp = solve_ivp(rates, t_span, y0, args=p, method='LSODA', t_eval=t)
end_ivp = time()
print("odeint: ", end_odeint - begin_odeint)
print("solve_ivp: ", end_ivp - begin_ivp)
print(result_odeint)
