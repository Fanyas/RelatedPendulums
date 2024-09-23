import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

g = 9.8
m = 4
L = 8
L1 = 4
B = 0
K = 1
Phi1 = 1
Phi2 = -1

state = [Phi1, Phi2, 0, 0]

def system(t, state):
    phi1, phi2, d_Phi1_dt, d_Phi2_dt = state
    d2_Phi1_dt2 = -g / L * phi1 + K * L1 ** 2 / m / L ** 2 * (phi2 - phi1) - 2 * B / L * d_Phi1_dt
    d2_Phi2_dt2 = -g / L * phi2 - K * L1 ** 2 / m / L ** 2 * (phi2 - phi1) - 2 * B / L * d_Phi2_dt
    return [d_Phi1_dt, d_Phi2_dt, d2_Phi1_dt2, d2_Phi2_dt2]

t_end = 200
t_eval = np.arange(0, t_end, 0.01)

sol = solve_ivp(system, [0, t_end], state, t_eval=t_eval)

for i in range(len(sol.t)):
    print(f"Time: {sol.t[i]:.2f}, Phi1: {sol.y[0, i]:.2f}, Phi2: {sol.y[1, i]:.2f}, omega1: {sol.y[2, i]:.2f}, omega2: {sol.y[3, i]:.2f}")

OMEGA1 = (g / L) ** 0.5
OMEGA2 = (g / L + 2 * K * L1 ** 2 / m / L ** 2) ** 0.5

figure, axis = plt.subplots(2, 1)
axis[0].plot(sol.t, sol.y[0], label="$\\phi_1(t)$")
axis[0].plot(sol.t, sol.y[1], label="$\\phi_2(t)$")
axis[0].grid(True)
axis[0].legend(loc='upper right')

axis[1].plot(sol.t, sol.y[2], label="$\\omega_1(t)$")
axis[1].plot(sol.t, sol.y[3], label="$\\omega_2(t)$")
axis[1].grid(True)
axis[1].legend(loc='upper right')

axis[0].set_ylabel('$\\phi$ (град)')
axis[1].set_ylabel('$\\omega$ (град/с)')
axis[1].set_xlabel('Время (с)')

figure.suptitle(
 "$\\Omega_{{1}} = {0:5.2f} Гц,\\Omega_{{2}} = {1:5.2f} Гц$".format(OMEGA1, OMEGA2))

plt.show()