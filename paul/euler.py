import numpy as np
import matplotlib.pyplot as plt


a=1

def f(t,u):
    return -(t - a) * u

u = []

def euler(f, t, u_0, dt):
    """
    u: function
    dt: step size
    u_0 initial value
    """
    T = np.arange(t_0, t + dt, dt)
    #iteration process
    u = np.empty_like(T)
    u[0] = u_0
    for n in range(1, len(T)):
        tn = T[n-1]
        u[n] = dt * f(tn, u[n-1]) + u[n-1]
    return u, T



t_0 = 0
u_0 = 2
t = 3
dt = 0.001

u, T = euler(f, t, u_0, dt)

u_a = []

def u_an(t, T):
    for t in T:
        u_a.append(np.exp(-1/2 * t**2 + a * t) * u_0)
    return u_a

u_a = u_an(t, T)



plt.figure()
plt.plot(T, u, label="u_num")
plt.plot(T, u_a, label="u_ana")
plt.legend()
plt.show()