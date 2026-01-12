import numpy as np
import matplotlib.pyplot as plt

# define function
def f(t,u):
    return -(t - a) * u

# analytical solution
def u_analytical(T, u_0, a):
    return u_0 * np.exp(-0.5 * T**2 + a * T)


def euler(f, u_0, dt, t_0=0, t_end=3):
    """
    f: function
    u_0: initial value
    dt: step size
    t_0: start time
    t_end: end time
    """
    # time array
    T = np.arange(t_0, t_end + dt, dt)
    # iteration process
    u = np.empty_like(T)
    u[0] = u_0
    for n in range(1, len(T)):
        tn = T[n-1]
        u[n] = dt * f(tn, u[n-1]) + u[n-1]
    return u, T

if __name__ == "__main__":
    t_0 = 0
    u_0 = 2
    t = 3
    dt = 0.001
    a = 2

    u, T = euler(f, u_0, dt, t_0, t)

    u_ana = []
    u_ana = u_analytical(T, u_0, a)


    plt.figure()
    plt.plot(T, u, label="u_num")
    plt.plot(T, u_ana, label="u_ana")
    plt.legend()
    plt.show()