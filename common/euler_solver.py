import numpy as np
import matplotlib.pyplot as plt

class EulerSolver:
    def __init__(self, func, u0, t0, t_end):
        """
        :param func: function f(t, u)
        :param u0: initial value
        :param t0: start time
        :param t_end: end time
        """
        self.f = func
        self.u0 = np.array(u0)
        self.t0 = t0
        self.t_end = t_end

    def solve(self, dt):
        """
        :param dt: step size
        """
        # time array
        T = np.arange(self.t0, self.t_end + 1/2*dt, dt)
        # iteration process
        if self.u0.ndim == 0:
            u = np.empty_like(T)
        else:
            u = np.zeros((len(T), len(self.u0)))
        u[0] = self.u0
        for n in range(1, len(T)):
            tn = T[n-1]
            u[n] = dt * self.f(tn, u[n-1]) + u[n-1]
        return u, T

    def convergence_test(self, dt, ana_func):
        """
        Docstring for convergence_test
        
        :param dt: step size
        :param ana_func: analytical solution
        """
        u_num, T = self.solve(dt)
        u_num2, T2 = self.solve(dt/2)
        frac = abs(ana_func[-1] - u_num[-1]) / abs(ana_func[-1] - u_num2[-1])
        p = np.log2(frac)
        return p

    def self_convergence_test(self, dt):
        u1, T1 = self.solve(dt)
        u2, T2 = self.solve(dt/2)
        u3, T3 = self.solve(dt/4)
        frac = abs(u1[-1] - u2[-1]) / abs(u2[-1] - u3[-1])
        p = np.log2(frac)
        return p
    
    def plot(self, T, u_num, u_ana, dt):
        plt.figure()
        plt.plot(T, u_num, label="numerical")
        plt.plot(T, u_ana, label="analytical")
        plt.title(f"solution for dt={dt}")
        plt.xlabel("t")
        plt.ylabel("u(t)")
        plt.legend()
        plt.show()

if __name__ == "__main__":
    a = 2
    u0 = 2
    t0 = 0
    t_end = 10

    f = lambda t, u: -(t - a) * u

    u_analytical = lambda T: u0 * np.exp(-0.5 * T**2 + a * T)

    solver = EulerSolver(f, u0, t0, t_end)

    dt = 1e-3
    u_num, T = solver.solve(dt)
    u_ana = u_analytical(T)

    p_conv = solver.convergence_test(dt, u_ana)
    p_self = solver.self_convergence_test(dt)

    print(f"order of convergence p = {p_conv} \\ order of self convergence p_self = {p_self} ")

    plt.figure()
    plt.plot(T, u_num, label="numerical")
    plt.plot(T, u_ana, label="analytical")
    plt.title(f"solution for dt={dt}")
    plt.xlabel("t")
    plt.ylabel("u(t)")
    plt.legend()
    plt.show()