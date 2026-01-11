from grid1D import Grid1D
import numpy as np
import matplotlib.pyplot as plt
from euler_solver import EulerSolver

class WaveEqSolver:
    def __init__(self, L, N, func, dt, u_boundaries, t0, t_end):
        """
        Docstring for __init__
        
        :param self: Description
        :param grid: Description
        :param func: Description
        :param dt: Description
        :param u0: Tupel at x[0] and x[-1]
        :param t0: Description
        :param t_end: Description
        """
        x = np.linspace(0, L, N)
        y = np.linspace(0, L, N)
        self.X, self.Y = np.meshgrid(x, y)
        self.h = x[1] - x[0]
        self.f = func
        self.u_start = func(self.X, self.Y)
        self.PI_start = np.zeros_like(self.u_start)
        self.y_0 = np.array([self.u_start, self.PI_start])
        self.dt = dt
        self.u_left = u_boundaries[0]
        self.u_right = u_boundaries[1]
        self.t0 = t0
        self.t_end = t_end
        self.T = np.arange(t0, t_end + dt, dt)

    def second_derivative(self, u):
        d2x = (u[1:-1,2:] - 2*u[1:-1,1:-1] + u[1:-1,:-2]) / (self.h**2)
        d2y = (u[2:,1:-1] - 2*u[1:-1,1:-1] + u[:-2,1:-1]) / (self.h**2)
        laplace = np.zeros_like(self.u_start)
        laplace[1:-1, 1:-1] = d2x + d2y
        return laplace

    def euler_rhs(self, state):
        u = state[0]
        PI = state[1]
        du_dt = PI
        c = 1
        dPI_dt = np.zeros_like(u)
        dPI_dt = c**2 * self.second_derivative(u)
        state_dt = np.array([du_dt, dPI_dt])
        return state_dt

    def line_solve(self):
        PI = EulerSolver(self.euler_rhs, self.y_0, self.t0, self.t_end)

        result, T = PI.solve(self.dt)

        u = result[:, 0, :, :] 
        
        return u, T

u0 = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)

wave_eq = WaveEqSolver(2, 20, u0, dt=1e-3, u_boundaries=(0,0), t0=0, t_end=10)

u_xy_num, T = wave_eq.line_solve()

u_ana = np.sin(np.pi * wave_eq.X) * np.sin(np.pi * wave_eq.Y) * np.cos(np.sqrt(2)*np.pi * T)


plt.figure()
plt.imshow(u_xy_num, T)

plt.figure()
plt.imshow(u_ana, T)
plt.plot()