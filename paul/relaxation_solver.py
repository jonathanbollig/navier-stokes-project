import numpy as np
import matplotlib.pyplot as plt
from grid1D import Grid1D
from euler_solver import EulerSolver

class RelaxationSolver:
    def __init__(self, grid, func, dt, u_boundaries, t0, t_end):
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
        self.grid = grid
        self.func = func
        self.dt = dt
        self.u_left = u_boundaries[0]
        self.u_right = u_boundaries[1]
        self.t0 = t0
        self.t_end = t_end
    
    def euler_rhs(self, t, u):
        self.grid.f = u
        Lu = self.grid.second_derivative()
        f_x = self.func(self.grid.xs)
        du_dt = np.zeros_like(u)
        du_dt[1:-1] = Lu - f_x[1:-1]

        # fix boundary conditions
        du_dt[0] = 0
        du_dt[-1] = 0
        return du_dt
    
    def solve(self, dt):
        u0 = np.linspace(self.u_left, self.u_right, len(self.grid.xs))
        solver = EulerSolver(self.euler_rhs, u0, self.t0, self.t_end)

        u_num, T = solver.solve(dt)
    
        return u_num, T
    
    def analytical(self, X):
        u_ana = - np.cos(X) + 2 / (3 * np.pi) * X - 1
        return u_ana

if __name__ == "__main__":
    my_grid = Grid1D(100, (0, 3 * np.pi))
    u_boundaries = (-2, 2)
    function = lambda x: np.cos(x)
    dt = 1e-3
    X = my_grid.xs


    solver = RelaxationSolver(my_grid, function, dt, u_boundaries, 0, 10)
    u_num, T = solver.solve(dt)
    
    u_ana = solver.analytical(X)

    plt.figure()
    plt.plot(X, u_num[-1], label = "numerical")
    plt.plot(X, u_ana, label ="analytical")
    plt.legend()
    plt.show()