import numpy as np 
import matplotlib.pyplot as plt

class Grid1D:
    def __init__(self, n=10, dim=(0, 1), n_ghost=2, f=None):
        self.n = n
        self.dim = dim
        self.n_ghost = n_ghost
        self.h = (dim[1] - dim[0])/n 
        self.xs = np.linspace(dim[0] + (-n_ghost+1/2)*self.h, dim[1] + (n_ghost-1/2)*self.h, n + 2*n_ghost)
        self.f = f  # could add sanity check if shape of f matches x_n 

    def set_f(self, values, boundary_condition):
        """
        boundary_condition defines the values at the ghost points, it can be:
        - dirichlet --> no condition is applied, since values include ghost cells
        - periodic --> value at N+1 equals value at 1 etc.
        - zeros --> ghost points are 0
        - vertical --> ghost point extend the last Value (e.g. 0,1,2 --> 0,0,1,2,2)
        Not implemented: Neumann
        """
        if boundary_condition == "dirichlet":
            self.f = values
        elif boundary_condition == "periodic":
            self.f = np.concatenate([values[-self.n_ghost:], values, values[:self.n_ghost]])
        elif boundary_condition == "zeros":
            self.f = np.concatenate([np.zeros(self.n_ghost), values, np.zeros(self.n_ghost)])
        elif boundary_condition == "vertical":
            self.f = np.concatenate([np.full(self.n_ghost, values[0]), values, np.full(self.n_ghost, values[-1])])
        else:
            raise ValueError(f"Boundary condition {boundary_condition} is not supported.")

    def calc_f(self, func, include_ghost_points=True):
        # if include_ghost_points is not True, it needs to be a boundary condition string, supported in set_f
        if include_ghost_points == True:
            values = func(self.xs)
            self.set_f(values, boundary_condition="dirichlet")
        else:
            values = func(self.xs[self.n_ghost:-self.n_ghost])  
            self.set_f(values, include_ghost_points)

    def plot(self, marker="x", linestyle="", extra_grids = []):
        fig, ax = plt.subplots()
        ax.plot(self.xs, self.f, marker=marker, linestyle=linestyle)
        for extra_grid in extra_grids:
            ax.plot(extra_grid.xs, extra_grid.f, marker=marker, linestyle=linestyle)
        ax.axvline(self.dim[0], c="black", linestyle=":")
        ax.axvline(self.dim[1], c="black", linestyle=":")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid()
        plt.show()

    def derivative(self, method="center"):
        """
        returns a new instances of a 1d grid class, containing the gradient as f
        allowed methods:
        - numpy  --> uses numpy gradient --> center differential
        - center
        - forward
        - backward 
        """
        if method == "numpy":
            x_grad = np.gradient(self.f, self.h) # explenation: https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
            x_grad_grid = Grid1D(self.n, self.dim, self.n_ghost, x_grad)
        elif method == "center":
            x_grad = (self.f[2:] - self.f[:-2])/2/self.h
            rhs = (self.f[-1]-self.f[-2])/self.h
            lhs = (self.f[1]-self.f[0])/self.h
            x_grad = np.concatenate(([lhs], x_grad, [rhs]))
            x_grad_grid = Grid1D(self.n, self.dim, self.n_ghost, x_grad)
        elif method == "forward":
            raise NotImplementedError()
        elif method == "backward":
            raise NotImplementedError()
        else:
            raise ValueError(f"Method {method} is not supported for 'derivative'")
        return x_grad_grid
    
    def second_derivative(self, method="center"):
        if method == "numpy":
            x_grad = np.gradient(self.f, self.h) # explenation: https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
            x_grad_grid = Grid1D(self.n, self.dim, self.n_ghost, x_grad)
            d2x = np.gradient(x_grad, self.h)
            d2x_grid = Grid1D(self.n, self.dim, self.n_ghost, d2x)
        elif method == "center":
            d2x = (self.f[2:] - 2*self.f[1:-1] + self.f[:-2]) / (self.h**2)
            d2x_grid = Grid1D(self.n, self.dim, self.n_ghost, d2x)
        elif method == "forward":
            raise NotImplementedError()
        elif method == "backward":
            raise NotImplementedError()
        else:
            raise ValueError(f"Method {method} is not supported for 'derivative'")
        return d2x

if __name__ == "__main__":
    my_grid = Grid1D(20, (0, 1.5*np.pi), 2)
    my_grid.calc_f(np.sin, "periodic")
    my_grid.plot()
    d_dx = my_grid.derivative(method="numpy")
    d_dx2 = my_grid.derivative(method="center")

    # check if both methods yield the same result
    print(np.max(np.abs(d_dx.f-d_dx2.f)))

    # plot the gradient
    d_dx.plot()

    # plot both together
    my_grid.plot(extra_grids=[d_dx])