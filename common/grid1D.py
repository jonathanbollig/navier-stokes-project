import numpy as np 
import matplotlib.pyplot as plt

class Grid1D:
    def __init__(self, x_n=10, x_dim=(0, 1), n_ghost=2, f=None):
        self.x_n = x_n
        self.x_dim = x_dim
        self.n_ghost = n_ghost
        self.dx = (x_dim[1] - x_dim[0])/x_n 
        self.xs = np.linspace(x_dim[0] + (-n_ghost+1/2)*self.dx, x_dim[1] + (n_ghost-1/2)*self.dx, x_n + 2*n_ghost)
        self.f = f  # could add sanity check if shape of f matches x_n 

    def calc_f(self, func):
        # maybe "func" should be saved as well? 
        self.f = func(self.xs)

    def imshow(self, marker="x", linestyle="", extra_grids = []):
        fig, ax = plt.subplots()
        ax.plot(self.xs, self.f, marker=marker, linestyle=linestyle)
        for extra_grid in extra_grids:
            ax.plot(extra_grid.xs, extra_grid.f, marker=marker, linestyle=linestyle)
        ax.axvline(self.x_dim[0], c="black", linestyle=":")
        ax.axvline(self.x_dim[1], c="black", linestyle=":")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.grid()
        plt.show()

    def center_difference(self, method="manual"):
        """
        returns a new instances of a 1d grid class, containing the gradient as f
        allowed methods:
        - numpy  --> uses numpy gradient --> center differential
        - manual 
        """
        if method == "numpy":
            x_grad = np.gradient(self.f, self.dx) # explenation: https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
        elif method == "manual":
            x_grad = (self.f[2:] - self.f[:-2])/2/self.dx
            rhs = (self.f[-1]-self.f[-2])/self.dx
            lhs = (self.f[1]-self.f[0])/self.dx
            x_grad = np.concatenate(([lhs], x_grad, [rhs]))
        else:
            raise ValueError(f"Method {method} is not supported for 'center difference'")
        x_grad_grid = Grid1D(self.x_n, self.x_dim, self.n_ghost, x_grad)
        return x_grad_grid

if __name__ == "__main__":
    my_grid = Grid1D(20, (0, 2*np.pi), 2)
    my_grid.calc_f(np.sin)
    my_grid.imshow()
    d_dx = my_grid.center_difference(method="numpy")
    d_dx2 = my_grid.center_difference(method="manual")

    # check if both methods yield the same result
    print(np.max(np.abs(d_dx.f-d_dx2.f)))

    # plot the gradient
    d_dx.imshow()

    # plot both together
    my_grid.imshow(extra_grids=[d_dx])