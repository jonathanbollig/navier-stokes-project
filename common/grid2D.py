import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Grid2D:
    def __init__(self, x_n=10, y_n=10, x_dim=(0, 1), y_dim=(0, 1), n_ghost=2, f=None):
        self.x_n, self.y_n = x_n, y_n
        self.n_ghost = n_ghost
        self.x_dim, self.y_dim = x_dim, y_dim
        self.dx, self.dy = (x_dim[1] - x_dim[0])/x_n, (y_dim[1] - y_dim[0])/y_n 
        self.xs, self.ys = self.create_vector(x_dim, self.dx, n_ghost, x_n), self.create_vector(y_dim, self.dy, n_ghost, y_n)
        self.f = f  # could add sanity check if shape of f matches x_n and y_n.

    @staticmethod
    def create_vector(dim, h, n_ghost, len_):
        return np.linspace(dim[0] + (-n_ghost+1/2)*h, dim[1] + (n_ghost-1/2)*h, len_ + 2*n_ghost)
    
    def calc_f(self, func):
        # maybe "func" should be saved as well? 
        # this uses broadcasting, see https://www.geeksforgeeks.org/numpy/numpy-array-broadcasting/
        self.f = func(self.xs[:, None], self.ys[None, :])

    def imshow(self):
        fig, ax = plt.subplots()
        ax.imshow(self.f, extent=(self.x_dim[0]-self.n_ghost*self.dx, 
                                  self.x_dim[1]+self.n_ghost*self.dx, 
                                  self.y_dim[0]-self.n_ghost*self.dy, 
                                  self.y_dim[1]+self.n_ghost*self.dy))
        rect = Rectangle((self.x_dim[0], self.y_dim[0]), (self.x_dim[1] - self.x_dim[0]), (self.y_dim[1] - self.y_dim[0]), 
                         linewidth=2, edgecolor='black', facecolor='none')
        ax.add_patch(rect)
        plt.show()

    def center_difference(self, method="numpy"):
        """
        returns 2 new instances of a 2d grid class, containing x and y gradient as f
        allowed methods:
        - numpy  # uses numpy gradient --> center differential
        - manual 
        """
        if method == "numpy":
            x_grad, y_grad = np.gradient(self.f, self.dx, self.dy) # explenation: https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
        elif method == "manual":
            x_grad = (self.f[2:, :] - self.f[:-2, :])/2/self.dx
            rhs = (self.f[-1, :]-self.f[-2, :])/self.dx
            lhs = (self.f[1, :]-self.f[0, :])/self.dx
            x_grad = np.concatenate(([lhs], x_grad, [rhs]))

            y_grad = (self.f[:, 2:] - self.f[:, :-2])/2/self.dx
            top = (self.f[:, -1]-self.f[:, -2])/self.dx
            bottom = (self.f[:, 1]-self.f[:, 0])/self.dx
            y_grad = np.concatenate(([top], y_grad.T, [bottom])).T  # a little hacky with the two '.T', but I think it's correct
        else:
            raise ValueError(f"Method {method} is not supported for 'center difference'")
        x_grad_grid = Grid2D(self.x_n, self.y_n, self.x_dim, self.y_dim, self.n_ghost, x_grad)
        y_grad_grid = Grid2D(self.x_n, self.y_n, self.x_dim, self.y_dim, self.n_ghost, y_grad)
        return x_grad_grid, y_grad_grid

if __name__ == "__main__":
    my_grid = Grid2D(10, 20, (0, np.pi), (0, 2*np.pi), 2)
    my_grid.calc_f(lambda x, y: np.sin(x)*y)
    my_grid.imshow()
    d_dx, d_dy = my_grid.center_difference(method="numpy")
    d_dx2, d_dy2 = my_grid.center_difference(method="manual")

    # check if both methods yield the same result
    print(np.max(np.abs(d_dx.f-d_dx2.f)))
    print(np.max(np.abs(d_dy.f-d_dy2.f)))
  
    # plot the gradients
    d_dx.imshow()
    d_dy.imshow()