import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GridND:   # depricated
    """
    Started with an n-dimensional class, and then decided, this is over engineered and a waste of time. 
    Maybe I will continue here in the future :)
    """
    def __init__(self, shape=(10, 10), dimensions=((0, 1), (0, 1)), n_ghost=0):
        self.shape = shape
        self.n_ghost = n_ghost
        self.dimensions = dimensions
        if len(self.shape) != len(self.dimensions):
            raise ValueError(f"Length of shape and dimensions have to be the same, but where {len(self.shape)} and {len(self.dimensions)}")
        self.grid = self.create_grid()

    def create_grid(self):
        vectors = []  # list that contains the vector of every dimension
        for length, dimension in zip(self.shape, self.dimensions):
            x = np.linspace(dimension[0], dimension[1], length)
            vectors.append(x)
        return np.meshgrid(*vectors)

    def finite_difference(self, method="center", implementation="numpy"):
        pass

class Grid2D:
    def __init__(self, x_n=10, y_n=10, x_dim=(0, 1), y_dim=(0, 1), n_ghost=2):
        self.x_n, self.y_n = x_n, y_n
        self.n_ghost = n_ghost
        self.x_dim, self.y_dim = x_dim, y_dim
        self.dx, self.dy = (x_dim[1] - x_dim[0])/x_n, (y_dim[1] - y_dim[0])/y_n 
        self.xs, self.ys = self.create_linspace(x_dim, self.dx, n_ghost, x_n), self.create_linspace(y_dim, self.dy, n_ghost, y_n)
        self.f = None

    @staticmethod
    def create_linspace(x_dim, dx, n_ghost, x_n):
        return np.linspace(x_dim[0] - n_ghost*dx + 1/2*dx, x_dim[1] + n_ghost*dx - 1/2*dx, x_n + 2*n_ghost)
    
    def calc_f(self, func):
        # maybe "func" should be saved as well? 
        # this uses broadcasting, see https://www.geeksforgeeks.org/numpy/numpy-array-broadcasting/
        self.f = func(self.xs[:, None], self.ys[None, :])

    def set_f(self, f):
        self.f = f

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

    def finite_difference(self, method="numpy"):
        """
        returns 2 new instances of a 2d grid class, containing x and y gradient as f
        allowed methods:
        - numpy  # uses numpy gradient --> center differential
        - center  # not implemented
        - left  # not implemented
        - right # not implemented
        """
        if method == "numpy":
            # np.gradient explenation: https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
            x_grad, y_grad = np.gradient(self.f, self.dx, self.dy)
            x_grad_grid = Grid2D(self.x_n, self.y_n, self.x_dim, self.y_dim, self.n_ghost)
            x_grad_grid.set_f(x_grad)
            y_grad_grid = Grid2D(self.x_n, self.y_n, self.x_dim, self.y_dim, self.n_ghost)
            y_grad_grid.set_f(y_grad)
            return x_grad_grid, y_grad_grid
        pass

if __name__ == "__main__":
    my_grid = Grid2D(10, 10, (0, np.pi), (0, np.pi), 2)
    my_grid.calc_f(lambda x, y: np.sin(x*y))
    my_grid.imshow()
    d_dx, d_dy = my_grid.finite_difference()
    d_dx.imshow()
    d_dy.imshow()
    """
    plt.imshow(d_dx)
    plt.show()
    plt.imshow(d_dy)
    plt.show()
    print(my_grid.xs)
    print(my_grid.ys)
    print(my_grid.f)
    """