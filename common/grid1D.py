import numpy as np 
import matplotlib.pyplot as plt

class Grid1D:
    def __init__(self, x_n=10, x_dim=(0, 1), n_ghost=2, f=None):
        self.x_n = x_n
        self.n_ghost = n_ghost
        self.x_dim = x_dim
        self.dx = (x_dim[1] - x_dim[0])/x_n 
        self.xs = self.create_vector(x_dim, self.dx, n_ghost, x_n)
        self.f = f  # could add sanity check if shape of f matches x_n and y_n.

    @staticmethod
    def create_vector(dim, h, n_ghost, len_):
        return np.linspace(dim[0] + (-n_ghost+1/2)*h, dim[1] + (n_ghost-1/2)*h, len_ + 2*n_ghost)
    
    def calc_f(self, func):
        # maybe "func" should be saved as well? 
        self.f = func(self.xs)

    def imshow(self):
        fig, ax = plt.subplots()
        ax.plot(self.xs, self.f)
        plt.show()

    def center_difference(self, method="numpy"):
        """
        returns 2 new instances of a 2d grid class, containing x and y gradient as f
        allowed methods:
        - numpy  # uses numpy gradient --> center differential
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
    my_grid.calc_f(lambda x: np.sin(x))
    my_grid.imshow()
    d_dx = my_grid.center_difference(method="numpy")
    d_dx2 = my_grid.center_difference(method="manual")

    # check if both methods yield the same result
    print(np.max(np.abs(d_dx.f-d_dx2.f)))
  
    # plot the gradients
    d_dx.imshow()