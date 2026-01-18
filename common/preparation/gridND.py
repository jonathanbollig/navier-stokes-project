import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class GridND:
    def __init__(self, shape=(10, 10), dimensions=((0, 1), (0, 1)), n_ghost=0, f=None):
        if len(shape) != len(dimensions):
            raise ValueError(f"Length of shape and dimensions have to be the same, but where {len(shape)} and {len(dimensions)}")
        if f is not None and (np.array(f.shape) != np.array(shape)+2*n_ghost).all():
            print(f)
            raise ValueError(f"The shape of f does not agree with shape: {np.array(f.shape)} vs {np.array(shape)+2*n_ghost}") 
        self.shape = shape  # length of every dimension without ghost! 
        self.n_ghost = n_ghost
        self.dimensions = dimensions
        self.hs = np.array([(dim[1] - dim[0])/len_ for len_, dim in zip(shape, dimensions)])
        self.vectors = [self.create_vector(dim, h, n_ghost, len_) for dim, h, len_ in zip(dimensions, self.hs, shape)]
        self.f = f # the actual values of our grid in the sense of f = f(x_i)
    
    @staticmethod
    def create_vector(dim, h, n_ghost, len_):
        return np.linspace(dim[0] + (-n_ghost+1/2)*h, dim[1] + (n_ghost-1/2)*h, len_ + 2*n_ghost)

    def calc_f(self, func):
        # maybe "func" should be saved as well? 
        # this uses broadcasting, see https://www.geeksforgeeks.org/numpy/numpy-array-broadcasting/
        # self.f = func(*(self.arrs[i][(slice(None) if dim == i else None) for dim in range(n)] for i in range(n)))
        broadcasted_views = np.ix_(*self.vectors)
        self.f = func(*broadcasted_views)

    def imshow(self):
        if len(self.f.shape) != 2:
            raise ValueError(f"Can only plot functions with 2 dimensions, but f has {len(self.f.shape)} dimensions.")
        fig, ax = plt.subplots()
        ax.imshow(self.f)
        plt.show()

    def center_difference(self):
        # returns n new instances of a nd grid class, containing gradients as f
        gradients = np.gradient(self.f, *self.hs) # explenation: https://numpy.org/doc/stable/reference/generated/numpy.gradient.html
        if isinstance(gradients, np.ndarray):
            # in the 1D-case, np.gradients returns an array, and not a tuple of arrays.
            # in that case, we want gradients to be a tuple with that one array for consistancy
            gradients = (gradients, )
        return [GridND(self.shape, self.dimensions, self.n_ghost, gradient) for gradient in gradients]

if __name__ == "__main__":
    # ND Version
    my_grid = GridND((10, 20), ((0, np.pi), (0, 2*np.pi)), 2)
    my_grid.calc_f(lambda x, y: np.sin(x)*y)
    my_grid.imshow()

    gradients = my_grid.center_difference()
    for gradient in gradients:
        gradient.imshow()
