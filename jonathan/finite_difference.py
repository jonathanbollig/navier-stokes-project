import numpy as np 
import matplotlib.pyplot as plt

class grid:
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
        self.grid = np.meshgrid(*vectors)
        print(self.grid)

    def finite_difference(self, method="center", implementation="numpy"):
        pass


if __name__ == "__main__":
    my_grid = grid((10, 10), ((0, 1), (0, 1)))
    print(my_grid.grid)