import numpy as np

class ScalarField:
    def __init__(self, size):
        self.size = size
        self.data = np.zeros((size, size), dtype=np.float32)

class VectorField:
    def __init__(self, size):
        self.size = size
        self.u = np.zeros((size, size), dtype=np.float32)
        self.v = np.zeros((size, size), dtype=np.float32)