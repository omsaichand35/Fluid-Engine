import numpy as np

from core.fields import ScalarField, VectorField

class Grid:
    def __init__(self, size):
        self.size = size

        self.velocity = VectorField(size)
        self.pressure = ScalarField(size)
        self.density = ScalarField(size)

        self.obstacle = np.zeros((size, size), dtype=np.uint8)

    def clear(self):
        self.velocity.u.fill(0)
        self.velocity.v.fill(0)
        self.pressure.data.fill(0)
        self.density.data.fill(0)
        self.obstacle.fill(0)