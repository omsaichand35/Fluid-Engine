import numpy as np

from core.fields import ScalarField, VectorField

class Grid:
    def __init__(self, size):
        self.size = size

        self.velocity = VectorField(size)
        self.pressure = ScalarField(size)
        self.density = ScalarField(size)
        self.temperature = ScalarField(size)
        self.fuel = ScalarField(size)
        self.oxygen = ScalarField(size)
        self.soot = ScalarField(size)
        
        # Water simulation: height-field based shallow water model
        self.water_height = ScalarField(size)  # Water depth per cell
        self.water_height_prev = np.zeros((size, size), dtype=np.float32)  # For integration

        self.oxygen.data.fill(1.0)

        self.element_mode = "fire"

        self.obstacle = np.zeros((size, size), dtype=np.uint8)

    def clear(self):
        self.velocity.u.fill(0)
        self.velocity.v.fill(0)
        self.pressure.data.fill(0)
        self.density.data.fill(0)
        self.temperature.data.fill(0)
        self.fuel.data.fill(0)
        self.oxygen.data.fill(1.0)
        self.soot.data.fill(0)
        self.water_height.data.fill(0)
        self.water_height_prev.fill(0)
        self.obstacle.fill(0)