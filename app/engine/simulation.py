from solvers.advection import AdvectionSolver
from solvers.boundary import BoundarySolver
from solvers.pressure import PressureSolver
from solvers.vorticity import VorticityConfinement


class Simulation:
    def __init__(self, grid):
        self.advection = AdvectionSolver(grid)
        self.pressure = PressureSolver(grid)
        self.vorticity = VorticityConfinement(grid)
        self.boundary = BoundarySolver(grid)

    def step(self, dt):
        self.advection.step(dt)
        self.vorticity.step(dt)
        self.pressure.project()
        self.boundary.apply()