import numpy as np
from core.grid import Grid
from interaction.obstacles import ObstacleBuilder
from engine.simulation import Simulation

class Engine:
    def __init__(self, grid_size=64):
        self.dt = 0.0
        self.grid = Grid(grid_size)
        self.simulation = Simulation(self.grid)

        self.obstacles = ObstacleBuilder(self.grid)

        # Add a circular obstacle
        self.obstacles.add_circle(
            cx=grid_size // 2,
            cy=grid_size // 2,
            radius=6
        )
        
        # Add some initial noise to help vorticity
        self.grid.velocity.u += np.random.uniform(-0.5, 0.5, (grid_size, grid_size))
        self.grid.velocity.v += np.random.uniform(-0.5, 0.5, (grid_size, grid_size))

    def update(self, dt):
        self.dt = dt

        cx = self.grid.size // 2
        cy = self.grid.size // 2 + 10

        radius = 3
        for i in range(cx - radius, cx + radius + 1):
            for j in range(cy - radius, cy + radius + 1):
                if 0 <= i < self.grid.size and 0 <= j < self.grid.size:
                    dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                    if dist <= radius:
                        strength = np.cos(dist / radius * np.pi / 2)

                        self.grid.density.data[i, j] = min(
                            1.0,
                            self.grid.density.data[i, j] + strength * dt * 5.0
                        )

                        self.grid.velocity.v[i, j] -= 20.0 * dt * strength
                        self.grid.velocity.u[i, j] += np.random.uniform(-2, 2) * dt * strength

        self.grid.density.data *= 0.995

        self.simulation.step(dt)
