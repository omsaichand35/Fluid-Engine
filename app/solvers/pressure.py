import numpy as np
from numba import njit


@njit(fastmath=True)
def pressure_project_jit(u, v, p, iterations):
    size = p.shape[0]
    div = np.zeros_like(p)

    # Compute divergence
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            div[i, j] = -0.5 * (
                u[i+1, j] - u[i-1, j] +
                v[i, j+1] - v[i, j-1]
            )
            p[i, j] = 0.0

    # Jacobi iterations
    for _ in range(iterations):
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                p[i, j] = (
                    div[i, j] +
                    p[i+1, j] + p[i-1, j] +
                    p[i, j+1] + p[i, j-1]
                ) * 0.25

    # Subtract pressure gradient
    for i in range(1, size - 1):
        for j in range(1, size - 1):
            u[i, j] -= 0.5 * (p[i+1, j] - p[i-1, j])
            v[i, j] -= 0.5 * (p[i, j+1] - p[i, j-1])

class PressureSolver:
    def __init__(self, grid, iterations=20):
        self.grid = grid
        self.iterations = iterations

    def project(self):
        pressure_project_jit(
            self.grid.velocity.u,
            self.grid.velocity.v,
            self.grid.pressure.data,
            self.iterations
        )
