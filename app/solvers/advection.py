import numpy as np
from numba import njit


@njit(fastmath=True)
def advect_bilinear_jit(field, u, v, dt):
    size = field.shape[0]
    new_field = np.zeros_like(field)

    for i in range(1, size - 1):
        for j in range(1, size - 1):

            x = i - u[i, j] * dt
            y = j - v[i, j] * dt

            if x < 0.5:
                x = 0.5
            elif x > size - 1.5:
                x = size - 1.5

            if y < 0.5:
                y = 0.5
            elif y > size - 1.5:
                y = size - 1.5

            i0 = int(x)
            j0 = int(y)
            i1 = i0 + 1
            j1 = j0 + 1

            s1 = x - i0
            s0 = 1.0 - s1
            t1 = y - j0
            t0 = 1.0 - t1

            new_field[i, j] = (
                s0 * (t0 * field[i0, j0] + t1 * field[i0, j1]) +
                s1 * (t0 * field[i1, j0] + t1 * field[i1, j1])
            )

    return new_field


class AdvectionSolver:
    def __init__(self, grid):
        self.grid = grid

    def step(self, dt):
        g = self.grid
        u = g.velocity.u
        v = g.velocity.v

        g.density.data = advect_bilinear_jit(
            g.density.data, u, v, dt
        )

        g.velocity.u = advect_bilinear_jit(
            u, u, v, dt
        )

        g.velocity.v = advect_bilinear_jit(
            v, u, v, dt
        )
