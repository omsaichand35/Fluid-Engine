import numpy as np

class VorticityConfinement:
    def __init__(self, grid, strength=2.0):
        self.grid = grid
        self.strength = strength

    def step(self, dt):
        u = self.grid.velocity.u
        v = self.grid.velocity.v

        # Enforce solid boundaries
        u[0, :] = u[-1, :] = 0
        u[:, 0] = u[:, -1] = 0
        v[0, :] = v[-1, :] = 0
        v[:, 0] = v[:, -1] = 0

        dv_dx = (np.roll(v, -1, axis=0) - np.roll(v, 1, axis=0)) * 0.5
        du_dy = (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1)) * 0.5
        curl = dv_dx - du_dy

        curl = np.clip(curl, -5.0, 5.0)

        abs_curl = np.abs(curl)
        d_abs_dx = (np.roll(abs_curl, -1, axis=0) - np.roll(abs_curl, 1, axis=0)) * 0.5
        d_abs_dy = (np.roll(abs_curl, -1, axis=1) - np.roll(abs_curl, 1, axis=1)) * 0.5

        length = np.sqrt(d_abs_dx**2 + d_abs_dy**2) + 1e-6
        nx = d_abs_dx / length
        ny = d_abs_dy / length

        fx = ny * curl * self.strength * dt
        fy = -nx * curl * self.strength * dt

        u[1:-1, 1:-1] += fx[1:-1, 1:-1]
        v[1:-1, 1:-1] += fy[1:-1, 1:-1]
