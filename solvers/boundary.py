class BoundarySolver:
    def __init__(self, grid):
        self.grid = grid

    def apply(self):
        obs = self.grid.obstacle
        u = self.grid.velocity.u
        v = self.grid.velocity.v

        # Zero velocity inside obstacles
        u[obs == 1] = 0
        v[obs == 1] = 0

        # Optional: no-slip walls (stronger realism)
        for i in range(1, self.grid.size - 1):
            for j in range(1, self.grid.size - 1):
                if obs[i, j] == 1:
                    u[i+1, j] = 0
                    u[i-1, j] = 0
                    v[i, j+1] = 0
                    v[i, j-1] = 0
