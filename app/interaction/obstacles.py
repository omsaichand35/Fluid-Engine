class ObstacleBuilder:
    def __init__(self, grid):
        self.grid = grid

    def add_circle(self, cx, cy, radius):
        for i in range(cx - radius, cx + radius + 1):
            for j in range(cy - radius, cy + radius + 1):
                if 0 <= i < self.grid.size and 0 <= j < self.grid.size:
                    if (i - cx)**2 + (j - cy)**2 <= radius**2:
                        self._make_solid(i, j)

    def add_cell(self, i, j, radius=1):
        for x in range(i - radius, i + radius + 1):
            for y in range(j - radius, j + radius + 1):
                if 0 <= x < self.grid.size and 0 <= y < self.grid.size:
                    self._make_solid(x, y)

    def _make_solid(self, i, j):
        self.grid.obstacle[i, j] = 1
        self.grid.velocity.u[i, j] = 0
        self.grid.velocity.v[i, j] = 0
        self.grid.density.data[i, j] = 0
        if hasattr(self.grid, "temperature"):
            self.grid.temperature.data[i, j] = 0
        if hasattr(self.grid, "fuel"):
            self.grid.fuel.data[i, j] = 0
        if hasattr(self.grid, "oxygen"):
            self.grid.oxygen.data[i, j] = 1.0
        if hasattr(self.grid, "soot"):
            self.grid.soot.data[i, j] = 0
