class MouseInteractor:
    def __init__(self, grid, force_strength=50.0, density_amount=1.0):
        self.grid = grid
        self.force_strength = force_strength
        self.density_amount = density_amount
        self.prev_pos = None

    def apply(self, mouse_pos, mouse_down, cell_size):
        if not mouse_down:
            self.prev_pos = None
            return

        x, y = mouse_pos
        i = int(x / cell_size)
        j = int(y / cell_size)

        if i <= 1 or j <= 1 or i >= self.grid.size - 2 or j >= self.grid.size - 2:
            return

        # Inject density
        self.grid.density.data[i, j] = self.density_amount

        # Apply force from mouse movement
        if self.prev_pos is not None:
            px, py = self.prev_pos
            fx = (x - px) * self.force_strength
            fy = (y - py) * self.force_strength

            self.grid.velocity.u[i, j] += fx
            self.grid.velocity.v[i, j] += fy

        self.prev_pos = (x, y)

class MouseObstacleDrawer:
    def __init__(self, grid, obstacle_builder):
        self.grid = grid
        self.obstacles = obstacle_builder

    def apply(self, mouse_pos, mouse_down, cell_size):
        if not mouse_down:
            return

        x, y = mouse_pos
        i = int(x / cell_size)
        j = int(y / cell_size)

        if 1 <= i < self.grid.size - 1 and 1 <= j < self.grid.size - 1:
            self.obstacles.add_cell(i, j, radius=1)
