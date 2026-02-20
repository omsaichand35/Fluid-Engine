import pygame
import numpy as np


class PygameRenderer:
    def __init__(self, screen, grid):
        self.screen = screen
        self.grid = grid

        self.width, self.height = screen.get_size()

        # Alpha smoke surface
        self.smoke_surface = pygame.Surface(
            (self.width, self.height),
            pygame.SRCALPHA
        )

    def draw_density(self):
        density = self.grid.density.data

        # Clamp + map to [0,255]
        d = np.clip(density * 255, 0, 255).astype(np.uint8)

        # ---- COLOR MAP (soft smoke blue-white) ----
        r = d
        g = d
        b = np.minimum(255, d + 40)

        rgb = np.stack((r, g, b), axis=-1)

        # ---- VERY IMPORTANT: transpose (row,col) → (x,y) ----
        rgb = np.transpose(rgb, (1, 0, 2))

        # Create surface
        surf = pygame.surfarray.make_surface(rgb)

        # Scale to screen
        surf = pygame.transform.smoothscale(surf, (self.width, self.height))

        # Apply alpha (soft smoke)
        surf.set_alpha(140)

        # Draw
        self.smoke_surface.blit(surf, (0, 0))

    def draw_velocity(self, step=5, scale=0.05):
        u = self.grid.velocity.u
        v = self.grid.velocity.v

        cell_w = self.width / self.grid.size
        cell_h = self.height / self.grid.size

        for i in range(1, self.grid.size, step):
            for j in range(1, self.grid.size, step):
                x = int(i * cell_w)
                y = int(j * cell_h)

                vx = u[i, j]
                vy = v[i, j]

                end_x = int(x + vx * scale)
                end_y = int(y + vy * scale)

                pygame.draw.line(
                    self.screen,
                    (0, 255, 0),
                    (x, y),
                    (end_x, end_y),
                    1
                )

    def draw_obstacles(self):
        obs = self.grid.obstacle
        cell_w = self.width / self.grid.size
        cell_h = self.height / self.grid.size

        for i in range(self.grid.size):
            for j in range(self.grid.size):
                if obs[i, j] == 1:
                    rect = pygame.Rect(
                        int(i * cell_w),
                        int(j * cell_h),
                        int(cell_w),
                        int(cell_h)
                    )
                    pygame.draw.rect(self.screen, (80, 80, 80), rect)

    def render(self):
        self.screen.fill((0, 0, 0))
        self.smoke_surface.fill((0, 0, 0, 0))

        self.draw_density()
        self.screen.blit(self.smoke_surface, (0, 0))

        self.draw_obstacles()

        self.draw_velocity()   # optional toggle later
        pygame.display.flip()

