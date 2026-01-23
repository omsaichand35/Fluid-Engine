import pygame
from core.constants import WINDOW_WIDTH, WINDOW_HEIGHT, TARGET_FPS
from engine.engine import Engine
from engine.timestep import Timestep
from rendering.pygame_renderer import PygameRenderer
from interaction.mouse import MouseInteractor, MouseObstacleDrawer


class WindowApp:
    def __init__(self):
        pygame.init()

        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Fluid Engine")

        self.clock = pygame.time.Clock()
        self.timestep = Timestep(self.clock, TARGET_FPS)

        self.engine = Engine()
        self.renderer = PygameRenderer(self.screen, self.engine.grid)

        self.cell_size = self.screen.get_width() // self.engine.grid.size
        self.mouse = MouseInteractor(self.engine.grid)

        self.cell_size = self.screen.get_width() // self.engine.grid.size

        self.obstacle_drawer = MouseObstacleDrawer(
            self.engine.grid,
            self.engine.obstacles
        )

        self.running = True

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.render()
        pygame.quit()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

    def update(self):
        dt = self.timestep.update()

        mouse_pos = pygame.mouse.get_pos()
        left, _, right = pygame.mouse.get_pressed()

        # Smoke + force (existing)
        self.mouse.apply(mouse_pos, left, self.cell_size)

        self.obstacle_drawer.apply(mouse_pos, right, self.cell_size)

        self.engine.update(dt)

    def render(self):
        self.renderer.render()