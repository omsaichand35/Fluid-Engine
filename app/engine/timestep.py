class Timestep:
    def __init__(self, clock, target_fps):
        self.clock = clock
        self.target_fps = target_fps
        self.dt = 0.0

    def update(self):
        self.dt = self.clock.tick(self.target_fps) / 1000.0
        return self.dt
