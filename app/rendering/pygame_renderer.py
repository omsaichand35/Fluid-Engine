import pygame
import numpy as np


class PygameRenderer:
    def __init__(self, screen, grid, engine=None):
        self.screen = screen
        self.grid = grid
        self.engine = engine
        self.time = 0.0

        self.width, self.height = screen.get_size()

        # Alpha smoke surface
        self.smoke_surface = pygame.Surface(
            (self.width, self.height),
            pygame.SRCALPHA
        )

    def _draw_fire_glow(self, heat_f, density_f):
        flicker = 0.88 + 0.12 * np.sin(self.time * 26.0)
        
        # Scale glow intensity with fire_power.
        fire_power_scale = 1.0
        if self.engine and hasattr(self.engine, 'fire_power'):
            fire_power_scale = self.engine.fire_power
        
        glow = np.clip((heat_f ** 1.7) * (0.40 + 0.75 * density_f) * flicker * fire_power_scale, 0.0, 1.0)

        glow_r = (255.0 * glow).astype(np.uint8)
        glow_g = (165.0 * glow).astype(np.uint8)
        glow_b = (70.0 * glow).astype(np.uint8)

        glow_rgb = np.stack((glow_r, glow_g, glow_b), axis=-1)

        glow_surf = pygame.surfarray.make_surface(glow_rgb)
        glow_surf = pygame.transform.smoothscale(glow_surf, (self.width, self.height))

        # Cheap blur pass via downsample + upsample for bloom-like emission.
        small_w = max(1, self.width // 5)
        small_h = max(1, self.height // 5)
        glow_surf = pygame.transform.smoothscale(glow_surf, (small_w, small_h))
        glow_surf = pygame.transform.smoothscale(glow_surf, (self.width, self.height))
        
        # Increase bloom alpha with fire_power.
        bloom_alpha = int(140 * fire_power_scale * 0.6)  # Scale but cap to stay visible
        glow_surf.set_alpha(bloom_alpha)

        self.screen.blit(glow_surf, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def draw_density(self):
        density = self.grid.density.data
        mode = getattr(self.grid, "element_mode", "fire")
        temperature = getattr(self.grid, "temperature", None)
        temp_data = temperature.data if hasattr(temperature, "data") else temperature
        soot = getattr(self.grid, "soot", None)
        soot_data = soot.data if hasattr(soot, "data") else None

        density_f = np.clip(density, 0.0, 1.0)
        d = (density_f * 255).astype(np.uint8)

        if mode == "fire" and temp_data is not None:
            temp_max = 1.85
            heat_f = np.clip(temp_data / temp_max, 0.0, 1.0)
            overburn = np.clip((temp_data - 1.0) / max(1e-6, temp_max - 1.0), 0.0, 1.0)
            soot_f = np.clip(soot_data, 0.0, 1.0) if soot_data is not None else np.zeros_like(heat_f)

            # Temporal-spatial flicker keeps flames alive and asymmetric.
            yy, xx = np.indices(heat_f.shape, dtype=np.float32)
            flicker_field = 0.90 + 0.10 * np.sin(0.31 * xx + 0.53 * yy + self.time * 19.0)
            flicker_field += 0.05 * np.sin(0.91 * xx - 0.37 * yy + self.time * 39.0)
            heat_f = np.clip(heat_f * flicker_field, 0.0, 1.0)

            # Smooth palette: black -> red -> orange -> yellow -> white.
            c0 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
            c1 = np.array([0.86, 0.04, 0.00], dtype=np.float32)
            c2 = np.array([1.00, 0.35, 0.00], dtype=np.float32)
            c3 = np.array([1.00, 0.85, 0.08], dtype=np.float32)
            c4 = np.array([1.00, 1.00, 1.00], dtype=np.float32)

            t = np.clip(0.78 * heat_f + 0.22 * density_f, 0.0, 1.0)

            # Aggressive contrast: darker darks, sharper bright cores.
            t = np.clip((t - 0.06) * 1.28, 0.0, 1.0)
            t = np.clip(t ** 0.78, 0.0, 1.0)

            rgb_f = np.zeros((*t.shape, 3), dtype=np.float32)

            m0 = t < 0.30
            m1 = (t >= 0.30) & (t < 0.55)
            m2 = (t >= 0.55) & (t < 0.82)
            m3 = t >= 0.82

            if np.any(m0):
                a = (t[m0] / 0.30).reshape(-1, 1)
                rgb_f[m0] = c0 + a * (c1 - c0)
            if np.any(m1):
                a = ((t[m1] - 0.30) / 0.25).reshape(-1, 1)
                rgb_f[m1] = c1 + a * (c2 - c1)
            if np.any(m2):
                a = ((t[m2] - 0.55) / 0.27).reshape(-1, 1)
                rgb_f[m2] = c2 + a * (c3 - c2)
            if np.any(m3):
                a = ((t[m3] - 0.82) / 0.18).reshape(-1, 1)
                rgb_f[m3] = c3 + a * (c4 - c3)

            # Dark smoke overlays in soot-rich pockets.
            smoke_dim = np.clip(0.78 * soot_f, 0.0, 0.75)
            rgb_f[:, :, 0] *= (1.0 - 0.58 * smoke_dim)
            rgb_f[:, :, 1] *= (1.0 - 0.85 * smoke_dim)
            rgb_f[:, :, 2] *= (1.0 - 0.92 * smoke_dim)

            # Overburn hotspot whitening for explosive cores.
            # Scale whitening intensity with fire_power.
            fire_power_scale = 1.0
            if self.engine and hasattr(self.engine, 'fire_power'):
                fire_power_scale = self.engine.fire_power
            
            overburn_r = 0.55 * fire_power_scale
            overburn_g = 0.48 * fire_power_scale
            overburn_b = 0.38 * fire_power_scale
            
            rgb_f[:, :, 0] = np.clip(rgb_f[:, :, 0] + overburn_r * overburn, 0.0, 1.0)
            rgb_f[:, :, 1] = np.clip(rgb_f[:, :, 1] + overburn_g * overburn, 0.0, 1.0)
            rgb_f[:, :, 2] = np.clip(rgb_f[:, :, 2] + overburn_b * overburn, 0.0, 1.0)

            pulse = 0.94 + 0.06 * np.sin(self.time * 33.0)
            # Slightly boost pulse brightness with fire_power.
            pulse = np.clip(pulse * (0.97 + 0.03 * fire_power_scale), 0.85, 1.0)
            rgb_f *= pulse
            rgb_f = np.clip(rgb_f, 0.0, 1.0)

            r = (np.clip(rgb_f[:, :, 0], 0.0, 1.0) * 255).astype(np.uint8)
            g = (np.clip(rgb_f[:, :, 1], 0.0, 1.0) * 255).astype(np.uint8)
            b = (np.clip(rgb_f[:, :, 2], 0.0, 1.0) * 255).astype(np.uint8)

            # Base alpha from flame with extra smoke opacity.
            alpha_field = np.clip(0.24 + 0.92 * density_f + 0.52 * soot_f + 0.25 * overburn, 0.0, 1.0)
            alpha = int(np.clip(np.mean(alpha_field) * 255, 120, 240))
        elif mode == "smoke":
            soot_f = np.clip(soot_data, 0.0, 1.0) if soot_data is not None else density_f
            smoke_luma = np.clip(0.18 + 0.35 * density_f - 0.30 * soot_f, 0.0, 1.0)
            r = (smoke_luma * 255).astype(np.uint8)
            g = (smoke_luma * 255).astype(np.uint8)
            b = (smoke_luma * 255).astype(np.uint8)
            alpha = int(np.clip(np.mean(0.25 + 0.80 * soot_f) * 255, 100, 210))
        elif mode == "water":
            # Render water height field
            water_height = getattr(self.grid, "water_height", None)
            height_data = water_height.data if hasattr(water_height, "data") else np.zeros_like(density)
            
            # Normalize height (0-1 for rendering, but height can be >1)
            height_norm = np.clip(height_data / 2.0, 0.0, 1.0)  # 2.0 is reasonable max display
            has_water = height_norm > 0.01
            
            # Water power scaling
            water_power_scale = 1.0
            if self.engine and hasattr(self.engine, 'water_power'):
                water_power_scale = self.engine.water_power
            
            # Base water color: deep blue to cyan/white
            # Deeper water = darker, surface = brighter
            depth_factor = 1.0 - height_norm
            
            # Surface wave shading (fake caustics/ripples)
            xx, yy = np.meshgrid(np.arange(height_norm.shape[1], dtype=np.float32),
                                  np.arange(height_norm.shape[0], dtype=np.float32))
            wave = 0.5 + 0.3 * np.sin(xx * 0.1 + self.time * 3.0) * np.cos(yy * 0.12 - self.time * 2.5)
            wave *= height_norm  # Only visible where water exists
            
            # Surface brightness: brighter at surface (shallow)
            surface_brightness = np.clip(0.4 + 0.45 * height_norm + 0.2 * wave, 0.0, 1.0)
            
            # Water color gradient
            r_f = 0.05 + 0.15 * height_norm + 0.15 * surface_brightness
            g_f = 0.25 + 0.3 * height_norm + 0.2 * surface_brightness
            b_f = 0.55 + 0.25 * height_norm + 0.15 * surface_brightness
            
            # God mode: brighter, more saturated
            intensity_mult = 0.85 + 0.15 * water_power_scale
            r_f *= intensity_mult
            g_f *= intensity_mult
            b_f *= intensity_mult
            
            r = (np.clip(r_f, 0.0, 1.0) * 255).astype(np.uint8)
            g = (np.clip(g_f, 0.0, 1.0) * 255).astype(np.uint8)
            b = (np.clip(b_f, 0.0, 1.0) * 255).astype(np.uint8)
            
            # Alpha based on water depth
            alpha_f = np.clip(0.35 + 0.65 * height_norm, 0.0, 1.0)
            alpha = int(np.mean(alpha_f) * 255)
        else:
            # ---- COLOR MAP (soft smoke blue-white) ----
            r = d
            g = d
            b = np.minimum(255, d + 40)
            alpha = 140

        rgb = np.stack((r, g, b), axis=-1)

        # Create surface
        surf = pygame.surfarray.make_surface(rgb)

        # Scale to screen
        surf = pygame.transform.smoothscale(surf, (self.width, self.height))

        # Apply alpha (soft smoke)
        surf.set_alpha(alpha)

        # Draw
        self.smoke_surface.blit(surf, (0, 0))

        if mode == "fire" and temp_data is not None:
            self._draw_fire_glow(np.clip(temp_data / 1.85, 0.0, 1.0), density_f)

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
        self.time += 1.0 / 60.0
        self.screen.fill((0, 0, 0))
        self.smoke_surface.fill((0, 0, 0, 0))
        mode = getattr(self.grid, "element_mode", "fire")

        self.draw_density()
        self.screen.blit(self.smoke_surface, (0, 0))

        self.draw_obstacles()

        if mode != "fire":
            self.draw_velocity()   # optional toggle later
        pygame.display.flip()

