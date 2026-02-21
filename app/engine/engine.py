import numpy as np
from core.grid import Grid
from interaction.obstacles import ObstacleBuilder
from engine.simulation import Simulation

class Engine:
    def __init__(self, grid_size=96):
        self.dt = 0.0
        self.sim_time = 0.0
        self.grid = Grid(grid_size)
        self.simulation = Simulation(self.grid)

        # Runtime mode: fire, wind, or smoke (extras).
        self.element_mode = "fire"
        self.wind_dir = np.array([1.0, -0.15], dtype=np.float32)

        self.fire_emission = 6.2
        self.fire_heat = 4.3
        self.fire_buoyancy = 36.0
        self.fire_cooling = 0.985
        self.fire_core_radius = 2
        self.fire_outer_radius = 5
        self.fire_turbulence = 9.0
        self.fire_updraft_bias = 26.0
        self.fire_diffusion = 0.18
        self.fire_noise_decay = 0.025
        self.fire_transport_cooling = 0.012
        self.fire_transport_blend = 0.48
        self.fire_fragmentation = 0.24

        self.combustion_rate = 5.6
        self.activation_energy = 7.2
        self.heat_release = 2.4
        self.oxygen_consumption = 0.72
        self.ignition_temperature = 0.18
        self.soot_yield = 0.55
        self.soot_oxidation = 1.3

        # Source sustain floors keep a continuous visible burner.
        self.source_fuel_floor = 0.40
        self.source_temp_floor = 0.34
        self.source_oxygen_floor = 0.58

        self.radiation_strength = 0.42
        self.radiation_reabsorb = 0.22

        self.wind_force = 45.0
        self.wind_noise = 6.0

        self.gravity = 9.8
        self.ambient_temperature = 0.15
        self.buoyancy_strength = 44.0
        self.buoyancy_exponent = 1.35
        self.cold_sink_strength = 18.0
        self.temperature_relaxation = 0.65

        self.turbulence_strength = 1.8
        self.turbulence_scale_x = 0.21
        self.turbulence_scale_y = 0.35
        self.turbulence_speed = 2.2
        self.turbulence_high_scale_x = 0.63
        self.turbulence_high_scale_y = 0.88
        self.turbulence_warp_strength = 8.5

        self.vertical_stretch = 0.16
        self.plume_centering_strength = 10.5
        self.drift_correction_strength = 0.85

        # Inferno controls (runtime tunable).
        self.chaos_level = 1.35
        self.explosion_strength = 1.15
        self.instability_frequency = 1.7

        self.combustion_nonlinearity = 1.8
        self.burst_strength = 0.95
        self.burst_frequency = 6.5
        self.expansion_strength = 16.0
        self.temp_velocity_coupling = 22.0
        self.max_temperature = 1.85
        self.source_wander = 3.0

        # Fire power-up system (1.0 = base, 4.0 = max).
        self.fire_power = 1.0

        # Water simulation parameters
        self.water_power = 1.0  # Water god mode (1.0 = normal, 4.0 = god mode)
        self.water_emission = 2.2
        self.water_gravity = 85.0
        self.water_pressure_strength = 12.0
        self.water_viscosity = 0.92  # High viscosity = slower diffusion
        self.water_diffusion = 0.08
        self.water_core_radius = 3
        self.water_outer_radius = 6
        
        # Water god mode controls
        self.water_wave_amplitude = 1.0
        self.water_turbulence_strength = 0.5

        self.cfl_target = 0.95
        self.max_substeps = 4

        self.obstacles = ObstacleBuilder(self.grid)
        self._x, self._y = np.indices((grid_size, grid_size), dtype=np.float32)

        # Add a circular obstacle
        self.obstacles.add_circle(
            cx=grid_size // 2,
            cy=grid_size // 2,
            radius=6
        )
        
        # Add some initial noise to help vorticity
        self.grid.velocity.u += np.random.uniform(-0.5, 0.5, (grid_size, grid_size))
        self.grid.velocity.v += np.random.uniform(-0.5, 0.5, (grid_size, grid_size))

        self.grid.element_mode = self.element_mode

    def _radial_strength(self, cx, cy, radius, core_radius):
        dx = self._x - float(cx)
        dy = self._y - float(cy)
        dist = np.sqrt(dx * dx + dy * dy)

        outer = np.zeros_like(dist, dtype=np.float32)
        core = np.zeros_like(dist, dtype=np.float32)

        outer_mask = dist <= radius
        if np.any(outer_mask):
            outer[outer_mask] = np.cos((dist[outer_mask] / radius) * np.pi * 0.5)

        core_mask = dist <= core_radius
        if np.any(core_mask):
            core[core_mask] = np.cos((dist[core_mask] / core_radius) * np.pi * 0.5)

        return outer, core, outer_mask

    def _coherent_turbulence(self, amplitude=1.0):
        phase = self.sim_time * self.turbulence_speed
        n1 = np.sin(self._x * self.turbulence_scale_x + phase)
        n2 = np.cos(self._y * self.turbulence_scale_y - 1.3 * phase)
        n3 = np.sin((self._x + self._y) * 0.17 + 0.7 * phase)
        h1 = np.sin(self._x * self.turbulence_high_scale_x - 1.7 * phase)
        h2 = np.cos(self._y * self.turbulence_high_scale_y + 1.2 * phase)
        noise = 0.42 * n1 + 0.26 * n2 + 0.12 * n3 + 0.12 * h1 + 0.08 * h2
        noise -= np.mean(noise)
        return amplitude * noise

    def _warp_velocity(self, dt):
        vel_u = self.grid.velocity.u
        vel_v = self.grid.velocity.v
        turb = self._coherent_turbulence(amplitude=1.0 + 0.55 * self.chaos_level)

        grad_x = 0.5 * (np.roll(turb, -1, axis=0) - np.roll(turb, 1, axis=0))
        grad_y = 0.5 * (np.roll(turb, -1, axis=1) - np.roll(turb, 1, axis=1))

        warp = self.turbulence_warp_strength * self.chaos_level * dt
        vel_u += warp * grad_y
        vel_v -= warp * grad_x

    def _inject_instability(self, dt):
        trigger_prob = self.instability_frequency * self.chaos_level * dt
        if np.random.random() > trigger_prob:
            return

        size = self.grid.size
        cx = np.random.randint(size // 4, 3 * size // 4)
        cy = np.random.randint(size // 3, size - 8)
        radius = np.random.randint(2, 6)

        outer, core, mask = self._radial_strength(cx, cy, radius, max(1, radius // 2))
        pocket = (0.45 * outer + 1.25 * core) * mask

        boost = self.explosion_strength * (0.45 + 0.75 * np.random.random())
        self.grid.temperature.data += boost * 0.35 * pocket
        self.grid.fuel.data += boost * 0.22 * pocket
        self.grid.velocity.v -= boost * 14.0 * dt * pocket
        self.grid.velocity.u += np.random.uniform(-1.0, 1.0, size=self.grid.velocity.u.shape) * boost * 6.0 * dt * pocket

        np.clip(self.grid.temperature.data, 0.0, self.max_temperature, out=self.grid.temperature.data)
        np.clip(self.grid.fuel.data, 0.0, 1.0, out=self.grid.fuel.data)

    def _enforce_vertical_fire_orientation(self, dt):
        temp = self.grid.temperature.data
        vel_u = self.grid.velocity.u

        hot = np.clip((temp - self.ignition_temperature) / max(1e-6, 1.0 - self.ignition_temperature), 0.0, 1.0)
        hot_inner = hot[2:-2, 2:-2]
        if hot_inner.size == 0:
            return

        mass = float(np.sum(hot_inner)) + 1e-6
        drift = float(np.sum(vel_u[2:-2, 2:-2] * hot_inner) / mass)

        # Cancel any net left/right drift in hot plume.
        vel_u[1:-1, 1:-1] -= self.drift_correction_strength * drift * hot[1:-1, 1:-1]

        # Restore plume to centerline so flame points upward.
        center = (self.grid.size - 1) * 0.5
        norm_offset = (self._x - center) / max(1.0, center)
        vel_u[1:-1, 1:-1] -= self.plume_centering_strength * norm_offset[1:-1, 1:-1] * hot[1:-1, 1:-1] * dt

    def set_element_mode(self, mode):
        if mode not in ("fire", "wind", "smoke", "water"):
            return
        self.element_mode = mode
        self.grid.element_mode = mode

    def spawn_fireball(self):
        self.set_element_mode("fire")

    def spawn_wind(self):
        self.set_element_mode("wind")

    def spawn_smoke(self):
        self.set_element_mode("smoke")

    def adjust_fire_power(self, delta):
        """Adjust fire power by delta (e.g., +0.2 or -0.2). Clamps to [1.0, 4.0]."""
        self.fire_power = np.clip(self.fire_power + delta, 1.0, 4.0)

    def adjust_water_power(self, delta):
        """Adjust water power (god mode) by delta. Clamps to [1.0, 4.0]."""
        self.water_power = np.clip(self.water_power + delta, 1.0, 4.0)

    def _apply_water_source(self, dt):
        """Inject water at bottom center, raising water height (depth)."""
        size = self.grid.size
        cx = size // 2
        cy = size - 8
        
        height = self.grid.water_height.data
        
        # Scale source inject with water_power
        core_radius = int(self.water_core_radius * (1.0 + 0.6 * (self.water_power - 1.0)))
        outer_radius = int(self.water_outer_radius * (1.0 + 0.8 * (self.water_power - 1.0)))
        
        outer_strength, core_strength, mask = self._radial_strength(cx, cy, outer_radius, core_radius)
        injection = (0.35 * outer_strength + 1.15 * core_strength) * mask
        
        # Scale emission with water_power (rate of height increase)
        emission_scale = 0.85 + 0.5 * (self.water_power - 1.0) / 3.0
        injection_rate = self.water_emission * emission_scale * dt
        height += injection_rate * injection
        
        # Add initial upward velocity at source for momentum
        vel_v = self.grid.velocity.v
        pressure_impulse = self.water_pressure_strength * (0.3 * outer_strength + 1.1 * core_strength) * mask
        vel_v -= pressure_impulse * dt

    def _apply_water_gravity(self, dt):
        """Apply gravity to water velocity (not to height directly)."""
        vel_v = self.grid.velocity.v
        
        # Gravity pulls water downward
        gravity_scale = self.water_gravity * (1.0 + 0.3 * (self.water_power - 1.0))
        vel_v += gravity_scale * dt
        
        # Clamp vertical velocity
        np.clip(vel_v, -130.0, 130.0, out=vel_v)

    def _compute_water_height_gradient(self):
        """Compute gradient of water height for pressure."""
        height = self.grid.water_height.data
        
        # Pressure gradient from height differences (shallow water assumption)
        grad_x = 0.5 * (np.roll(height, -1, axis=0) - np.roll(height, 1, axis=0))
        grad_y = 0.5 * (np.roll(height, -1, axis=1) - np.roll(height, 1, axis=1))
        
        return grad_x, grad_y

    def _apply_water_height_pressure_forces(self, dt):
        """Apply pressure forces from water height to velocity."""
        vel_u = self.grid.velocity.u
        vel_v = self.grid.velocity.v
        grad_x, grad_y = self._compute_water_height_gradient()
        
        # Stronger pressure for god mode
        pressure_scale = self.water_pressure_strength * (0.8 + 0.2 * self.water_power)
        vel_u -= pressure_scale * grad_x * dt
        vel_v -= pressure_scale * grad_y * dt

    def _apply_water_surface_smoothing(self, dt):
        """Smooth water surface (shallow water diffusion) to prevent spikes."""
        height = self.grid.water_height.data
        
        # Laplacian diffusion (prevents numerical instability)
        lap = (
            np.roll(height, 1, axis=0) + np.roll(height, -1, axis=0) +
            np.roll(height, 1, axis=1) + np.roll(height, -1, axis=1) -
            4.0 * height
        )
        
        # Light smoothing (high viscosity for water)
        smooth_amount = self.water_diffusion * 0.5 * dt
        height += smooth_amount * lap
        np.clip(height, 0.0, 3.0, out=height)  # Allow pooling (>1)

    def _compute_water_divergence(self, dt):
        """Compute divergence of velocity field (how much water is being compressed)."""
        vel_u = self.grid.velocity.u
        vel_v = self.grid.velocity.v
        size = self.grid.size
        
        # Divergence: ∇·v = ∂u/∂x + ∂v/∂y
        div_u = np.roll(vel_u, -1, axis=0) - np.roll(vel_u, 1, axis=0)
        div_v = np.roll(vel_v, -1, axis=1) - np.roll(vel_v, 1, axis=1)
        divergence = (div_u + div_v) * 0.25
        
        return divergence

    def _pressure_projection_step(self, dt):
        """Simple Jacobi iteration for pressure projection to enforce incompressibility."""
        height = self.grid.water_height.data
        vel_u = self.grid.velocity.u
        vel_v = self.grid.velocity.v
        
        # For water: incompressibility is enforced via pressure from height
        # Simple method: use height as proxy for pressure (shallow water)
        # Move water away from high-height regions
        
        grad_x, grad_y = self._compute_water_height_gradient()
        
        # Reduce water flux away from peaks
        reduction = 0.15 * dt
        vel_u -= reduction * grad_x * np.maximum(0.0, height - 0.5)
        vel_v -= reduction * grad_y * np.maximum(0.0, height - 0.5)

    def _advect_water_height(self, dt):
        """Semi-Lagrangian advection of water height by velocity field."""
        height = self.grid.water_height.data
        vel_u = self.grid.velocity.u
        vel_v = self.grid.velocity.v
        size = self.grid.size
        
        # Create grid coordinates [row, col]
        yy, xx = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32), indexing='ij')
        
        # Trace backwards
        x_back = xx - vel_u * dt
        y_back = yy - vel_v * dt
        
        # Clamp to bounds
        x_back = np.clip(x_back, 0.0, size - 1.0)
        y_back = np.clip(y_back, 0.0, size - 1.0)
        
        # Bilinear interpolation
        x0 = np.floor(x_back).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, size - 1)
        y0 = np.floor(y_back).astype(np.int32)
        y1 = np.clip(y0 + 1, 0, size - 1)
        
        sx = x_back - x0
        sy = y_back - y0
        
        height_new = (
            height[y0, x0] * (1 - sx) * (1 - sy) +
            height[y0, x1] * sx * (1 - sy) +
            height[y1, x0] * (1 - sx) * sy +
            height[y1, x1] * sx * sy
        )
        
        height[:] = height_new
        np.clip(height, 0.0, 3.0, out=height)

    def _apply_water_fire_interaction(self, dt):
        """Water cools fire and creates steam."""
        height = self.grid.water_height.data
        temp = self.grid.temperature.data
        soot = self.grid.soot.data
        
        # Water reduces temperature
        cooling = 0.4 * np.clip(height, 0.0, 1.0)
        temp *= (1.0 - cooling * dt)
        
        # Steam generation (soot) where water meets heat
        steam = np.clip(height * temp, 0.0, 1.0)
        soot += 0.2 * steam * dt
        np.clip(soot, 0.0, 1.0, out=soot)

    def _apply_fire_source(self, dt):
        size = self.grid.size
        wander = self.source_wander * self.chaos_level
        cx = int(size // 2 + wander * np.sin(self.sim_time * 2.7) + 0.5 * wander * np.sin(self.sim_time * 7.9))
        cx = int(np.clip(cx, 6, size - 7))
        cy = size - 9
        
        # Scale source geometry with fire_power (aggressive expansion for width).
        core_radius = int(self.fire_core_radius * (1.0 + 0.8 * (self.fire_power - 1.0)))
        outer_radius = int(self.fire_outer_radius * (1.0 + 1.2 * (self.fire_power - 1.0)))

        fuel = self.grid.fuel.data
        oxygen = self.grid.oxygen.data
        temp = self.grid.temperature.data
        vel_u = self.grid.velocity.u
        vel_v = self.grid.velocity.v

        outer_strength, core_strength, mask = self._radial_strength(cx, cy, outer_radius + int(self.chaos_level), core_radius)
        injection = (0.40 * outer_strength + 1.20 * core_strength) * mask

        base_pulse = 1.0 + 0.28 * self.chaos_level * np.sin(self.sim_time * 18.0)
        base_pulse += 0.16 * self.chaos_level * np.sin(self.sim_time * 43.0)
        base_pulse = max(0.65, base_pulse)

        # Scale emission and heat with fire_power.
        emission_scale = 0.8 + 0.6 * (self.fire_power - 1.0) / 3.0
        heat_scale = emission_scale
        
        fuel[:] = np.minimum(1.0, fuel + self.fire_emission * emission_scale * base_pulse * dt * injection)
        temp[:] = np.minimum(self.max_temperature, temp + self.fire_heat * heat_scale * base_pulse * dt * (0.25 * outer_strength + 1.20 * core_strength) * mask)
        oxygen[:] = np.minimum(1.0, oxygen + 0.40 * dt * outer_strength * mask)

        # Pilot core: prevent source from fully dying between substeps.
        core_zone = core_strength > 0.30
        fuel[core_zone] = np.maximum(fuel[core_zone], self.source_fuel_floor)
        temp[core_zone] = np.maximum(temp[core_zone], self.source_temp_floor)
        oxygen[core_zone] = np.maximum(oxygen[core_zone], self.source_oxygen_floor)

        # Scale buoyancy with fire_power.
        buoyancy_scale = 0.9 + 0.1 * (self.fire_power - 1.0) / 3.0
        up_force = self.fire_buoyancy * buoyancy_scale * dt * (0.4 * outer_strength + 1.2 * core_strength) * mask
        vel_v -= up_force

        coherent = self._coherent_turbulence(amplitude=self.fire_turbulence * (1.0 + 0.6 * self.chaos_level) * (0.8 + 0.2 * self.fire_power))
        random_jitter = np.random.uniform(-1.0, 1.0, size=vel_u.shape)
        vel_u += dt * (0.20 * coherent + 0.28 * self.fire_turbulence * (0.9 + 0.1 * self.fire_power) * random_jitter) * outer_strength * mask

        # Chaotic lateral sway to avoid static vertical columns (greatly scales with fire_power for width).
        sway = (0.8 * np.sin(self.sim_time * 5.2) + 0.6 * np.sin(self.sim_time * 11.7 + 1.3))
        vel_u += dt * sway * outer_strength * mask * (3.5 * self.chaos_level * (0.5 + 0.5 * self.fire_power))

        # Upward jet at burner for a clearly vertical flame direction.
        nozzle_half_width = 2
        nozzle_top = max(2, cy - 3)
        i0 = max(1, cx - nozzle_half_width)
        i1 = min(size - 1, cx + nozzle_half_width + 1)
        vel_v[i0:i1, nozzle_top:cy + 1] -= (self.fire_updraft_bias + 4.0 + 5.0 * self.chaos_level) * self.fire_power * dt
        vel_u[i0:i1, nozzle_top:cy + 1] *= 0.58

        # Relax lateral damping to allow wider fire spread (fire_power reduces constraint).
        y = np.arange(size, dtype=np.float32)
        height = (size - 1 - y) / max(1, size - 1)
        # Base damping reduced from 0.996 to 0.98, and scales down with fire_power for more spread.
        base_damp = 0.98 - 0.02 * height
        damp_reduction = 1.0 - 0.15 * (self.fire_power - 1.0) / 3.0  # Relaxes constraint as power increases
        lateral_damp = (base_damp * damp_reduction).reshape(1, -1)
        vel_u *= np.clip(lateral_damp, 0.80, 0.99)

        self._warp_velocity(dt)

    def _apply_diffusion(self, field, amount, dt, max_value=1.0):
        lap = (
            np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
            np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
            4.0 * field
        )
        field += amount * dt * lap
        np.clip(field, 0.0, max_value, out=field)

    def _apply_combustion(self, dt):
        fuel = self.grid.fuel.data
        oxygen = self.grid.oxygen.data
        temp = self.grid.temperature.data
        soot = self.grid.soot.data

        # Arrhenius-style combustion: k * fuel * oxygen * exp(-E / (R*T)).
        temp_k = 300.0 + 1400.0 * temp
        rate = self.combustion_rate * fuel * oxygen * np.exp(-self.activation_energy * 1000.0 / (8.314 * temp_k))

        # Ignition gate keeps flames alive once hot, avoids abrupt extinction.
        ignition = np.clip((temp - self.ignition_temperature) / max(1e-6, 1.0 - self.ignition_temperature), 0.0, 1.0)
        temp_norm = np.clip(temp / self.max_temperature, 0.0, 1.0)
        burst = 1.0 + self.burst_strength * self.chaos_level * np.maximum(0.0, np.sin(self.sim_time * self.burst_frequency))
        rate *= ignition * (0.35 + 1.35 * (temp_norm ** self.combustion_nonlinearity)) * burst

        burn = np.minimum(fuel, rate * dt)
        fuel -= burn
        oxygen -= self.oxygen_consumption * burn
        temp += self.heat_release * burn

        # Incomplete combustion forms soot in oxygen-poor regions.
        oxygen_deficit = np.clip(1.0 - oxygen, 0.0, 1.0)
        chaoticity = np.abs(self.grid.velocity.u) + np.abs(self.grid.velocity.v)
        chaoticity = np.clip(chaoticity / (1.0 + np.max(chaoticity) + 1e-6), 0.0, 1.0)
        soot += self.soot_yield * burn * (0.35 + 0.65 * oxygen_deficit) * (1.0 + 0.55 * chaoticity)

        # Hot oxygen-rich zones oxidize soot.
        soot_burn = np.minimum(soot, self.soot_oxidation * oxygen * temp * dt)
        soot -= soot_burn
        oxygen -= 0.25 * soot_burn
        temp += 0.35 * soot_burn

        np.clip(fuel, 0.0, 1.0, out=fuel)
        np.clip(oxygen, 0.0, 1.0, out=oxygen)
        # Temporary overburn allowed for explosive peaks.
        np.clip(temp, 0.0, self.max_temperature, out=temp)
        np.clip(soot, 0.0, 1.0, out=soot)

        # Smoke density is a blend of luminous flame + soot.
        lum = np.clip(temp / self.max_temperature, 0.0, 1.0)
        self.grid.density.data = np.clip(0.90 * lum + 0.40 * burn + 0.55 * soot, 0.0, 1.0)

    def _apply_radiation(self, dt):
        temp = self.grid.temperature.data
        amb = self.ambient_temperature

        # Radiative loss ~ T^4 (normalized), then neighborhood re-absorption.
        temp_norm = np.clip(temp / self.max_temperature, 0.0, 1.0)
        amb_norm = np.clip(amb / self.max_temperature, 0.0, 1.0)
        emissive = np.clip(temp_norm**4 - amb_norm**4, 0.0, 1.0)
        rad_loss = self.radiation_strength * emissive * dt
        temp -= rad_loss * self.max_temperature

        # Simple local re-absorption to preheat nearby cells/fuel.
        received = (
            np.roll(rad_loss, 1, axis=0) + np.roll(rad_loss, -1, axis=0) +
            np.roll(rad_loss, 1, axis=1) + np.roll(rad_loss, -1, axis=1)
        ) * 0.25
        temp += self.radiation_reabsorb * received

        fuel_preheat = self.radiation_reabsorb * 0.35 * received
        self.grid.fuel.data += fuel_preheat

        np.clip(self.grid.fuel.data, 0.0, 1.0, out=self.grid.fuel.data)
        np.clip(temp, 0.0, self.max_temperature, out=temp)

    def _apply_smoke_mode(self, dt):
        size = self.grid.size
        cx = size // 2
        cy = size - 8
        radius = 4

        fuel = self.grid.fuel.data
        temp = self.grid.temperature.data
        soot = self.grid.soot.data
        vel_u = self.grid.velocity.u
        vel_v = self.grid.velocity.v

        # Cooler source: mostly soot/smoke with modest heat.
        for i in range(cx - radius, cx + radius + 1):
            for j in range(cy - radius, cy + radius + 1):
                if 0 <= i < size and 0 <= j < size:
                    dist = np.sqrt((i - cx) ** 2 + (j - cy) ** 2)
                    if dist <= radius:
                        s = np.cos(dist / radius * np.pi / 2)
                        soot[i, j] = min(1.0, soot[i, j] + 1.7 * dt * s)
                        fuel[i, j] = min(1.0, fuel[i, j] + 0.8 * dt * s)
                        temp[i, j] = min(1.0, temp[i, j] + 0.35 * dt * s)
                        vel_v[i, j] -= 9.0 * dt * s
                        vel_u[i, j] += np.random.uniform(-2.0, 2.0) * dt * s

        self._apply_diffusion(soot, 0.16, dt)
        self._apply_diffusion(temp, 0.10, dt, max_value=self.max_temperature)
        self.grid.density.data = np.clip(0.18 * temp + 0.92 * soot, 0.0, 1.0)
        temp *= 0.992

    def _apply_upward_fire_transport(self):
        temp = self.grid.temperature.data
        density = self.grid.density.data

        transported = temp.copy()

        below = temp[1:-1, 2:]
        below_left = temp[:-2, 2:]
        below_right = temp[2:, 2:]

        doom_temp = (below + below_left + below_right) / 3.0
        doom_temp -= self.fire_transport_cooling * (1.0 + 0.7 * self.chaos_level)
        doom_temp -= np.random.uniform(0.0, self.fire_noise_decay, size=doom_temp.shape)
        doom_temp = np.clip(doom_temp, 0.0, self.max_temperature)

        blend = self.fire_transport_blend
        transported[1:-1, 1:-1] = (1.0 - blend) * temp[1:-1, 1:-1] + blend * doom_temp

        # Directional stretching: sample farther from below to elongate tongues upward.
        far_below = temp[1:-1, 3:]
        stretch_mix = np.clip(self.vertical_stretch, 0.0, 0.35)
        transported[1:-1, 1:-2] = (1.0 - stretch_mix) * transported[1:-1, 1:-2] + stretch_mix * far_below

        # Fragmentation creates splitting flame streaks.
        frag = np.sin(self._x[1:-1, 1:-1] * 0.55 + self.sim_time * 9.0) * np.cos(self._y[1:-1, 1:-1] * 0.33 - self.sim_time * 7.0)
        frag = np.clip(frag, -1.0, 1.0)
        transported[1:-1, 1:-1] += self.fire_fragmentation * self.chaos_level * frag * 0.04 * transported[1:-1, 1:-1]

        temp[:, :] = transported

        # Keep bright upward tongues but reduce side spread.
        density[:, :] = np.clip(0.70 * density + 0.30 * temp, 0.0, 1.0)

        # Ensure hot core does not collapse downward.
        hot_mask = temp > 0.56
        self.grid.velocity.v[hot_mask] = np.minimum(self.grid.velocity.v[hot_mask], -0.20)
        np.clip(temp, 0.0, self.max_temperature, out=temp)

    def _apply_thermal_expansion(self, dt):
        temp_norm = np.clip(self.grid.temperature.data / self.max_temperature, 0.0, 1.0)
        grad_x = 0.5 * (np.roll(temp_norm, -1, axis=0) - np.roll(temp_norm, 1, axis=0))
        grad_y = 0.5 * (np.roll(temp_norm, -1, axis=1) - np.roll(temp_norm, 1, axis=1))

        pressure_push = self.expansion_strength * self.chaos_level * dt
        self.grid.velocity.u += pressure_push * grad_x
        self.grid.velocity.v += pressure_push * grad_y

    def _apply_fire_chemistry_and_transport(self, dt):
        self._apply_combustion(dt)
        self._apply_diffusion(self.grid.temperature.data, self.fire_diffusion, dt, max_value=self.max_temperature)
        self._apply_diffusion(self.grid.fuel.data, 0.10, dt)
        self._apply_diffusion(self.grid.oxygen.data, 0.08, dt)
        self._apply_diffusion(self.grid.soot.data, 0.11, dt)
        self._apply_radiation(dt)
        self._apply_thermal_expansion(dt)

        # Cooling and oxygen replenishment from ambient air.
        self.grid.temperature.data *= self.fire_cooling
        self.grid.oxygen.data += 0.28 * dt * (1.0 - self.grid.oxygen.data)

        np.clip(self.grid.oxygen.data, 0.0, 1.0, out=self.grid.oxygen.data)
        np.clip(self.grid.temperature.data, 0.0, self.max_temperature, out=self.grid.temperature.data)
        np.clip(self.grid.soot.data, 0.0, 1.0, out=self.grid.soot.data)

        self._apply_upward_fire_transport()
        self._inject_instability(dt)

    def _compute_substeps(self, dt):
        # CFL-based adaptive stepping for stability at varying FPS/velocity.
        speed = np.sqrt(self.grid.velocity.u**2 + self.grid.velocity.v**2)
        vmax = float(np.max(speed))

        cfl_steps = int(np.ceil((vmax * dt) / self.cfl_target)) if vmax > 1e-6 else 1
        dt_steps = int(np.ceil(dt / (1.0 / 90.0))) if dt > 0.0 else 1

        return max(1, min(self.max_substeps, max(cfl_steps, dt_steps)))

    def _apply_thermodynamics_and_gravity(self, dt):
        temp = self.grid.temperature.data
        vel_v = self.grid.velocity.v
        vel_u = self.grid.velocity.u

        # Gravity acts downward (+v).
        vel_v[1:-1, 1:-1] += self.gravity * dt

        # Non-linear buoyancy: stronger lift at high temperature.
        temp_delta = temp - self.ambient_temperature
        hot = np.clip(temp_delta, 0.0, None)
        cold = np.clip(-temp_delta, 0.0, None)

        vel_v[1:-1, 1:-1] -= self.buoyancy_strength * (hot[1:-1, 1:-1] ** self.buoyancy_exponent) * dt
        vel_v[1:-1, 1:-1] += self.cold_sink_strength * (cold[1:-1, 1:-1] ** 1.05) * dt

        # Strong temperature-velocity feedback for aggressive inferno acceleration.
        temp_norm = np.clip(temp / self.max_temperature, 0.0, 1.0)
        vel_v -= self.temp_velocity_coupling * (temp_norm ** 1.7) * dt

        # Procedural turbulence for chaotic but coherent rolling motion.
        turb = self._coherent_turbulence(amplitude=self.turbulence_strength)
        hot_mask_strength = np.clip((temp - self.ignition_temperature) / (1.0 - self.ignition_temperature), 0.0, 1.0)
        vel_u += 0.35 * turb * hot_mask_strength * dt
        vel_v += -0.18 * np.roll(turb, 1, axis=0) * hot_mask_strength * dt

        # Slight damping of downward velocity in very hot zones to keep plume upward.
        hot_mask = temp > 0.52
        vel_v[hot_mask] = np.minimum(vel_v[hot_mask], -0.08)

        # Relax temperature toward ambient, then clamp.
        temp += (self.ambient_temperature - temp) * self.temperature_relaxation * dt
        np.clip(temp, 0.0, self.max_temperature, out=temp)

        # Velocity clamp for numeric stability during aggressive bursts.
        np.clip(vel_u, -110.0, 110.0, out=vel_u)
        np.clip(vel_v, -130.0, 130.0, out=vel_v)

    def _apply_wind_source(self, dt):
        size = self.grid.size
        src_x = 4
        src_y = size // 2
        radius = 6

        outer_strength, _, mask = self._radial_strength(src_x, src_y, radius, 1)
        falloff = outer_strength * mask
        gust = self.wind_force * dt * falloff

        self.grid.velocity.u += self.wind_dir[0] * gust
        self.grid.velocity.v += self.wind_dir[1] * gust
        self.grid.velocity.v += np.random.uniform(-self.wind_noise, self.wind_noise, size=self.grid.velocity.v.shape) * dt * falloff

        self.grid.density.data[:] = np.minimum(
            0.35,
            self.grid.density.data + 0.8 * dt * falloff
        )

        self.grid.temperature.data *= 0.94
        self.grid.density.data *= 0.988
        self.grid.oxygen.data += 0.10 * dt * (1.0 - self.grid.oxygen.data)
        np.clip(self.grid.oxygen.data, 0.0, 1.0, out=self.grid.oxygen.data)

    def update(self, dt):
        self.dt = dt
        self.sim_time += dt

        if hasattr(self.simulation, "vorticity"):
            if self.element_mode == "fire":
                self.simulation.vorticity.strength = (4.6 + 2.2 * self.chaos_level) * self.fire_power
            else:
                self.simulation.vorticity.strength = 2.0

        substeps = self._compute_substeps(dt)
        sub_dt = dt / substeps

        for _ in range(substeps):
            if self.element_mode == "wind":
                self._apply_wind_source(sub_dt)
            elif self.element_mode == "smoke":
                self._apply_smoke_mode(sub_dt)
            elif self.element_mode == "water":
                self._apply_water_source(sub_dt)
                self._apply_water_gravity(sub_dt)
                self._apply_water_height_pressure_forces(sub_dt)
                self._pressure_projection_step(sub_dt)
                self._advect_water_height(sub_dt)
                self._apply_water_surface_smoothing(sub_dt)
                self._apply_water_fire_interaction(sub_dt)
            else:
                self._apply_fire_source(sub_dt)
                self._apply_fire_chemistry_and_transport(sub_dt)

            self._apply_thermodynamics_and_gravity(sub_dt)
            if self.element_mode == "fire":
                self._enforce_vertical_fire_orientation(sub_dt)
            self.simulation.step(sub_dt)
