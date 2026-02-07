//// Horde - Massively Parallel Entity Physics
////
//// A Structure-of-Arrays (SoA) physics engine optimized for 10,000+ entities.
//// Data is stored in contiguous buffers (Positions, Velocities) for SIMD acceleration.
////
//// Ideal for: particle systems, boids, swarms, and massive game entities.

import viva_tensor/core/ffi

/// Horde reference (physics engine state)
pub type Horde =
  ffi.HordeRef

/// Create a new Horde
///
/// - count: Number of entities
/// - dims: Dimensions per entity (1, 2, or 3)
pub fn new(count: Int, dims: Int) -> Horde {
  let assert Ok(h) = ffi.horde_create(count, dims)
  h
}

/// Set positions for all entities
///
/// Data must be a flat list: [x0, y0, x1, y1, ...]
pub fn set_positions(horde: Horde, data: List(Float)) -> Horde {
  let assert Ok(Nil) = ffi.horde_set_positions(horde, data)
  horde
}

/// Set velocities for all entities
pub fn set_velocities(horde: Horde, data: List(Float)) -> Horde {
  let assert Ok(Nil) = ffi.horde_set_velocities(horde, data)
  horde
}

/// Advance simulation by dt (seconds)
///
/// Positions += Velocities * dt
pub fn integrate(horde: Horde, dt: Float) -> Horde {
  let assert Ok(Nil) = ffi.horde_integrate(horde, dt)
  horde
}

/// Apply damping (friction)
///
/// Velocities *= friction (0.0 to 1.0)
pub fn dampen(horde: Horde, friction: Float) -> Horde {
  let assert Ok(Nil) = ffi.horde_dampen(horde, friction)
  horde
}

/// Wrap positions torus-style
///
/// If pos > max, pos -= max
pub fn wrap(horde: Horde, max_bound: Float) -> Horde {
  let assert Ok(Nil) = ffi.horde_wrap(horde, max_bound)
  horde
}

/// Get current positions
pub fn positions(horde: Horde) -> List(Float) {
  let assert Ok(data) = ffi.horde_get_positions(horde)
  data
}

/// Get current velocities
pub fn velocities(horde: Horde) -> List(Float) {
  let assert Ok(data) = ffi.horde_get_velocities(horde)
  data
}

/// Get total kinetic energy of the system
pub fn kinetic_energy(horde: Horde) -> Float {
  let assert Ok(ke) = ffi.horde_kinetic_energy(horde)
  ke
}
