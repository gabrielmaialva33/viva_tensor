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
pub fn new(count: Int, dims: Int) -> Result(Horde, String) {
  ffi.horde_create(count, dims)
}

/// Set positions for all entities
///
/// Data must be a flat list: [x0, y0, x1, y1, ...]
pub fn set_positions(horde: Horde, data: List(Float)) -> Result(Horde, String) {
  case ffi.horde_set_positions(horde, data) {
    Ok(Nil) -> Ok(horde)
    Error(error) -> Error(error)
  }
}

/// Set velocities for all entities
pub fn set_velocities(
  horde: Horde,
  data: List(Float),
) -> Result(Horde, String) {
  case ffi.horde_set_velocities(horde, data) {
    Ok(Nil) -> Ok(horde)
    Error(error) -> Error(error)
  }
}

/// Advance simulation by dt (seconds)
///
/// Positions += Velocities * dt
pub fn integrate(horde: Horde, dt: Float) -> Result(Horde, String) {
  case ffi.horde_integrate(horde, dt) {
    Ok(Nil) -> Ok(horde)
    Error(error) -> Error(error)
  }
}

/// Apply damping (friction)
///
/// Velocities *= friction (0.0 to 1.0)
pub fn dampen(horde: Horde, friction: Float) -> Result(Horde, String) {
  case ffi.horde_dampen(horde, friction) {
    Ok(Nil) -> Ok(horde)
    Error(error) -> Error(error)
  }
}

/// Wrap positions torus-style
///
/// If pos > max, pos -= max
pub fn wrap(horde: Horde, max_bound: Float) -> Result(Horde, String) {
  case ffi.horde_wrap(horde, max_bound) {
    Ok(Nil) -> Ok(horde)
    Error(error) -> Error(error)
  }
}

/// Get current positions
pub fn positions(horde: Horde) -> Result(List(Float), String) {
  ffi.horde_get_positions(horde)
}

/// Get current velocities
pub fn velocities(horde: Horde) -> Result(List(Float), String) {
  ffi.horde_get_velocities(horde)
}

/// Get total kinetic energy of the system
pub fn kinetic_energy(horde: Horde) -> Result(Float, String) {
  ffi.horde_kinetic_energy(horde)
}
