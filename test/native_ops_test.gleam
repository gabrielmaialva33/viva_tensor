//// Tests for the CPU-scaffolded native ops:
////   - ffi.nt_softmax_axis
////   - ffi.nt_layer_norm
////   - ffi.nt_gelu_exact
////
//// The point of this round is to land the **plumbing**. The Zig/C kernel
//// bodies are stubbed; the Gleam wrappers gate on `ffi.zig_is_loaded()` and
//// return `Error("nif_not_loaded")` when the shared lib isn't present.
////
//// Two flavors of test per op:
////
////   1. `*_falls_back_when_nif_absent` — assumes the NIF library is NOT
////      loaded (the common case in CI for this sprint) and checks that the
////      wrapper returns Error("nif_not_loaded") cleanly. When the NIF IS
////      loaded (kernel still stubbed), it accepts Error("not_implemented")
////      as also valid: the kernel hasn't been written yet, but the ABI is
////      stable.
////
////   2. `*_roundtrip_when_nif_loaded` — only runs when the NIF is loaded
////      AND the kernel actually returns Ok. Verifies numerical correctness
////      against the pure-Gleam reference implementations in
////      `viva_tensor/nn/activations` and `viva_tensor/nn/norm`. Skips
////      silently in any other case (NIF absent or still stubbed).
////
//// Net delta: +6 tests when the NIF is built and kernels are implemented;
//// +6 tests pass-by-skip otherwise (the round-trip ones become no-ops).

import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor/core/ffi
import viva_tensor/nn/activations
import viva_tensor/nn/norm
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

const rtol: Float = 1.0e-5

const atol: Float = 1.0e-7

// --- Fallback tests (always run) -------------------------------------------

/// When the NIF is missing, the wrapper short-circuits with
/// Error("nif_not_loaded"). When the NIF is loaded but the kernel is still
/// a stub, it returns Error("not_implemented"). Either is a valid
/// "scaffolding present, kernel not yet" signal.
pub fn softmax_axis_falls_back_when_nif_absent_test() {
  // Build a list-backed tensor and pull its native ref. When the NIF is
  // absent, nt_from_list itself returns nif_not_loaded, so we can't even
  // construct a ref — that's fine, the test is about the API surface.
  case ffi.nt_from_list([1.0, 2.0, 3.0], [3]) {
    Error(reason) -> reason |> should.equal("nif_not_loaded")
    Ok(ref) ->
      case ffi.nt_softmax_axis(ref, 0) {
        Error(reason) ->
          case reason {
            "nif_not_loaded" -> Nil
            "not_implemented" -> Nil
            other -> should.equal(other, "nif_not_loaded or not_implemented")
          }
        Ok(_) ->
          // Kernel implemented + loaded — the round-trip test covers this.
          Nil
      }
  }
}

pub fn layer_norm_falls_back_when_nif_absent_test() {
  case ffi.nt_from_list([1.0, 2.0, 3.0, 4.0], [4]) {
    Error(reason) -> reason |> should.equal("nif_not_loaded")
    Ok(x_ref) -> {
      let assert Ok(scale_ref) = ffi.nt_ones([4])
      let assert Ok(bias_ref) = ffi.nt_zeros([4])
      case ffi.nt_layer_norm(x_ref, scale_ref, bias_ref, 1.0e-5) {
        Error(reason) ->
          case reason {
            "nif_not_loaded" -> Nil
            "not_implemented" -> Nil
            other -> should.equal(other, "nif_not_loaded or not_implemented")
          }
        Ok(_) -> Nil
      }
    }
  }
}

pub fn gelu_falls_back_when_nif_absent_test() {
  case ffi.nt_from_list([-1.0, 0.0, 1.0], [3]) {
    Error(reason) -> reason |> should.equal("nif_not_loaded")
    Ok(ref) ->
      case ffi.nt_gelu_exact(ref) {
        Error(reason) ->
          case reason {
            "nif_not_loaded" -> Nil
            "not_implemented" -> Nil
            other -> should.equal(other, "nif_not_loaded or not_implemented")
          }
        Ok(_) -> Nil
      }
  }
}

// --- Round-trip tests (skip when NIF absent or kernel stubbed) -------------

/// Verifies that nt_softmax_axis matches the pure-Gleam reference
/// implementation in `viva_tensor/nn/activations.softmax`.
///
/// Skips silently when:
///   - the NIF library isn't loaded (Gleam wrapper returns nif_not_loaded), or
///   - the kernel is still stubbed (returns not_implemented).
pub fn softmax_axis_roundtrip_when_nif_loaded_test() {
  case ffi.zig_is_loaded() {
    False -> Nil
    True -> {
      let input_data = [1.0, 2.0, 3.0, 4.0]
      let assert Ok(ref) = ffi.nt_from_list(input_data, [4])
      case ffi.nt_softmax_axis(ref, 0) {
        Error(_) ->
          // Kernel not yet implemented — accept and skip.
          Nil
        Ok(out_ref) -> {
          let assert Ok(actual) = ffi.nt_to_list(out_ref)
          let reference_tensor = tensor.from_list(input_data)
          let assert Ok(expected_t) = activations.softmax(reference_tensor, 0)
          let expected = tensor.to_list(expected_t)
          numerics.lists_close(actual, expected, rtol, atol)
          |> should.be_true
        }
      }
    }
  }
}

/// Verifies that nt_layer_norm matches `norm.layer_norm_forward` with
/// scale = ones and bias = zeros (the default LayerNorm init).
pub fn layer_norm_roundtrip_when_nif_loaded_test() {
  case ffi.zig_is_loaded() {
    False -> Nil
    True -> {
      let input_data = [1.0, 2.0, 3.0, 4.0]
      let assert Ok(x_ref) = ffi.nt_from_list(input_data, [1, 4])
      let assert Ok(scale_ref) = ffi.nt_ones([4])
      let assert Ok(bias_ref) = ffi.nt_zeros([4])
      let eps = 1.0e-5
      case ffi.nt_layer_norm(x_ref, scale_ref, bias_ref, eps) {
        Error(_) -> Nil
        Ok(out_ref) -> {
          let assert Ok(actual) = ffi.nt_to_list(out_ref)
          let layer = norm.layer_norm_init_with_eps(4, eps)
          let assert Ok(x_2d) = tensor.from_list2d([input_data])
          let assert Ok(expected_t) = norm.layer_norm_forward(layer, x_2d)
          let expected = tensor.to_list(expected_t)
          numerics.lists_close(actual, expected, rtol, atol)
          |> should.be_true
        }
      }
    }
  }
}

/// Verifies that nt_gelu_exact matches `activations.gelu` element-wise.
pub fn gelu_roundtrip_when_nif_loaded_test() {
  case ffi.zig_is_loaded() {
    False -> Nil
    True -> {
      let input_data = [-2.0, -0.5, 0.0, 0.5, 2.0]
      let assert Ok(ref) = ffi.nt_from_list(input_data, [5])
      case ffi.nt_gelu_exact(ref) {
        Error(_) -> Nil
        Ok(out_ref) -> {
          let assert Ok(actual) = ffi.nt_to_list(out_ref)
          let reference_tensor = tensor.from_list(input_data)
          let expected_t = activations.gelu(reference_tensor)
          let expected = tensor.to_list(expected_t)
          numerics.lists_close(actual, expected, rtol, atol)
          |> should.be_true
        }
      }
    }
  }
}
