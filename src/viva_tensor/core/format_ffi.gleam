//// FFI wrappers for Erlang's float formatting primitives.
////
//// These give us precision-bound and scientific-notation rendering of
//// f64 values without re-implementing Dragon4 on the BEAM. Everything
//// here is internal — public formatting goes through
//// `viva_tensor/core/format`.

/// Render a float in fixed notation with `decimals` fractional digits.
///
/// Backed by Erlang `float_to_binary(F, [{decimals, N}])`.
@external(erlang, "viva_tensor_format_ffi", "fmt_fixed")
pub fn fmt_fixed(value: Float, decimals: Int) -> String

/// Render a float in scientific notation with `decimals` fractional digits
/// in the mantissa.
///
/// Backed by Erlang `float_to_binary(F, [{scientific, N}])`.
@external(erlang, "viva_tensor_format_ffi", "fmt_sci")
pub fn fmt_sci(value: Float, decimals: Int) -> String

/// Render a float using shortest-roundtrip decimal (OTP ≥ 25).
///
/// Backed by Erlang `float_to_binary(F, [short])`.
@external(erlang, "viva_tensor_format_ffi", "fmt_short")
pub fn fmt_short(value: Float) -> String

/// True when the float is finite (not NaN, not +/-Inf).
@external(erlang, "viva_tensor_format_ffi", "is_finite")
pub fn is_finite(value: Float) -> Bool

/// True when the float is NaN.
@external(erlang, "viva_tensor_format_ffi", "is_nan")
pub fn is_nan(value: Float) -> Bool

/// True when the float is +Inf or -Inf.
@external(erlang, "viva_tensor_format_ffi", "is_inf")
pub fn is_inf(value: Float) -> Bool
