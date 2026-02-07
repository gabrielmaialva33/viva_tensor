//// LNS - Log-Number System
////
//// Efficient arithmetic for resonance and probabilistic computing.
////
//// Represents numbers as their logarithms: x => log(|x|) + sign.
//// Multiplication becomes addition: log(a*b) = log(a) + log(b).
//// Division becomes subtraction: log(a/b) = log(a) - log(b).
//// Power becomes multiplication: log(x^p) = p * log(x).
////
//// Used for:
//// - High-dynamic range calculations
//// - Chain multiplications (probability chains)
//// - "Resonance" calculations (heavy multiplication/exponentiation)
////
//// Precision is traded for speed (up to 8x faster than FP32 FMA).

import viva_tensor/core/ffi

/// LNS Tensor reference (storage in f32 log-domain)
pub type LogNumber =
  ffi.LnsTensorRef

/// Convert standard f64 tensor to LNS format
pub fn from_tensor(ref: ffi.NativeTensorRef) -> LogNumber {
  let assert Ok(lns) = ffi.lns_from_f64(ref)
  lns
}

/// Convert LNS tensor back to standard f64 tensor
pub fn to_tensor(lns: LogNumber) -> ffi.NativeTensorRef {
  let assert Ok(ref) = ffi.lns_to_f64(lns)
  ref
}

/// Multiply two LNS tensors (Addition in log-domain)
pub fn mul(a: LogNumber, b: LogNumber) -> LogNumber {
  let assert Ok(res) = ffi.lns_mul(a, b)
  res
}

/// Divide two LNS tensors (Subtraction in log-domain)
pub fn div(a: LogNumber, b: LogNumber) -> LogNumber {
  let assert Ok(res) = ffi.lns_div(a, b)
  res
}

/// Square root of LNS tensor (Bit shift / division by 2 in log-domain)
pub fn sqrt(a: LogNumber) -> LogNumber {
  let assert Ok(res) = ffi.lns_sqrt(a)
  res
}
