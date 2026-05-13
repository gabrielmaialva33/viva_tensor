//// SafeTensors I/O — HuggingFace's safe binary format for tensor weights.
////
//// Wire layout (little-endian, row-major):
////
//// ```
//// [8 bytes: u64 header length N]
//// [N bytes: UTF-8 JSON header object]
//// [remaining: tensor payload bytes, concatenated in header declaration order]
//// ```
////
//// Header is a JSON object mapping tensor name to
//// `{ "dtype": "F64", "shape": [..], "data_offsets": [start, end] }` with an
//// optional sibling `__metadata__` for string-to-string metadata.
////
//// This module exposes:
//// - `read/1`     — parse a `.safetensors` file into a `Dict(String, Tensor)`.
//// - `write/2`    — serialize tensors as F64 little-endian.
//// - `metadata_of/1` — peek `__metadata__` without materializing tensors.
////
//// v1 supports `F32` and `F64` dtypes on read; other dtypes return a
//// `DtypeError`. Writes always use `F64`.

import gleam/bit_array
import gleam/dict.{type Dict}
import gleam/dynamic/decode
import gleam/int
import gleam/json
import gleam/list
import gleam/result
import gleam/string
import simplifile
import viva_tensor/core/error.{type TensorError, DtypeError, InvalidShape}
import viva_tensor/tensor.{type Tensor, Tensor}

// --- Public API -------------------------------------------------------------

/// Read a SafeTensors file into a `Dict(String, Tensor)` keyed by tensor name.
///
/// Supports `F32` and `F64` payloads (other dtypes return `DtypeError`).
/// Malformed headers or out-of-range offsets surface as `InvalidShape`.
///
/// ## Example
///
/// ```gleam
/// import gleam/dict
/// import viva_tensor/io/safetensors
///
/// let assert Ok(tensors) = safetensors.read("./weights.safetensors")
/// let _ = dict.get(tensors, "weight")
/// ```
pub fn read(path: String) -> Result(Dict(String, Tensor), TensorError) {
  use bytes <- result.try(read_file_bytes(path))
  use #(header, payload) <- result.try(split_header(bytes))
  use entries <- result.try(parse_header_entries(header))
  decode_tensors(entries, payload)
}

/// Write a `Dict(String, Tensor)` to disk in SafeTensors format.
///
/// All tensors are emitted as `F64` little-endian since Gleam `Float` is the
/// BEAM 64-bit float. Tensor names are sorted alphabetically for stable output.
///
/// ## Example
///
/// ```gleam
/// import gleam/dict
/// import viva_tensor as t
/// import viva_tensor/io/safetensors
///
/// let weights = dict.from_list([#("w", t.ones([2, 2]))])
/// let assert Ok(Nil) = safetensors.write("./out.safetensors", weights)
/// ```
pub fn write(
  path: String,
  tensors: Dict(String, Tensor),
) -> Result(Nil, TensorError) {
  write_with_metadata(path, tensors, dict.new())
}

/// Like `write/2`, but also embeds a `__metadata__` block in the header.
///
/// Metadata is restricted to string-to-string mappings (per the SafeTensors
/// spec) and survives a `metadata_of/1` round-trip.
///
/// ## Example
///
/// ```gleam
/// import gleam/dict
/// import viva_tensor/io/safetensors
///
/// let meta = dict.from_list([#("framework", "viva_tensor")])
/// let assert Ok(Nil) =
///   safetensors.write_with_metadata("./out.safetensors", dict.new(), meta)
/// ```
pub fn write_with_metadata(
  path: String,
  tensors: Dict(String, Tensor),
  metadata: Dict(String, String),
) -> Result(Nil, TensorError) {
  let entries = sorted_entries(tensors)
  use plans <- result.try(plan_entries(entries, 0))
  let header_json = build_header_json(plans, metadata)
  let header_bytes = bit_array.from_string(header_json)
  let header_len = bit_array.byte_size(header_bytes)
  let len_prefix = <<header_len:little-size(64)>>
  let payload =
    list.fold(plans, <<>>, fn(acc, plan) {
      bit_array.append(acc, floats_to_bytes(plan.data))
    })
  let blob =
    len_prefix
    |> bit_array.append(header_bytes)
    |> bit_array.append(payload)
  case simplifile.write_bits(to: path, bits: blob) {
    Ok(Nil) -> Ok(Nil)
    Error(err) ->
      Error(InvalidShape(
        "SafeTensors: write failed: " <> file_error_to_string(err),
      ))
  }
}

/// Read the `__metadata__` block of a SafeTensors file without materializing
/// any tensor data. Returns an empty dict when the header has no metadata.
///
/// ## Example
///
/// ```gleam
/// import viva_tensor/io/safetensors
///
/// let assert Ok(meta) = safetensors.metadata_of("./weights.safetensors")
/// ```
pub fn metadata_of(path: String) -> Result(Dict(String, String), TensorError) {
  use bytes <- result.try(read_file_bytes(path))
  use #(header, _payload) <- result.try(split_header(bytes))
  parse_metadata(header)
}

// --- Byte helpers -----------------------------------------------------------

/// Encode a single `Float` (BEAM f64) as 8 little-endian bytes.
fn f64_to_bytes(value: Float) -> BitArray {
  <<value:little-float-size(64)>>
}

/// Decode 8 little-endian bytes into a `Float`. Fails (returns 0.0) when the
/// input is not exactly 8 bytes; callers must slice first.
fn bytes_to_f64(bytes: BitArray) -> Float {
  case bytes {
    <<value:little-float-size(64)>> -> value
    _ -> 0.0
  }
}

fn f32_bytes_to_f64(bytes: BitArray) -> Float {
  case bytes {
    <<value:little-float-size(32)>> -> value
    _ -> 0.0
  }
}

fn floats_to_bytes(values: List(Float)) -> BitArray {
  list.fold(values, <<>>, fn(acc, v) { bit_array.append(acc, f64_to_bytes(v)) })
}

// --- File reading -----------------------------------------------------------

fn read_file_bytes(path: String) -> Result(BitArray, TensorError) {
  case simplifile.read_bits(from: path) {
    Ok(bytes) -> Ok(bytes)
    Error(err) ->
      Error(InvalidShape(
        "SafeTensors: cannot read file: " <> file_error_to_string(err),
      ))
  }
}

fn file_error_to_string(err: simplifile.FileError) -> String {
  simplifile.describe_error(err)
}

fn split_header(bytes: BitArray) -> Result(#(String, BitArray), TensorError) {
  case bytes {
    <<header_len:little-unsigned-size(64), rest:bits>> -> {
      let rest_size = bit_array.byte_size(rest)
      case header_len <= rest_size {
        False ->
          Error(InvalidShape(
            "SafeTensors: header length "
            <> int.to_string(header_len)
            <> " exceeds remaining "
            <> int.to_string(rest_size)
            <> " bytes",
          ))
        True -> {
          use header_bits <- result.try(slice_or_fail(
            rest,
            0,
            header_len,
            "header bytes",
          ))
          use payload <- result.try(slice_or_fail(
            rest,
            header_len,
            rest_size - header_len,
            "payload bytes",
          ))
          case bit_array.to_string(header_bits) {
            Ok(header_text) -> Ok(#(header_text, payload))
            Error(_) ->
              Error(InvalidShape("SafeTensors: header is not valid UTF-8"))
          }
        }
      }
    }
    _ ->
      Error(InvalidShape("SafeTensors: file too short for 8-byte header length"))
  }
}

fn slice_or_fail(
  source: BitArray,
  start: Int,
  length: Int,
  label: String,
) -> Result(BitArray, TensorError) {
  case bit_array.slice(from: source, at: start, take: length) {
    Ok(slice) -> Ok(slice)
    Error(_) -> Error(InvalidShape("SafeTensors: cannot slice " <> label))
  }
}

// --- Header parsing ---------------------------------------------------------

type HeaderEntry {
  HeaderEntry(
    name: String,
    dtype: String,
    shape: List(Int),
    offsets: #(Int, Int),
  )
}

fn parse_header_entries(
  header: String,
) -> Result(List(HeaderEntry), TensorError) {
  let entry_decoder = {
    use dtype <- decode.field("dtype", decode.string)
    use shape <- decode.field("shape", decode.list(decode.int))
    use offsets <- decode.field("data_offsets", decode.list(decode.int))
    case offsets {
      [start, end] -> decode.success(#(dtype, shape, #(start, end)))
      _ ->
        decode.failure(
          #("F64", [], #(0, 0)),
          "data_offsets must be [start, end]",
        )
    }
  }
  // Decode each value lazily as Dynamic so we can skip `__metadata__` before
  // applying the entry-shaped decoder (which would fail on the metadata block).
  let raw_decoder = decode.dict(decode.string, decode.dynamic)
  case json.parse(from: header, using: raw_decoder) {
    Ok(raw_entries) -> {
      let pairs =
        dict.to_list(raw_entries)
        |> list.filter(fn(pair) {
          let #(name, _) = pair
          name != "__metadata__"
        })
      decode_entries_from_dynamics(pairs, entry_decoder, [])
    }
    Error(_) -> Error(InvalidShape("SafeTensors: malformed JSON header"))
  }
}

fn decode_entries_from_dynamics(
  pairs: List(#(String, decode.Dynamic)),
  entry_decoder: decode.Decoder(#(String, List(Int), #(Int, Int))),
  acc: List(HeaderEntry),
) -> Result(List(HeaderEntry), TensorError) {
  case pairs {
    [] -> Ok(list.reverse(acc))
    [#(name, dyn), ..rest] -> {
      case decode.run(dyn, entry_decoder) {
        Ok(#(dtype, shape, offsets)) ->
          decode_entries_from_dynamics(rest, entry_decoder, [
            HeaderEntry(name, dtype, shape, offsets),
            ..acc
          ])
        Error(_) ->
          Error(InvalidShape("SafeTensors: malformed entry for tensor " <> name))
      }
    }
  }
}

fn parse_metadata(header: String) -> Result(Dict(String, String), TensorError) {
  let metadata_decoder = decode.dict(decode.string, decode.string)
  let outer_decoder = {
    use meta <- decode.optional_field(
      "__metadata__",
      dict.new(),
      metadata_decoder,
    )
    decode.success(meta)
  }
  case json.parse(from: header, using: outer_decoder) {
    Ok(meta) -> Ok(meta)
    Error(_) -> Error(InvalidShape("SafeTensors: malformed JSON header"))
  }
}

// --- Tensor decoding --------------------------------------------------------

fn decode_tensors(
  entries: List(HeaderEntry),
  payload: BitArray,
) -> Result(Dict(String, Tensor), TensorError) {
  let payload_size = bit_array.byte_size(payload)
  decode_tensors_loop(entries, payload, payload_size, dict.new())
}

fn decode_tensors_loop(
  entries: List(HeaderEntry),
  payload: BitArray,
  payload_size: Int,
  acc: Dict(String, Tensor),
) -> Result(Dict(String, Tensor), TensorError) {
  case entries {
    [] -> Ok(acc)
    [entry, ..rest] -> {
      use tensor <- result.try(decode_entry(entry, payload, payload_size))
      decode_tensors_loop(
        rest,
        payload,
        payload_size,
        dict.insert(acc, entry.name, tensor),
      )
    }
  }
}

fn decode_entry(
  entry: HeaderEntry,
  payload: BitArray,
  payload_size: Int,
) -> Result(Tensor, TensorError) {
  let #(start, end) = entry.offsets
  case start < 0 || end < start || end > payload_size {
    True ->
      Error(InvalidShape(
        "SafeTensors: data_offsets out of range for tensor " <> entry.name,
      ))
    False -> {
      let length = end - start
      use slice <- result.try(slice_or_fail(
        payload,
        start,
        length,
        "tensor " <> entry.name,
      ))
      use data <- result.try(decode_payload(entry.dtype, slice, length))
      use _ <- result.try(verify_shape(entry, list.length(data)))
      Ok(Tensor(data: data, shape: entry.shape))
    }
  }
}

fn verify_shape(entry: HeaderEntry, count: Int) -> Result(Nil, TensorError) {
  let expected = list.fold(entry.shape, 1, fn(acc, dim) { acc * dim })
  case expected == count {
    True -> Ok(Nil)
    False ->
      Error(InvalidShape(
        "SafeTensors: tensor "
        <> entry.name
        <> " expected "
        <> int.to_string(expected)
        <> " elements but found "
        <> int.to_string(count),
      ))
  }
}

fn decode_payload(
  dtype: String,
  slice: BitArray,
  length: Int,
) -> Result(List(Float), TensorError) {
  case dtype {
    "F64" -> decode_f64_chunks(slice, length, [])
    "F32" -> decode_f32_chunks(slice, length, [])
    other -> Error(DtypeError("SafeTensors: unsupported dtype " <> other))
  }
}

fn decode_f64_chunks(
  slice: BitArray,
  remaining: Int,
  acc: List(Float),
) -> Result(List(Float), TensorError) {
  case remaining {
    0 -> Ok(list.reverse(acc))
    _ -> {
      case remaining < 8 {
        True ->
          Error(InvalidShape(
            "SafeTensors: F64 payload length not a multiple of 8",
          ))
        False -> {
          let offset = bit_array.byte_size(slice) - remaining
          use chunk <- result.try(slice_or_fail(slice, offset, 8, "F64 chunk"))
          let value = bytes_to_f64(chunk)
          decode_f64_chunks(slice, remaining - 8, [value, ..acc])
        }
      }
    }
  }
}

fn decode_f32_chunks(
  slice: BitArray,
  remaining: Int,
  acc: List(Float),
) -> Result(List(Float), TensorError) {
  case remaining {
    0 -> Ok(list.reverse(acc))
    _ -> {
      case remaining < 4 {
        True ->
          Error(InvalidShape(
            "SafeTensors: F32 payload length not a multiple of 4",
          ))
        False -> {
          let offset = bit_array.byte_size(slice) - remaining
          use chunk <- result.try(slice_or_fail(slice, offset, 4, "F32 chunk"))
          let value = f32_bytes_to_f64(chunk)
          decode_f32_chunks(slice, remaining - 4, [value, ..acc])
        }
      }
    }
  }
}

// --- Header building (write path) ------------------------------------------

type WritePlan {
  WritePlan(
    name: String,
    shape: List(Int),
    data: List(Float),
    start: Int,
    end: Int,
  )
}

fn sorted_entries(tensors: Dict(String, Tensor)) -> List(#(String, Tensor)) {
  tensors
  |> dict.to_list
  |> list.sort(fn(a, b) {
    let #(name_a, _) = a
    let #(name_b, _) = b
    string.compare(name_a, name_b)
  })
}

fn plan_entries(
  entries: List(#(String, Tensor)),
  cursor: Int,
) -> Result(List(WritePlan), TensorError) {
  plan_entries_loop(entries, cursor, [])
}

fn plan_entries_loop(
  entries: List(#(String, Tensor)),
  cursor: Int,
  acc: List(WritePlan),
) -> Result(List(WritePlan), TensorError) {
  case entries {
    [] -> Ok(list.reverse(acc))
    [#(name, t), ..rest] -> {
      let data = tensor.to_list(t)
      let shape = tensor.shape(t)
      let expected = list.fold(shape, 1, fn(acc_dim, dim) { acc_dim * dim })
      case expected == list.length(data) {
        False ->
          Error(InvalidShape(
            "SafeTensors: tensor " <> name <> " has shape/data length mismatch",
          ))
        True -> {
          let byte_len = list.length(data) * 8
          let end = cursor + byte_len
          let plan = WritePlan(name, shape, data, cursor, end)
          plan_entries_loop(rest, end, [plan, ..acc])
        }
      }
    }
  }
}

fn build_header_json(
  plans: List(WritePlan),
  metadata: Dict(String, String),
) -> String {
  let tensor_entries =
    list.map(plans, fn(plan) {
      #(
        plan.name,
        json.object([
          #("dtype", json.string("F64")),
          #("shape", json.array(plan.shape, of: json.int)),
          #(
            "data_offsets",
            json.preprocessed_array([json.int(plan.start), json.int(plan.end)]),
          ),
        ]),
      )
    })
  let entries = case dict.size(metadata) {
    0 -> tensor_entries
    _ -> {
      let meta_pairs =
        dict.to_list(metadata)
        |> list.map(fn(pair) {
          let #(k, v) = pair
          #(k, json.string(v))
        })
      [#("__metadata__", json.object(meta_pairs)), ..tensor_entries]
    }
  }
  json.object(entries) |> json.to_string
}
