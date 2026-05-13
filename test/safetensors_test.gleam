import gleam/bit_array
import gleam/dict
import gleeunit/should
import simplifile
import viva_tensor as t
import viva_tensor/io/safetensors
import viva_tensor/tensor.{Tensor}

const tmp_dir = "./tmp/safetensors_test"

fn ensure_tmp() -> Nil {
  let _ = simplifile.create_directory_all(tmp_dir)
  Nil
}

fn cleanup(path: String) -> Nil {
  let _ = simplifile.delete(path)
  Nil
}

pub fn safetensors_roundtrip_single_test() {
  ensure_tmp()
  let path = tmp_dir <> "/single.safetensors"
  let t1 = Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], shape: [2, 3])
  let tensors = dict.from_list([#("weight", t1)])

  let assert Ok(Nil) = safetensors.write(path, tensors)
  let assert Ok(loaded) = safetensors.read(path)
  let assert Ok(out) = dict.get(loaded, "weight")

  t.shape(out) |> should.equal([2, 3])
  t.to_list(out) |> should.equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

  cleanup(path)
}

pub fn safetensors_roundtrip_multi_test() {
  ensure_tmp()
  let path = tmp_dir <> "/multi.safetensors"
  let vec1d = Tensor(data: [1.0, 2.0, 3.0, 4.0], shape: [4])
  let mat3x3 =
    Tensor(data: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], shape: [3, 3])
  let cube =
    Tensor(data: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], shape: [2, 2, 2])
  let tensors = dict.from_list([#("a", vec1d), #("b", mat3x3), #("c", cube)])

  let assert Ok(Nil) = safetensors.write(path, tensors)
  let assert Ok(loaded) = safetensors.read(path)

  let assert Ok(out_a) = dict.get(loaded, "a")
  t.shape(out_a) |> should.equal([4])
  t.to_list(out_a) |> should.equal([1.0, 2.0, 3.0, 4.0])

  let assert Ok(out_b) = dict.get(loaded, "b")
  t.shape(out_b) |> should.equal([3, 3])
  t.to_list(out_b)
  |> should.equal([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])

  let assert Ok(out_c) = dict.get(loaded, "c")
  t.shape(out_c) |> should.equal([2, 2, 2])
  t.to_list(out_c)
  |> should.equal([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])

  cleanup(path)
}

pub fn safetensors_metadata_test() {
  ensure_tmp()
  let path = tmp_dir <> "/with_meta.safetensors"
  let t1 = Tensor(data: [1.0, 2.0], shape: [2])
  let tensors = dict.from_list([#("x", t1)])
  let metadata =
    dict.from_list([
      #("framework", "viva_tensor"),
      #("version", "1.0"),
    ])

  let assert Ok(Nil) = safetensors.write_with_metadata(path, tensors, metadata)
  let assert Ok(loaded_meta) = safetensors.metadata_of(path)

  dict.get(loaded_meta, "framework") |> should.equal(Ok("viva_tensor"))
  dict.get(loaded_meta, "version") |> should.equal(Ok("1.0"))

  // Tensors should still load correctly with metadata present.
  let assert Ok(loaded) = safetensors.read(path)
  let assert Ok(x) = dict.get(loaded, "x")
  t.to_list(x) |> should.equal([1.0, 2.0])

  cleanup(path)
}

pub fn safetensors_bad_header_test() {
  ensure_tmp()
  let path = tmp_dir <> "/bad_header.safetensors"
  // Header length claims 1000 bytes, but file only has a few bytes of header.
  let bad_bytes = <<1000:little-size(64), "garbage":utf8>>
  let assert Ok(Nil) = simplifile.write_bits(to: path, bits: bad_bytes)

  let result = safetensors.read(path)
  result |> should.be_error()

  cleanup(path)
}

pub fn safetensors_unsupported_dtype_test() {
  ensure_tmp()
  let path = tmp_dir <> "/bad_dtype.safetensors"
  let header =
    "{\"x\":{\"dtype\":\"I16\",\"shape\":[2],\"data_offsets\":[0,4]}}"
  let header_bytes = <<header:utf8>>
  let header_len = bit_array.byte_size(header_bytes)
  let payload = <<0:little-size(16), 0:little-size(16)>>
  let blob = <<
    header_len:little-size(64),
    header_bytes:bits,
    payload:bits,
  >>
  let assert Ok(Nil) = simplifile.write_bits(to: path, bits: blob)

  let result = safetensors.read(path)
  result |> should.be_error()

  cleanup(path)
}
