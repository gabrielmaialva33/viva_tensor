import gleam/float
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor as t

pub fn main() {
  gleeunit.main()
}

fn close_enough(a: Float, b: Float) -> Bool {
  float.absolute_value(a -. b) <. 0.000001
}

pub fn hardware_profiles_include_unavailable_rubin_targets_test() {
  let profiles = t.hardware_profiles()

  let assert Ok(rubin) =
    list.find(profiles, fn(profile) { profile.name == "Rubin R100" })

  rubin.available |> should.be_false()
  rubin.memory_bandwidth_gbps |> should.equal(22_000)
  rubin.preferred_micro_block |> should.equal(16)

  let assert Ok(vera) =
    list.find(profiles, fn(profile) { profile.name == "Vera CPU" })

  vera.available |> should.be_false()
  vera.nvlink_c2c_gbps |> should.equal(1800)
}

pub fn nvfp4_layout_uses_rubin_micro_blocks_test() {
  let layout = t.nvfp4_block_scaled_layout([2, 16])

  layout.storage_bits_per_value |> should.equal(4)
  layout.native_micro_block |> should.equal(16)
  t.quant_layout_memory_bytes(layout) |> should.equal(16)
  t.quant_layout_compression_ratio_against(layout, 16) |> should.equal(4.0)
  t.quant_layout_is_rubin_native_candidate(layout) |> should.be_true()
}

pub fn int2_progressive_layout_requires_hadamard_test() {
  case t.int2_progressive_layout([13], 16) {
    Ok(layout) -> {
      layout.storage_bits_per_value |> should.equal(2)
      layout.requires_hadamard |> should.be_true()
      t.quant_layout_memory_bytes(layout) |> should.equal(4)
      t.quant_layout_compression_ratio_against(layout, 16) |> should.equal(8.0)
      t.quant_layout_is_rubin_native_candidate(layout) |> should.be_true()
    }
    Error(_) -> should.fail()
  }

  case t.int2_progressive_layout([13], 0) {
    Ok(_) -> should.fail()
    Error(_) -> should.be_true(True)
  }
}

pub fn normalized_walsh_hadamard_matches_known_vector_test() {
  case t.try_normalized_walsh_hadamard([1.0, 2.0, 3.0, 4.0]) {
    Ok(values) -> values |> should.equal([5.0, -1.0, -2.0, 0.0])
    Error(_) -> should.fail()
  }
}

pub fn hadamard_preprocess_is_reversible_test() {
  let input = t.from_list([1.0, 2.0, 3.0])

  case t.try_hadamard_preprocess(input, 42) {
    Ok(preprocessed) -> {
      preprocessed.original_dim |> should.equal(3)
      preprocessed.padded_dim |> should.equal(4)
      t.shape(preprocessed.tensor) |> should.equal([4])

      case t.try_inverse_hadamard_preprocess(preprocessed) {
        Ok(restored) -> {
          let values = t.to_list(restored)
          let assert [a, b, c] = values
          close_enough(a, 1.0) |> should.be_true()
          close_enough(b, 2.0) |> should.be_true()
          close_enough(c, 3.0) |> should.be_true()
        }
        Error(_) -> should.fail()
      }
    }
    Error(_) -> should.fail()
  }
}
