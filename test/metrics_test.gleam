import gleam/float
import gleeunit
import gleeunit/should
import viva_tensor/observability/metrics
import viva_tensor/tensor

pub fn main() -> Nil {
  gleeunit.main()
}

fn assert_close(actual: Float, expected: Float, tolerance: Float) {
  { float.absolute_value(actual -. expected) <=. tolerance }
  |> should.be_true()
}

pub fn try_basic_metrics_test() {
  let original = tensor.from_list([1.0, 2.0])
  let quantized = tensor.from_list([2.0, 4.0])

  metrics.try_mse(original, quantized) |> should.equal(Ok(2.5))
  metrics.try_mae(original, quantized) |> should.equal(Ok(1.5))
  metrics.try_max_error(original, quantized) |> should.equal(Ok(2.0))
}

pub fn try_rmse_test() {
  let original = tensor.from_list([1.0, 2.0])
  let quantized = tensor.from_list([2.0, 4.0])

  case metrics.try_rmse(original, quantized) {
    Ok(value) -> assert_close(value, 1.5811388300841898, 0.0000001)
    Error(_) -> should.fail()
  }
}

pub fn try_cosine_similarity_test() {
  let original = tensor.from_list([1.0, 2.0, 3.0])

  case metrics.try_cosine_similarity(original, original) {
    Ok(value) -> assert_close(value, 1.0, 0.0000001)
    Error(_) -> should.fail()
  }
}

pub fn try_snr_db_identical_tensors_is_capped_test() {
  let original = tensor.from_list([1.0, 2.0, 3.0])

  metrics.try_snr_db(original, original) |> should.equal(Ok(100.0))
}

pub fn try_error_percentile_test() {
  let original = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  let quantized = tensor.from_list([1.0, 3.0, 5.0, 7.0])

  case metrics.try_error_percentile(original, quantized, 50.0) {
    Ok(value) -> assert_close(value, 1.5, 0.0000001)
    Error(_) -> should.fail()
  }
}

pub fn try_outlier_percentage_test() {
  let original = tensor.from_list([1.0, 2.0, 3.0, 4.0])
  let quantized = tensor.from_list([1.0, 2.1, 3.2, 4.3])

  case metrics.try_outlier_percentage(original, quantized, 0.15) {
    Ok(value) -> assert_close(value, 50.0, 0.0000001)
    Error(_) -> should.fail()
  }
}

pub fn try_compute_all_test() {
  let original = tensor.from_list([1.0, 2.0, 3.0])
  let quantized = tensor.from_list([1.0, 2.0, 4.0])

  case metrics.try_compute_all(original, quantized) {
    Ok(all) -> {
      assert_close(all.mse, 0.3333333333333333, 0.0000001)
      assert_close(all.mae, 0.3333333333333333, 0.0000001)
      all.max_error |> should.equal(1.0)
    }
    Error(_) -> should.fail()
  }
}

pub fn try_metrics_reject_shape_mismatch_test() {
  let original = tensor.Tensor(data: [1.0, 2.0], shape: [2])
  let quantized = tensor.Tensor(data: [1.0, 2.0], shape: [1, 2])

  metrics.try_mse(original, quantized) |> should.be_error()
  metrics.try_compute_all(original, quantized) |> should.be_error()
}

pub fn try_metrics_reject_empty_tensors_test() {
  let empty = tensor.from_list([])

  metrics.try_mae(empty, empty) |> should.be_error()
  metrics.try_error_percentile(empty, empty, 99.0) |> should.be_error()
}

pub fn try_error_percentile_rejects_invalid_percentile_test() {
  let original = tensor.from_list([1.0])
  let quantized = tensor.from_list([1.0])

  metrics.try_error_percentile(original, quantized, -1.0) |> should.be_error()
  metrics.try_error_percentile(original, quantized, 101.0) |> should.be_error()
}
