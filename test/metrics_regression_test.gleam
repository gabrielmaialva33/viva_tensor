import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor as t
import viva_tensor/core/error

pub fn main() -> Nil {
  gleeunit.main()
}

pub fn mae_test() {
  let preds = t.from_list([1.0, 2.0, 3.0])
  let targets = t.from_list([1.5, 1.5, 2.5])
  // |0.5| + |0.5| + |0.5| = 1.5, /3 = 0.5
  case t.mean_absolute_error(preds, targets) {
    Ok(value) -> {
      numerics.floats_close(value, 0.5, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn mse_test() {
  let preds = t.from_list([1.0, 2.0, 3.0])
  let targets = t.from_list([1.5, 1.5, 2.5])
  // 0.25*3 / 3 = 0.25
  case t.mean_squared_error(preds, targets) {
    Ok(value) -> {
      numerics.floats_close(value, 0.25, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn rmse_test() {
  let preds = t.from_list([1.0, 2.0, 3.0])
  let targets = t.from_list([1.5, 1.5, 2.5])
  // sqrt(0.25) = 0.5
  case t.root_mean_squared_error(preds, targets) {
    Ok(value) -> {
      numerics.floats_close(value, 0.5, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn r_squared_perfect_prediction_test() {
  let preds = t.from_list([1.0, 2.0, 3.0, 4.0])
  let targets = t.from_list([1.0, 2.0, 3.0, 4.0])
  case t.r_squared(preds, targets) {
    Ok(value) -> {
      numerics.floats_close(value, 1.0, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn r_squared_mean_prediction_test() {
  // Predict the mean for every sample -> SS_res == SS_tot -> R^2 == 0.
  let targets = t.from_list([1.0, 2.0, 3.0, 4.0])
  // mean = 2.5
  let preds = t.from_list([2.5, 2.5, 2.5, 2.5])
  case t.r_squared(preds, targets) {
    Ok(value) -> {
      numerics.floats_close(value, 0.0, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn mape_test() {
  let preds = t.from_list([100.0, 200.0])
  let targets = t.from_list([110.0, 180.0])
  // (|100-110|/110 + |200-180|/180) / 2 * 100
  // = (10/110 + 20/180) / 2 * 100
  // = (0.0909090... + 0.1111111...) / 2 * 100
  // = 0.10101010... * 100 = 10.1010101...
  case t.mean_absolute_percentage_error(preds, targets) {
    Ok(value) -> {
      let expected = { 10.0 /. 110.0 +. 20.0 /. 180.0 } /. 2.0 *. 100.0
      numerics.floats_close(value, expected, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn mape_zero_target_error_test() {
  let preds = t.from_list([1.0, 2.0])
  let targets = t.from_list([0.0, 1.0])
  case t.mean_absolute_percentage_error(preds, targets) {
    Error(error.InvalidShape(reason)) ->
      should.equal(reason, "MAPE: target contains zero values")
    _ -> should.fail()
  }
}
