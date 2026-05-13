//// Tests for NumPy-style fancy and boolean indexing.
////
//// Note: eunit auto-discovers functions whose names end in `_test`, so
//// the test spec names like `take_test_1d` are rendered as `take_1d_test`
//// to keep them runnable while preserving the original intent.

import gleeunit
import gleeunit/should
import viva_tensor as t
import viva_tensor/core/error

pub fn main() -> Nil {
  gleeunit.main()
}

// =============================================================================
// take
// =============================================================================

pub fn take_1d_test() {
  let v = t.from_list([10.0, 20.0, 30.0, 40.0])
  case t.take(v, [2, 0, 3], 0) {
    Ok(out) -> {
      t.to_list(out) |> should.equal([30.0, 10.0, 40.0])
      t.shape(out) |> should.equal([3])
    }
    Error(_) -> should.fail()
  }
}

pub fn take_2d_axis0_test() {
  let assert Ok(m) = t.matrix(3, 2, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  case t.take(m, [2, 0], 0) {
    Ok(out) -> {
      t.shape(out) |> should.equal([2, 2])
      t.to_list(out) |> should.equal([5.0, 6.0, 1.0, 2.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn take_2d_axis1_test() {
  let assert Ok(m) = t.matrix(2, 3, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
  case t.take(m, [2, 0], 1) {
    Ok(out) -> {
      t.shape(out) |> should.equal([2, 2])
      t.to_list(out) |> should.equal([3.0, 1.0, 6.0, 4.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn take_negative_indices_test() {
  let v = t.from_list([10.0, 20.0, 30.0, 40.0])
  case t.take(v, [-1, -2], 0) {
    Ok(out) -> {
      t.to_list(out) |> should.equal([40.0, 30.0])
      t.shape(out) |> should.equal([2])
    }
    Error(_) -> should.fail()
  }
}

pub fn take_bad_axis_test() {
  let v = t.from_list([10.0, 20.0, 30.0])
  case t.take(v, [0], 5) {
    Ok(_) -> should.fail()
    Error(error.DimensionError(_)) -> should.be_true(True)
    Error(_) -> should.fail()
  }
}

pub fn take_index_out_of_bounds_test() {
  let v = t.from_list([10.0, 20.0, 30.0])
  case t.take(v, [5], 0) {
    Ok(_) -> should.fail()
    Error(error.IndexOutOfBounds(5, 3)) -> should.be_true(True)
    Error(_) -> should.fail()
  }
}

// =============================================================================
// gather
// =============================================================================

pub fn gather_test() {
  let v = t.from_list([10.0, 20.0, 30.0, 40.0])
  let idx = t.from_list([3.0, 1.0, 0.0])
  case t.gather(v, idx) {
    Ok(out) -> {
      t.to_list(out) |> should.equal([40.0, 20.0, 10.0])
      t.shape(out) |> should.equal([3])
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// mask_select
// =============================================================================

pub fn mask_select_1d_test() {
  let v = t.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
  let mask = t.from_list([1.0, 0.0, 1.0, 0.0, 1.0])
  case t.mask_select(v, mask) {
    Ok(out) -> {
      t.to_list(out) |> should.equal([1.0, 3.0, 5.0])
      t.shape(out) |> should.equal([3])
    }
    Error(_) -> should.fail()
  }
}

pub fn mask_select_shape_mismatch_test() {
  let v = t.from_list([1.0, 2.0, 3.0])
  let bad_mask = t.from_list([1.0, 0.0])
  case t.mask_select(v, bad_mask) {
    Ok(_) -> should.fail()
    Error(error.ShapeMismatch(_, _)) -> should.be_true(True)
    Error(_) -> should.fail()
  }
}

// =============================================================================
// where
// =============================================================================

pub fn where_test() {
  let cond = t.from_list([1.0, 0.0, 1.0, 0.0])
  let a = t.from_list([10.0, 20.0, 30.0, 40.0])
  let b = t.from_list([-1.0, -2.0, -3.0, -4.0])
  case t.where(cond, a, b) {
    Ok(out) -> {
      t.to_list(out) |> should.equal([10.0, -2.0, 30.0, -4.0])
      t.shape(out) |> should.equal([4])
    }
    Error(_) -> should.fail()
  }
}

// =============================================================================
// nonzero
// =============================================================================

pub fn nonzero_1d_test() {
  let v = t.from_list([0.0, 1.0, 0.0, 3.0])
  case t.nonzero(v) {
    Ok(idx) -> idx |> should.equal([[1], [3]])
    Error(_) -> should.fail()
  }
}

pub fn nonzero_2d_test() {
  let assert Ok(m) = t.matrix(2, 3, [0.0, 1.0, 0.0, 2.0, 0.0, 3.0])
  case t.nonzero(m) {
    Ok(idx) -> idx |> should.equal([[0, 1], [1, 0], [1, 2]])
    Error(_) -> should.fail()
  }
}

pub fn nonzero_empty_test() {
  let v = t.from_list([0.0, 0.0, 0.0])
  case t.nonzero(v) {
    Ok(idx) -> idx |> should.equal([])
    Error(_) -> should.fail()
  }
}
