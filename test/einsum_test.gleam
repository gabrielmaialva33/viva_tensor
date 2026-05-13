import gleeunit
import gleeunit/should
import viva_tensor as t
import viva_tensor/tensor as core_tensor

pub fn main() -> Nil {
  gleeunit.main()
}

// --- Helpers ----------------------------------------------------------------

fn mat2x3() -> core_tensor.Tensor {
  let assert Ok(m) = t.from_list2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  m
}

fn mat3x3() -> core_tensor.Tensor {
  let assert Ok(m) =
    t.from_list2d([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
  m
}

fn mat3x4() -> core_tensor.Tensor {
  let assert Ok(m) =
    t.from_list2d([
      [1.0, 2.0, 3.0, 4.0],
      [5.0, 6.0, 7.0, 8.0],
      [9.0, 10.0, 11.0, 12.0],
    ])
  m
}

fn mat2x2_a() -> core_tensor.Tensor {
  let assert Ok(m) = t.from_list2d([[1.0, 2.0], [3.0, 4.0]])
  m
}

fn mat2x2_b() -> core_tensor.Tensor {
  let assert Ok(m) = t.from_list2d([[5.0, 6.0], [7.0, 8.0]])
  m
}

// =============================================================================
// TESTS
// =============================================================================

pub fn einsum_transpose_test() {
  let a = mat2x3()
  let assert Ok(r) = t.einsum("ij->ji", [a])
  t.shape(r) |> should.equal([3, 2])
  t.to_list(r) |> should.equal([1.0, 4.0, 2.0, 5.0, 3.0, 6.0])
}

pub fn einsum_trace_test() {
  let a = mat3x3()
  let assert Ok(r) = t.einsum("ii->", [a])
  t.shape(r) |> should.equal([])
  // diag = 1 + 5 + 9 = 15
  t.to_list(r) |> should.equal([15.0])
}

pub fn einsum_matmul_test() {
  let a = mat2x3()
  let b = mat3x4()
  let assert Ok(r) = t.einsum("ij,jk->ik", [a, b])
  t.shape(r) |> should.equal([2, 4])
  // a @ b (row 0): [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12]
  //              = [38, 44, 50, 56]
  // row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12]
  //      = [83, 98, 113, 128]
  t.to_list(r)
  |> should.equal([38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0])
}

pub fn einsum_inner_test() {
  let a = t.from_list([1.0, 2.0, 3.0, 4.0])
  let b = t.from_list([5.0, 6.0, 7.0, 8.0])
  let assert Ok(r) = t.einsum("i,i->", [a, b])
  t.shape(r) |> should.equal([])
  // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
  t.to_list(r) |> should.equal([70.0])
}

pub fn einsum_outer_test() {
  let a = t.from_list([1.0, 2.0, 3.0])
  let b = t.from_list([4.0, 5.0, 6.0, 7.0])
  let assert Ok(r) = t.einsum("i,j->ij", [a, b])
  t.shape(r) |> should.equal([3, 4])
  t.to_list(r)
  |> should.equal([
    4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 12.0, 15.0, 18.0, 21.0,
  ])
}

pub fn einsum_elementwise_mul_test() {
  let a = mat2x2_a()
  let b = mat2x2_b()
  let assert Ok(r) = t.einsum("ij,ij->ij", [a, b])
  t.shape(r) |> should.equal([2, 2])
  // [[1*5, 2*6], [3*7, 4*8]] = [[5, 12], [21, 32]]
  t.to_list(r) |> should.equal([5.0, 12.0, 21.0, 32.0])
}

pub fn einsum_frobenius_test() {
  let a = mat2x2_a()
  let b = mat2x2_b()
  let assert Ok(r) = t.einsum("ij,ij->", [a, b])
  t.shape(r) |> should.equal([])
  // 5 + 12 + 21 + 32 = 70
  t.to_list(r) |> should.equal([70.0])
}

pub fn einsum_row_sum_test() {
  let a = mat2x3()
  let assert Ok(r) = t.einsum("ij->i", [a])
  t.shape(r) |> should.equal([2])
  // rows: 1+2+3=6, 4+5+6=15
  t.to_list(r) |> should.equal([6.0, 15.0])
}

pub fn einsum_col_sum_test() {
  let a = mat2x3()
  let assert Ok(r) = t.einsum("ij->j", [a])
  t.shape(r) |> should.equal([3])
  // cols: 1+4=5, 2+5=7, 3+6=9
  t.to_list(r) |> should.equal([5.0, 7.0, 9.0])
}

pub fn einsum_total_sum_test() {
  let a = mat2x3()
  let assert Ok(r) = t.einsum("ij->", [a])
  t.shape(r) |> should.equal([])
  // 1+2+3+4+5+6 = 21
  t.to_list(r) |> should.equal([21.0])
}

pub fn einsum_invalid_equation_test() {
  let a = mat2x3()
  // Implicit output mode is rejected in v1.
  case t.einsum("ij", [a]) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

pub fn einsum_arity_mismatch_test() {
  let a = mat2x3()
  // Equation needs two operands but we pass one.
  case t.einsum("ij,jk->ik", [a]) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

pub fn einsum_unknown_rhs_label_test() {
  // 'k' on RHS but not in any LHS.
  let a = mat2x3()
  case t.einsum("ij->k", [a]) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}

pub fn einsum_ellipsis_rejected_test() {
  let a = mat2x3()
  case t.einsum("...ij->...ji", [a]) {
    Ok(_) -> should.fail()
    Error(_) -> Nil
  }
}
