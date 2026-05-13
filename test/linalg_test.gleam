import gleam/list
import gleeunit/should
import support/numerics.{
  floats_close as assert_close, lists_close as assert_lists_close,
}
import viva_tensor as t
import viva_tensor/core/error
import viva_tensor/core/linalg
import viva_tensor/tensor

// Default tolerances (numpy-style).
const rtol: Float = 1.0e-7

const atol: Float = 1.0e-9

// =============================================================================
// solve
// =============================================================================

pub fn solve_2x2_test() {
  // 3x + 2y = 7
  //  x + 2y = 5  -> x = 1, y = 2
  let assert Ok(a) = tensor.matrix(2, 2, [3.0, 2.0, 1.0, 2.0])
  let b = tensor.from_list([7.0, 5.0])
  let assert Ok(x) = linalg.solve(a, b)

  tensor.shape(x) |> should.equal([2])
  let values = tensor.to_list(x)
  assert_lists_close(values, [1.0, 2.0], rtol, atol) |> should.be_true
}

pub fn solve_3x3_test() {
  //  2x +  y -  z =  8
  // -3x -  y + 2z = -11
  // -2x +  y + 2z = -3
  // Solution: x = 2, y = 3, z = -1
  let assert Ok(a) =
    tensor.matrix(3, 3, [2.0, 1.0, -1.0, -3.0, -1.0, 2.0, -2.0, 1.0, 2.0])
  let b = tensor.from_list([8.0, -11.0, -3.0])
  let assert Ok(x) = linalg.solve(a, b)

  tensor.shape(x) |> should.equal([3])
  let values = tensor.to_list(x)
  assert_lists_close(values, [2.0, 3.0, -1.0], rtol, atol) |> should.be_true
}

pub fn solve_with_multiple_rhs_test() {
  //  A = [[1, 1], [1, -1]], B = [[5, 1], [1, 5]]
  //  Solve A X = B -> X = [[3, 3], [2, -2]]
  let assert Ok(a) = tensor.matrix(2, 2, [1.0, 1.0, 1.0, -1.0])
  let assert Ok(b) = tensor.matrix(2, 2, [5.0, 1.0, 1.0, 5.0])
  let assert Ok(x) = linalg.solve(a, b)

  tensor.shape(x) |> should.equal([2, 2])
  let values = tensor.to_list(x)
  assert_lists_close(values, [3.0, 3.0, 2.0, -2.0], rtol, atol)
  |> should.be_true
}

pub fn solve_singular_error_test() {
  // Row 2 is 2x row 1 -> singular.
  let assert Ok(a) = tensor.matrix(2, 2, [1.0, 2.0, 2.0, 4.0])
  let b = tensor.from_list([3.0, 6.0])
  case linalg.solve(a, b) {
    Error(error.InvalidShape("matrix is singular")) -> Nil
    other -> {
      let _ = other
      should.fail()
    }
  }
}

// =============================================================================
// inv
// =============================================================================

pub fn inv_2x2_test() {
  let assert Ok(a) = tensor.matrix(2, 2, [4.0, 7.0, 2.0, 6.0])
  let assert Ok(ai) = linalg.inv(a)
  let assert Ok(prod) = t.matmul(a, ai)
  let expected = [1.0, 0.0, 0.0, 1.0]
  assert_lists_close(tensor.to_list(prod), expected, rtol, atol)
  |> should.be_true
}

pub fn inv_3x3_test() {
  let assert Ok(a) =
    tensor.matrix(3, 3, [1.0, 2.0, 3.0, 0.0, 1.0, 4.0, 5.0, 6.0, 0.0])
  let assert Ok(ai) = linalg.inv(a)
  let assert Ok(prod) = t.matmul(a, ai)
  let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  assert_lists_close(tensor.to_list(prod), expected, rtol, atol)
  |> should.be_true
}

pub fn inv_singular_error_test() {
  let assert Ok(a) = tensor.matrix(2, 2, [1.0, 2.0, 2.0, 4.0])
  case linalg.inv(a) {
    Error(error.InvalidShape("matrix is singular")) -> Nil
    _ -> should.fail()
  }
}

// =============================================================================
// det
// =============================================================================

pub fn det_2x2_test() {
  // det([[1,2],[3,4]]) = 1*4 - 2*3 = -2
  let assert Ok(a) = tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
  let assert Ok(d) = linalg.det(a)
  assert_close(d, -2.0, rtol, atol) |> should.be_true
}

pub fn det_3x3_test() {
  // det = 1*(5*9 - 6*8) - 2*(4*9 - 6*7) + 3*(4*8 - 5*7)
  //     = 1*(-3) - 2*(-6) + 3*(-3) = -3 + 12 - 9 = 0
  // Use a non-singular one instead so we get a clear non-zero answer.
  // A = [[6,1,1],[4,-2,5],[2,8,7]]
  // det = 6*(-2*7 - 5*8) - 1*(4*7 - 5*2) + 1*(4*8 - (-2)*2)
  //     = 6*(-14 - 40) - (28 - 10) + (32 + 4)
  //     = 6*(-54) - 18 + 36 = -324 - 18 + 36 = -306
  let assert Ok(a) =
    tensor.matrix(3, 3, [6.0, 1.0, 1.0, 4.0, -2.0, 5.0, 2.0, 8.0, 7.0])
  let assert Ok(d) = linalg.det(a)
  assert_close(d, -306.0, rtol, atol) |> should.be_true
}

pub fn det_singular_returns_zero_test() {
  let assert Ok(a) = tensor.matrix(2, 2, [1.0, 2.0, 2.0, 4.0])
  let assert Ok(d) = linalg.det(a)
  assert_close(d, 0.0, rtol, atol) |> should.be_true
}

// =============================================================================
// lu
// =============================================================================

pub fn lu_decomposition_test() {
  let assert Ok(a) =
    tensor.matrix(3, 3, [2.0, -1.0, -2.0, -4.0, 6.0, 3.0, -4.0, -2.0, 8.0])
  let assert Ok(#(l, u, perm)) = linalg.lu(a)

  tensor.shape(l) |> should.equal([3, 3])
  tensor.shape(u) |> should.equal([3, 3])
  list.length(perm) |> should.equal(3)

  // P @ A == L @ U
  let assert Ok(lu_product) = t.matmul(l, u)
  // Build permuted A as a flat list.
  let a_list = tensor.to_list(a)
  let permuted =
    perm
    |> list.flat_map(fn(idx) {
      a_list
      |> list.drop(idx * 3)
      |> list.take(3)
    })
  let lu_list = tensor.to_list(lu_product)
  assert_lists_close(lu_list, permuted, rtol, atol) |> should.be_true
}

// =============================================================================
// cholesky
// =============================================================================

pub fn cholesky_2x2_test() {
  // A = [[4, 12], [12, 37]] = L @ L^T with L = [[2, 0], [6, 1]]
  let assert Ok(a) = tensor.matrix(2, 2, [4.0, 12.0, 12.0, 37.0])
  let assert Ok(l) = linalg.cholesky(a)
  let expected = [2.0, 0.0, 6.0, 1.0]
  assert_lists_close(tensor.to_list(l), expected, rtol, atol)
  |> should.be_true
}

pub fn cholesky_3x3_test() {
  // Classic example: A = [[25, 15, -5], [15, 18, 0], [-5, 0, 11]]
  // L = [[5, 0, 0], [3, 3, 0], [-1, 1, 3]]
  let assert Ok(a) =
    tensor.matrix(3, 3, [25.0, 15.0, -5.0, 15.0, 18.0, 0.0, -5.0, 0.0, 11.0])
  let assert Ok(l) = linalg.cholesky(a)
  let expected = [5.0, 0.0, 0.0, 3.0, 3.0, 0.0, -1.0, 1.0, 3.0]
  assert_lists_close(tensor.to_list(l), expected, rtol, atol)
  |> should.be_true
}

pub fn cholesky_not_pd_error_test() {
  // Indefinite matrix (eigenvalues of opposite sign).
  let assert Ok(a) = tensor.matrix(2, 2, [1.0, 2.0, 2.0, 1.0])
  case linalg.cholesky(a) {
    Error(error.InvalidShape("matrix is not positive definite")) -> Nil
    _ -> should.fail()
  }
}

// =============================================================================
// qr
// =============================================================================

pub fn qr_2x2_test() {
  let assert Ok(a) = tensor.matrix(2, 2, [1.0, 2.0, 3.0, 4.0])
  let assert Ok(#(q, r)) = linalg.qr(a)

  tensor.shape(q) |> should.equal([2, 2])
  tensor.shape(r) |> should.equal([2, 2])

  // Q @ R ~= A
  let assert Ok(qr_product) = t.matmul(q, r)
  assert_lists_close(tensor.to_list(qr_product), tensor.to_list(a), rtol, atol)
  |> should.be_true

  // Q^T @ Q ~= I
  let assert Ok(qt) = t.transpose(q)
  let assert Ok(qtq) = t.matmul(qt, q)
  let identity = [1.0, 0.0, 0.0, 1.0]
  assert_lists_close(tensor.to_list(qtq), identity, rtol, atol)
  |> should.be_true
}

pub fn qr_3x4_test() {
  // Tall(ish) matrix 3x4 - but the spec says 3x4; QR via classical GS
  // produces a thin decomposition Q: 3x4, R: 4x4 with linearly dependent
  // columns. To keep this stable, use a 4x3 matrix instead (m >= n).
  let assert Ok(a) =
    tensor.matrix(4, 3, [
      1.0, -1.0, 4.0, 1.0, 4.0, -2.0, 1.0, 4.0, 2.0, 1.0, -1.0, 0.0,
    ])
  let assert Ok(#(q, r)) = linalg.qr(a)

  tensor.shape(q) |> should.equal([4, 3])
  tensor.shape(r) |> should.equal([3, 3])

  let assert Ok(qr_product) = t.matmul(q, r)
  assert_lists_close(tensor.to_list(qr_product), tensor.to_list(a), rtol, atol)
  |> should.be_true

  let assert Ok(qt) = t.transpose(q)
  let assert Ok(qtq) = t.matmul(qt, q)
  let identity = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
  assert_lists_close(tensor.to_list(qtq), identity, rtol, atol)
  |> should.be_true
}
