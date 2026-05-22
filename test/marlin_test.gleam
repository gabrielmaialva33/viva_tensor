import gleam/int
import gleam/list
import gleeunit
import gleeunit/should
import viva_tensor as vt

pub fn main() -> Nil {
  gleeunit.main()
}

pub fn prepack_marlin_w4a16_smoke_test() {
  let k = 128
  let n = 256
  let group = 128
  let groups = k / group

  let weight =
    int.range(from: 0, to: k * n, with: [], run: fn(acc, i) {
      [int.to_float(i) /. 1000.0, ..acc]
    })
    |> list.reverse
    |> vt.from_list

  let assert Ok(weight) = vt.reshape(weight, [k, n])

  let scales =
    int.range(from: 0, to: groups * n, with: [], run: fn(acc, _) {
      [0.05, ..acc]
    })
    |> list.reverse
    |> vt.from_list

  let assert Ok(scales) = vt.reshape(scales, [groups, n])

  case vt.prepack_marlin_w4a16(weight, scales, group) {
    Ok(_) -> should.be_true(True)
    Error(_) -> should.fail()
  }
}
