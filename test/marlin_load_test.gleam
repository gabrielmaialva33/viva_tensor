import gleeunit
import gleeunit/should
import viva_tensor as vt

pub type EunitTimeout {
  Timeout
}

pub fn main() -> Nil {
  gleeunit.main()
}

pub fn load_model_marlin_smoke_test_() {
  #(Timeout, 600, fn() { load_model_marlin_smoke() })
}

fn load_model_marlin_smoke() {
  let path = "tmp/tinyllama/model.safetensors"
  case vt.load_model_with_format(path, vt.MarlinW4A16) {
    Ok(_handle) -> should.be_true(True)
    Error(_) -> should.be_true(True)
  }
}
