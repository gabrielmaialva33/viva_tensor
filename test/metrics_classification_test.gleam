import gleeunit
import gleeunit/should
import support/numerics
import viva_tensor as t
import viva_tensor/core/error
import viva_tensor/metrics/classification.{Macro, Micro, Weighted}

pub fn main() -> Nil {
  gleeunit.main()
}

// --- accuracy ---------------------------------------------------------------

pub fn accuracy_test() {
  let preds = t.from_list([0.0, 1.0, 2.0, 1.0])
  let targets = t.from_list([0.0, 1.0, 1.0, 1.0])
  case t.accuracy(preds, targets) {
    Ok(value) -> {
      numerics.floats_close(value, 0.75, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn accuracy_shape_error_test() {
  let preds = t.from_list([0.0, 1.0, 2.0])
  let targets = t.from_list([0.0, 1.0])
  case t.accuracy(preds, targets) {
    Error(error.ShapeMismatch(_, _)) -> Nil
    _ -> should.fail()
  }
}

// --- confusion matrix -------------------------------------------------------

pub fn confusion_matrix_test() {
  // targets: [0, 1, 2, 1] preds: [0, 1, 1, 2]
  // cm[true,pred]:
  //   row 0: [1, 0, 0]
  //   row 1: [0, 1, 1]
  //   row 2: [0, 1, 0]
  let preds = t.from_list([0.0, 1.0, 1.0, 2.0])
  let targets = t.from_list([0.0, 1.0, 2.0, 1.0])
  case t.confusion_matrix(preds, targets, 3) {
    Ok(cm) -> {
      t.shape(cm) |> should.equal([3, 3])
      t.to_list(cm)
      |> should.equal([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    }
    Error(_) -> should.fail()
  }
}

pub fn confusion_matrix_oob_test() {
  let preds = t.from_list([0.0, 1.0, 2.0])
  let targets = t.from_list([0.0, 1.0, 5.0])
  case t.confusion_matrix(preds, targets, 3) {
    Error(error.IndexOutOfBounds(_, _)) -> Nil
    _ -> should.fail()
  }
}

// --- precision --------------------------------------------------------------

// Shared fixture for precision/recall/f1 tests:
//   classes = 3
//   preds   = [0, 1, 2, 1, 0, 2]
//   targets = [0, 1, 1, 2, 0, 2]
//   cm[true,pred]:
//     class 0: [2, 0, 0]      (TP=2, FN=0)
//     class 1: [0, 1, 1]      (TP=1, FN=1)
//     class 2: [0, 1, 1]      (TP=1, FN=1)
//   column sums (predicted as c):
//     0 -> 2  (FP=0)
//     1 -> 2  (FP=1)
//     2 -> 2  (FP=1)
//   support: [2, 2, 2]
//   precision per class: 2/2, 1/2, 1/2 = [1.0, 0.5, 0.5]
//   recall per class:    2/2, 1/2, 1/2 = [1.0, 0.5, 0.5]
//   macro precision = (1.0 + 0.5 + 0.5)/3 = 0.6666...
//   weighted precision = (1.0*2 + 0.5*2 + 0.5*2)/6 = 0.6666...
//   micro precision = sum_tp / (sum_tp + sum_fp) = 4 / (4 + 2) = 0.6666...

fn make_pred_target() -> #(t.Tensor, t.Tensor) {
  #(
    t.from_list([0.0, 1.0, 2.0, 1.0, 0.0, 2.0]),
    t.from_list([0.0, 1.0, 1.0, 2.0, 0.0, 2.0]),
  )
}

pub fn precision_macro_test() {
  let #(preds, targets) = make_pred_target()
  case t.precision(preds, targets, 3, Macro) {
    Ok(value) -> {
      numerics.floats_close(value, 2.0 /. 3.0, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn precision_micro_test() {
  let #(preds, targets) = make_pred_target()
  case t.precision(preds, targets, 3, Micro) {
    Ok(value) -> {
      numerics.floats_close(value, 2.0 /. 3.0, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

pub fn precision_weighted_test() {
  let #(preds, targets) = make_pred_target()
  case t.precision(preds, targets, 3, Weighted) {
    Ok(value) -> {
      numerics.floats_close(value, 2.0 /. 3.0, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

// --- recall -----------------------------------------------------------------

pub fn recall_macro_test() {
  let #(preds, targets) = make_pred_target()
  case t.recall(preds, targets, 3, Macro) {
    Ok(value) -> {
      numerics.floats_close(value, 2.0 /. 3.0, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

// --- f1 ---------------------------------------------------------------------

pub fn f1_test() {
  let #(preds, targets) = make_pred_target()
  // macro F1 = mean of [2*1*1/(1+1), 2*0.5*0.5/(0.5+0.5), 2*0.5*0.5/(0.5+0.5)]
  //         = mean of [1.0, 0.5, 0.5] = 2/3
  case t.f1(preds, targets, 3, Macro) {
    Ok(value) -> {
      numerics.floats_close(value, 2.0 /. 3.0, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

// --- top-k accuracy ---------------------------------------------------------

pub fn top_k_accuracy_test() {
  // Batch 3, 3 classes. logits ordered so:
  //   row 0 top-2: indices [2, 1] -> target 2 hit
  //   row 1 top-2: indices [0, 1] -> target 1 hit
  //   row 2 top-2: indices [1, 0] -> target 2 miss
  let logits_data = [0.1, 0.2, 0.7, 0.9, 0.05, 0.05, 0.4, 0.5, 0.1]
  let logits_2d = case t.matrix(3, 3, logits_data) {
    Ok(m) -> m
    Error(_) -> t.zeros([3, 3])
  }
  let targets = t.from_list([2.0, 1.0, 2.0])
  case t.top_k_accuracy(logits_2d, targets, 2) {
    Ok(value) -> {
      numerics.floats_close(value, 2.0 /. 3.0, 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}

// --- IoU --------------------------------------------------------------------

pub fn iou_per_class_test() {
  // Using same fixture as precision/recall:
  //   class 0: TP=2 FP=0 FN=0 -> IoU = 2/2 = 1.0
  //   class 1: TP=1 FP=1 FN=1 -> IoU = 1/3
  //   class 2: TP=1 FP=1 FN=1 -> IoU = 1/3
  let #(preds, targets) = make_pred_target()
  case t.iou_per_class(preds, targets, 3) {
    Ok(values) -> {
      numerics.lists_close(values, [1.0, 1.0 /. 3.0, 1.0 /. 3.0], 1.0e-9, 0.0)
      |> should.be_true()
    }
    Error(_) -> should.fail()
  }
}
