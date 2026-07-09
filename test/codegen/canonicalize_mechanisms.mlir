// RUN: daphne-opt --canonicalize %s | FileCheck %s

// Checks that each rewrite still fires after being moved to a different
// canonicalization mechanism. Folders, DRR patterns and C++ patterns all run
// inside --canonicalize, so one pass covers every case here.

// Fold: numRows(X) folds to an index constant when the operand's
// row extent is statically known. The unknown-extent case must survive.
// CHECK-LABEL: func.func @num_rows_known
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 3 : index}>
// CHECK-NEXT: "daphne.return"(%[[C]])
// CHECK-NOT: daphne.numRows
func.func @num_rows_known(%arg0: !daphne.Matrix<3x4xf64>) -> index {
    %0 = "daphne.numRows"(%arg0) : (!daphne.Matrix<3x4xf64>) -> index
    "daphne.return"(%0) : (index) -> ()
}

// Negative case: an unknown row extent leaves numRows in place.
// CHECK-LABEL: func.func @num_rows_unknown
// CHECK: daphne.numRows
func.func @num_rows_unknown(%arg0: !daphne.Matrix<?x4xf64>) -> index {
    %0 = "daphne.numRows"(%arg0) : (!daphne.Matrix<?x4xf64>) -> index
    "daphne.return"(%0) : (index) -> ()
}

// Rung 1 (Fold): sparsity(X) folds to an f64 constant when the operand's
// sparsity is statically known. The unknown-sparsity case must survive.
// CHECK-LABEL: func.func @sparsity_known
// CHECK: %[[C:.*]] = "daphne.constant"() <{value = 2.500000e-01 : f64}>
// CHECK-NEXT: "daphne.return"(%[[C]])
// CHECK-NOT: daphne.sparsity
func.func @sparsity_known(%arg0: !daphne.Matrix<8x8xf64:sp[2.500000e-01]>) -> f64 {
    %0 = "daphne.sparsity"(%arg0) : (!daphne.Matrix<8x8xf64:sp[2.500000e-01]>) -> f64
    "daphne.return"(%0) : (f64) -> ()
}

// Rung 1 negative case: an unknown sparsity leaves sparsity in place.
// CHECK-LABEL: func.func @sparsity_unknown
// CHECK: daphne.sparsity
func.func @sparsity_unknown(%arg0: !daphne.Matrix<8x8xf64>) -> f64 {
    %0 = "daphne.sparsity"(%arg0) : (!daphne.Matrix<8x8xf64>) -> f64
    "daphne.return"(%0) : (f64) -> ()
}
