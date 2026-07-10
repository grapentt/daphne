// RUN: daphne-opt --daphne-algebraic-simplify %s | FileCheck %s

// Locks in the linear-algebra simplification rewrites driven by
// --daphne-algebraic-simplify.

// Trace idiom: sum(diagVector(X @ Y)) with X @ Y square is rewritten to
// sum(X * t(Y)): the matMul and diagVector are gone, replaced by a transpose
// and an ewMul feeding the sumAll.
// CHECK-LABEL: func.func @trace_idiom
// CHECK-NOT: daphne.matMul
// CHECK-NOT: daphne.diagVector
// CHECK: daphne.transpose
// CHECK: daphne.ewMul
// CHECK: daphne.sumAll
func.func @trace_idiom(%x: !daphne.Matrix<3x4xf64>, %y: !daphne.Matrix<4x3xf64>) -> f64 {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %false) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x3xf64>, i1, i1) -> !daphne.Matrix<3x3xf64>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<3x3xf64>) -> !daphne.Matrix<3x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<3x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Negative case: unknown shape. Without static dimensions the rewrite cannot
// prove X @ Y is square, so it fails closed and the matMul survives.
// CHECK-LABEL: func.func @trace_idiom_unknown_shape
// CHECK: daphne.matMul
// CHECK: daphne.diagVector
func.func @trace_idiom_unknown_shape(%x: !daphne.Matrix<?x?xf64>, %y: !daphne.Matrix<?x?xf64>) -> f64 {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %false) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<?x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Negative case: a transposed matMul flag is a different chain the rewrite does
// not handle, so the matMul survives.
// CHECK-LABEL: func.func @trace_idiom_transposed
// CHECK: daphne.matMul
// CHECK: daphne.diagVector
func.func @trace_idiom_transposed(%x: !daphne.Matrix<3x4xf64>, %y: !daphne.Matrix<3x4xf64>) -> f64 {
    %true = "daphne.constant"() <{value = true}> : () -> i1
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %true) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<3x4xf64>, i1, i1) -> !daphne.Matrix<3x3xf64>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<3x3xf64>) -> !daphne.Matrix<3x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<3x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Row scaling: diag(v) @ X with v an n x 1 column vector and X n x m is
// rewritten to X * v: the matMul and the materialized n x n diagonal are
// gone, replaced by a single ewMul broadcasting v down the rows of X. The
// operand order is X (matrix) first, v (vector) second.
// CHECK-LABEL: func.func @row_scale
// CHECK-NOT: daphne.matMul
// CHECK-NOT: daphne.diagMatrix
// CHECK: daphne.ewMul
func.func @row_scale(%v: !daphne.Matrix<3x1xf64>, %x: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64>
    %p = "daphne.matMul"(%d, %x, %false, %false) : (!daphne.Matrix<3x3xf64>, !daphne.Matrix<3x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Negative case: unknown shape. Without a statically known row extent the
// rewrite cannot prove the broadcast branch will fire, so it fails closed and
// the matMul survives.
// CHECK-LABEL: func.func @row_scale_unknown_shape
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @row_scale_unknown_shape(%v: !daphne.Matrix<?x1xf64>, %x: !daphne.Matrix<?x?xf64>) -> !daphne.Matrix<?x?xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64>
    %p = "daphne.matMul"(%d, %x, %false, %false) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    "daphne.return"(%p) : (!daphne.Matrix<?x?xf64>) -> ()
}

// Negative case: a transposed matMul flag is a different chain the rewrite does
// not handle, so the matMul survives.
// CHECK-LABEL: func.func @row_scale_transposed
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @row_scale_transposed(%v: !daphne.Matrix<3x1xf64>, %x: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64> {
    %true = "daphne.constant"() <{value = true}> : () -> i1
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64>
    %p = "daphne.matMul"(%d, %x, %true, %false) : (!daphne.Matrix<3x3xf64>, !daphne.Matrix<3x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}
