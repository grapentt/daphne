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

// Column scaling: X @ diag(v) with X n x m and v an m x 1 column vector is
// rewritten to X * t(v): the matMul and the materialized m x m diagonal are
// gone, replaced by a transpose of v to a 1 x m row vector and a single ewMul
// broadcasting it across the columns of X. The operand order is X (matrix)
// first, t(v) (row vector) second.
// CHECK-LABEL: func.func @col_scale
// CHECK-NOT: daphne.matMul
// CHECK-NOT: daphne.diagMatrix
// CHECK: daphne.transpose
// CHECK: daphne.ewMul
func.func @col_scale(%x: !daphne.Matrix<3x4xf64>, %v: !daphne.Matrix<4x1xf64>) -> !daphne.Matrix<3x4xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<4x1xf64>) -> !daphne.Matrix<4x4xf64>
    %p = "daphne.matMul"(%x, %d, %false, %false) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}

// Negative case: unknown shape. Without a statically known column extent the
// rewrite cannot prove the broadcast branch will fire, so it fails closed and
// the matMul survives.
// CHECK-LABEL: func.func @col_scale_unknown_shape
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @col_scale_unknown_shape(%x: !daphne.Matrix<?x?xf64>, %v: !daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<?x1xf64>) -> !daphne.Matrix<?x?xf64>
    %p = "daphne.matMul"(%x, %d, %false, %false) : (!daphne.Matrix<?x?xf64>, !daphne.Matrix<?x?xf64>, i1, i1) -> !daphne.Matrix<?x?xf64>
    "daphne.return"(%p) : (!daphne.Matrix<?x?xf64>) -> ()
}

// Negative case: a transposed matMul flag is a different chain the rewrite does
// not handle, so the matMul survives.
// CHECK-LABEL: func.func @col_scale_transposed
// CHECK: daphne.diagMatrix
// CHECK: daphne.matMul
func.func @col_scale_transposed(%x: !daphne.Matrix<3x4xf64>, %v: !daphne.Matrix<4x1xf64>) -> !daphne.Matrix<3x4xf64> {
    %true = "daphne.constant"() <{value = true}> : () -> i1
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<4x1xf64>) -> !daphne.Matrix<4x4xf64>
    %p = "daphne.matMul"(%x, %d, %false, %true) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64>) -> ()
}

// The emitted ewMul must NOT inherit the sparsity/symmetry inferred for the ops
// it replaces: those properties describe a matrix product (or, for the trace
// idiom, the left factor), not the element-wise result, so the rewrites reset
// them to unknown for a later inference pass to re-derive. Inference does not
// re-run on the default pipeline after this pass, so a leaked value would
// persist, so these cases lock the reset in.

// Row scaling: the matMul result carries a concrete sparsity, and the emitted
// ewMul must drop it (result type is a plain 3x4xf64, no sp[...] annotation).
// CHECK-LABEL: func.func @row_scale_resets_sparsity
// CHECK: %[[M:.*]] = "daphne.ewMul"
// CHECK-SAME: -> !daphne.Matrix<3x4xf64>
func.func @row_scale_resets_sparsity(%v: !daphne.Matrix<3x1xf64>, %x: !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64:sp[5.000000e-01]> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<3x1xf64>) -> !daphne.Matrix<3x3xf64>
    %p = "daphne.matMul"(%d, %x, %false, %false) : (!daphne.Matrix<3x3xf64>, !daphne.Matrix<3x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64:sp[5.000000e-01]>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64:sp[5.000000e-01]>) -> ()
}

// Column scaling: same invariant, the concrete matMul sparsity must not reach
// the ewMul result.
// CHECK-LABEL: func.func @col_scale_resets_sparsity
// CHECK: %[[M:.*]] = "daphne.ewMul"
// CHECK-SAME: -> !daphne.Matrix<3x4xf64>
func.func @col_scale_resets_sparsity(%x: !daphne.Matrix<3x4xf64>, %v: !daphne.Matrix<4x1xf64>) -> !daphne.Matrix<3x4xf64:sp[2.500000e-01]> {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %d = "daphne.diagMatrix"(%v) : (!daphne.Matrix<4x1xf64>) -> !daphne.Matrix<4x4xf64>
    %p = "daphne.matMul"(%x, %d, %false, %false) : (!daphne.Matrix<3x4xf64>, !daphne.Matrix<4x4xf64>, i1, i1) -> !daphne.Matrix<3x4xf64:sp[2.500000e-01]>
    "daphne.return"(%p) : (!daphne.Matrix<3x4xf64:sp[2.500000e-01]>) -> ()
}

// Trace idiom: the left factor X carries a concrete sparsity; the emitted ewMul
// result must reset it (X remains its own type as an operand, only the product's
// result type is reset). Also exercises a symmetric matMul result being dropped.
// CHECK-LABEL: func.func @trace_idiom_resets_props
// CHECK: %[[M:.*]] = "daphne.ewMul"
// CHECK-SAME: -> !daphne.Matrix<3x3xf64>
func.func @trace_idiom_resets_props(%x: !daphne.Matrix<3x3xf64:sp[3.000000e-01]>, %y: !daphne.Matrix<3x3xf64>) -> f64 {
    %false = "daphne.constant"() <{value = false}> : () -> i1
    %p = "daphne.matMul"(%x, %y, %false, %false) : (!daphne.Matrix<3x3xf64:sp[3.000000e-01]>, !daphne.Matrix<3x3xf64>, i1, i1) -> !daphne.Matrix<3x3xf64:symmetric[true]>
    %d = "daphne.diagVector"(%p) : (!daphne.Matrix<3x3xf64:symmetric[true]>) -> !daphne.Matrix<3x1xf64>
    %s = "daphne.sumAll"(%d) : (!daphne.Matrix<3x1xf64>) -> f64
    "daphne.return"(%s) : (f64) -> ()
}

// Scalar-factor hoisting: sum(s * X) is rewritten to s * sum(X), the scalar
// multiply moves out of the element-wise product to after the aggregate. The
// aggregate now runs directly on the matrix X, and the surviving ewMul is a
// scalar-by-scalar product (f64, f64) -> f64. Here s and X share the f64 element
// type, so the accumulation type is unchanged and the rewrite fires.
// CHECK-LABEL: func.func @sum_scalar_factor
// CHECK: %[[S:.*]] = "daphne.sumAll"(%{{.*}}) : (!daphne.Matrix<3x4xf64>) -> f64
// CHECK: "daphne.ewMul"(%{{.*}}, %[[S]]) : (f64, f64) -> f64
func.func @sum_scalar_factor(%s: f64, %x: !daphne.Matrix<3x4xf64>) -> f64 {
    %m = "daphne.ewMul"(%s, %x) : (f64, !daphne.Matrix<3x4xf64>) -> !daphne.Matrix<3x4xf64>
    %r = "daphne.sumAll"(%m) : (!daphne.Matrix<3x4xf64>) -> f64
    "daphne.return"(%r) : (f64) -> ()
}

// Negative case: the scalar promotes the matrix element type (s : f64 over an
// si64 matrix, so the product is f64). Moving the sum inside would accumulate in
// si64, a different accumulation type that can overflow, so the rewrite fails
// closed and the element-wise product over the matrix survives.
// CHECK-LABEL: func.func @sum_scalar_factor_promoting
// CHECK: daphne.ewMul
// CHECK: daphne.sumAll
func.func @sum_scalar_factor_promoting(%s: f64, %x: !daphne.Matrix<3x4xsi64>) -> f64 {
    %m = "daphne.ewMul"(%s, %x) : (f64, !daphne.Matrix<3x4xsi64>) -> !daphne.Matrix<3x4xf64>
    %r = "daphne.sumAll"(%m) : (!daphne.Matrix<3x4xf64>) -> f64
    "daphne.return"(%r) : (f64) -> ()
}
