// RUN: daphne-opt --daphne-algebraic-simplify %s | FileCheck %s

// Locks in that ops annotated with an algebraic trait are simplified by
// --daphne-algebraic-simplify. One representative per trait is enough — the
// trait mechanism is the invariant, not the individual op.

// Involutive: EwMinusOp collapses --a to a.
// CHECK-LABEL: func.func @double_negate
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewMinus
func.func @double_negate(%arg0: f64) -> f64 {
    %0 = "daphne.ewMinus"(%arg0) : (f64) -> f64
    %1 = "daphne.ewMinus"(%0) : (f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// Involutive negative case: a single ewMinus must survive.
// CHECK-LABEL: func.func @single_negate
// CHECK: daphne.ewMinus
func.func @single_negate(%arg0: f64) -> f64 {
    %0 = "daphne.ewMinus"(%arg0) : (f64) -> f64
    "daphne.return"(%0) : (f64) -> ()
}

// IdentityOnIntegerElementType: floor of a signed integer scalar is a no-op.
// CHECK-LABEL: func.func @floor_int_scalar
// CHECK-SAME: (%[[ARG:.*]]: si64) -> si64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewFloor
func.func @floor_int_scalar(%arg0: si64) -> si64 {
    %0 = "daphne.ewFloor"(%arg0) : (si64) -> si64
    "daphne.return"(%0) : (si64) -> ()
}

// IdentityOnIntegerElementType negative case: floor of f64 must survive.
// CHECK-LABEL: func.func @floor_float_scalar
// CHECK: daphne.ewFloor
func.func @floor_float_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.ewFloor"(%arg0) : (f64) -> f64
    "daphne.return"(%0) : (f64) -> ()
}

// IdentityOnIntegerElementType also fires on matrix operands with integer
// element type — checked here for ceil to exercise a second adopter.
// CHECK-LABEL: func.func @ceil_int_matrix
// CHECK-SAME: (%[[ARG:.*]]: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewCeil
func.func @ceil_int_matrix(%arg0: !daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64> {
    %0 = "daphne.ewCeil"(%arg0) : (!daphne.Matrix<2x3xsi64>) -> !daphne.Matrix<2x3xsi64>
    "daphne.return"(%0) : (!daphne.Matrix<2x3xsi64>) -> ()
}

// Round on a float matrix must survive — third adopter, exercised on the
// negative side to confirm the trait's element-type gate is honored.
// CHECK-LABEL: func.func @round_float_matrix
// CHECK: daphne.ewRound
func.func @round_float_matrix(%arg0: !daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64> {
    %0 = "daphne.ewRound"(%arg0) : (!daphne.Matrix<2x3xf64>) -> !daphne.Matrix<2x3xf64>
    "daphne.return"(%0) : (!daphne.Matrix<2x3xf64>) -> ()
}

// NeutralOnZeroRHS on EwAddOp: x + 0 collapses to x for an integer operand.
// Gated to integers: over floats (-0.0) + 0.0 = +0.0, so x + 0.0 -> x would
// wrongly preserve a negative zero.
// CHECK-LABEL: func.func @add_zero_int
// CHECK-SAME: (%[[ARG:.*]]: si64) -> si64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewAdd
func.func @add_zero_int(%arg0: si64) -> si64 {
    %0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %1 = "daphne.ewAdd"(%arg0, %0) : (si64, si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// NeutralOnZeroRHS on EwAddOp negative case: over a float element type the
// rewrite must not fire (signed-zero fidelity). The ewAdd must survive.
// CHECK-LABEL: func.func @add_zero_float
// CHECK: daphne.ewAdd
func.func @add_zero_float(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %1 = "daphne.ewAdd"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnZeroRHS negative case: a non-zero RHS must survive.
// CHECK-LABEL: func.func @add_nonzero_scalar
// CHECK: daphne.ewAdd
func.func @add_nonzero_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %1 = "daphne.ewAdd"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnZeroRHS on EwSubOp: x - 0 collapses to x. This stays valid over
// floats too: x - 0.0 = x is exact for every IEEE value (including signed
// zero), so unlike ewAdd it is not gated to integers. Exercising the second
// adopter also catches a broken trait attachment on EwSubOp independently.
// CHECK-LABEL: func.func @sub_zero_scalar
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewSub
func.func @sub_zero_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %1 = "daphne.ewSub"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnOneRHS on EwMulOp: x * 1 collapses to x.
// CHECK-LABEL: func.func @mul_one_scalar
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewMul
func.func @mul_one_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %1 = "daphne.ewMul"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnOneRHS on EwDivOp: x / 1 collapses to x. Exercises the second
// adopter of NeutralOnOneRHS so a broken attachment on EwDivOp is caught
// independently of EwMulOp.
// CHECK-LABEL: func.func @div_one_scalar
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewDiv
func.func @div_one_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 1.0 : f64} : () -> f64
    %1 = "daphne.ewDiv"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// NeutralOnOneRHS negative case: x / 2 must survive.
// CHECK-LABEL: func.func @div_two_scalar
// CHECK: daphne.ewDiv
func.func @div_two_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 2.0 : f64} : () -> f64
    %1 = "daphne.ewDiv"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// RightAbsorbingOnZero on EwMulOp: x * 0 collapses to 0, but only for integer
// element types. Over IEEE floating point the identity is false (NaN * 0 = NaN,
// Inf * 0 = NaN), so the rewrite is gated on an integer element type.
// CHECK-LABEL: func.func @mul_zero_rhs_int
// CHECK: %[[Z:.*]] = "daphne.constant"() <{value = 0 : si64}>
// CHECK-NEXT: "daphne.return"(%[[Z]])
// CHECK-NOT: daphne.ewMul
func.func @mul_zero_rhs_int(%arg0: si64) -> si64 {
    %0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %1 = "daphne.ewMul"(%arg0, %0) : (si64, si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// LeftAbsorbingOnZero on EwMulOp: 0 * x collapses to 0 for an integer operand.
// CHECK-LABEL: func.func @mul_zero_lhs_int
// CHECK: %[[Z:.*]] = "daphne.constant"() <{value = 0 : si64}>
// CHECK-NEXT: "daphne.return"(%[[Z]])
// CHECK-NOT: daphne.ewMul
func.func @mul_zero_lhs_int(%arg0: si64) -> si64 {
    %0 = "daphne.constant"() {value = 0 : si64} : () -> si64
    %1 = "daphne.ewMul"(%0, %arg0) : (si64, si64) -> si64
    "daphne.return"(%1) : (si64) -> ()
}

// Absorbing-zero negative case: over a float element type the rewrite must not
// fire, because NaN * 0.0 = NaN and Inf * 0.0 = NaN. The ewMul must survive.
// CHECK-LABEL: func.func @mul_zero_float
// CHECK: daphne.ewMul
func.func @mul_zero_float(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %1 = "daphne.ewMul"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}
