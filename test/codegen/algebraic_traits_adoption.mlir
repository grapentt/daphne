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

// NeutralOnZeroRHS on EwAddOp: x + 0 collapses to x for a scalar operand.
// CHECK-LABEL: func.func @add_zero_scalar
// CHECK-SAME: (%[[ARG:.*]]: f64) -> f64
// CHECK-NEXT: "daphne.return"(%[[ARG]])
// CHECK-NOT: daphne.ewAdd
func.func @add_zero_scalar(%arg0: f64) -> f64 {
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

// NeutralOnZeroRHS on EwSubOp: x - 0 collapses to x. One adopter shared per
// trait is enough to lock in the mechanism; this exercises the second adopter
// so a broken trait attachment on EwSubOp is caught independently of EwAddOp.
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

// RightAbsorbingOnZero on EwMulOp: x * 0 collapses to 0 for a scalar operand.
// The rewrite requires the zero constant's type to match the result type,
// which is trivially the case here (both f64).
// CHECK-LABEL: func.func @mul_zero_rhs_scalar
// CHECK: %[[Z:.*]] = "daphne.constant"() <{value = 0.000000e+00 : f64}>
// CHECK-NEXT: "daphne.return"(%[[Z]])
// CHECK-NOT: daphne.ewMul
func.func @mul_zero_rhs_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %1 = "daphne.ewMul"(%arg0, %0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}

// LeftAbsorbingOnZero on EwMulOp: 0 * x collapses to 0 for a scalar operand.
// CHECK-LABEL: func.func @mul_zero_lhs_scalar
// CHECK: %[[Z:.*]] = "daphne.constant"() <{value = 0.000000e+00 : f64}>
// CHECK-NEXT: "daphne.return"(%[[Z]])
// CHECK-NOT: daphne.ewMul
func.func @mul_zero_lhs_scalar(%arg0: f64) -> f64 {
    %0 = "daphne.constant"() {value = 0.0 : f64} : () -> f64
    %1 = "daphne.ewMul"(%0, %arg0) : (f64, f64) -> f64
    "daphne.return"(%1) : (f64) -> ()
}
