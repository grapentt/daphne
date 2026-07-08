/*
 * Copyright 2026 The DAPHNE Consortium
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_TD
#define SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_TD

include "mlir/IR/OpBase.td"

// ****************************************************************************
// Non-parametric algebraic traits
// ****************************************************************************

// f(f(x)) = x. Unlike MLIR's upstream IsInvolution, this tolerates ops
// whose result type differs from the operand type (e.g. TransposeOp).
def Involutive : NativeOpTrait<"Involutive">;

// (No umbrella Idempotent marker: MLIR reserves that name in
// mlir/IR/OpBase.td for its own trait.)

// Unary f: f(f(x)) = f(x).
def IdempotentUnary : NativeOpTrait<"IdempotentUnary">;

// Binary f: f(x, x) = x.
def IdempotentBinary : NativeOpTrait<"IdempotentBinary">;

// Binary f: f(f(x, y), y) = f(x, y).
def IdempotentOnSelf : NativeOpTrait<"IdempotentOnSelf">;

// f(X) = X when X's element type is any integer.
def IdentityOnIntegerElementType : NativeOpTrait<"IdentityOnIntegerElementType">;

// f(X) = X when X's inferred symmetric property is true.
def IdentityWhenSymmetric : NativeOpTrait<"IdentityWhenSymmetric">;

#endif // SRC_IR_DAPHNEIR_DAPHNEALGEBRAICTRAITS_TD
