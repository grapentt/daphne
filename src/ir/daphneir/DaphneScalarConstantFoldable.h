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

#ifndef SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_H
#define SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_H

#include "ir/daphneir/Daphne.h"

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::daphne {
#include <ir/daphneir/DaphneScalarConstantFoldable.h.inc>
} // namespace mlir::daphne

namespace mlir::daphne {

/**
 * @brief Fold a scalar op with constant operands by delegating to the op's
 *        `ScalarConstantFoldable` methods.
 *
 * The driver walks numeric kinds in the same precedence order the legacy
 * per-op templates used (float, then signed int, unsigned int, bool) and returns
 * the first non-empty result. If none of the kinds apply, a null Attribute is
 * returned — the caller should propagate that as "no folding possible".
 *
 * Arity is inferred from `operands.size()`; the driver dispatches to the
 * unary or binary methods accordingly. Comparison-flavoured ops (numeric in,
 * bool out) are recognised by result type: when `resultType` is `i1`, the
 * driver calls the `foldScalarCmp*` methods instead of the arithmetic ones.
 *
 * String semantics and division-by-zero raising are intentionally out of
 * scope; the corresponding ops keep their bespoke folders in `Fold.cpp`.
 */
::mlir::Attribute foldScalarOp(ScalarConstantFoldable op, ::llvm::ArrayRef<::mlir::Attribute> operands,
                               ::mlir::Type resultType, ::mlir::Location loc);

} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DAPHNESCALARCONSTANTFOLDABLE_H
