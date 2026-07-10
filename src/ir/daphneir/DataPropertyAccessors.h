/*
 * Copyright 2025 The DAPHNE Consortium
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

#ifndef SRC_IR_DAPHNEIR_DATAPROPERTYACCESSORS_H
#define SRC_IR_DAPHNEIR_DATAPROPERTYACCESSORS_H

#include <ir/daphneir/DataPropertyTypes.h>

#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

#include <optional>

// Fail-closed accessors for the inferred data properties (shape, sparsity,
// symmetry) that DAPHNE stores on the MLIR type of a matrix or frame value. Each
// accessor returns std::nullopt when the property is unknown, either because
// the type is not a matrix/frame or because the property still holds its
// "unknown" sentinel (numRows/numCols == -1, sparsity == -1.0, symmetric ==
// BoolOrUnknown::Unknown).
//
// This matters because the canonicalizer runs both before property inference has
// filled these in and after it, so a rewrite may see a property unknown.
// Comparing the raw getter against a threshold is unsafe there: the -1.0 sentinel
// compares as an ordinary number, so `getSparsity() < 0.25` is true for an
// uninferred sparsity. These accessors only yield a value once the property is
// genuinely known:
//
//     if (std::optional<double> sp = knownSparsity(v); sp && *sp < 0.25)
//         ... // safe: never entered when sparsity is unknown

namespace mlir::daphne {

// Returns the statically known number of rows of `type`, or nullopt if the row
// extent is unknown or `type` is not a matrix/frame.
std::optional<ssize_t> knownNumRows(mlir::Type type);

// Returns the statically known number of columns of `type`, or nullopt if the
// column extent is unknown or `type` is not a matrix/frame.
std::optional<ssize_t> knownNumCols(mlir::Type type);

// Returns the statically known sparsity (nonzero fraction) of `type`, or nullopt
// if the sparsity is unknown or `type` is not a matrix. Frames carry no sparsity.
std::optional<double> knownSparsity(mlir::Type type);

// Returns the statically known symmetry of `type` as a bool, or nullopt if the
// symmetry is unknown or `type` is not a matrix. Frames carry no symmetry.
std::optional<bool> knownSymmetric(mlir::Type type);

// Convenience overloads reading the property off the type of a value.
inline std::optional<ssize_t> knownNumRows(mlir::Value value) { return knownNumRows(value.getType()); }
inline std::optional<ssize_t> knownNumCols(mlir::Value value) { return knownNumCols(value.getType()); }
inline std::optional<double> knownSparsity(mlir::Value value) { return knownSparsity(value.getType()); }
inline std::optional<bool> knownSymmetric(mlir::Value value) { return knownSymmetric(value.getType()); }

// Boolean predicates for the common "known-and-true / known-and-false" query,
// where the caller does not need the value itself. Each is fail-closed: an
// unknown property yields false. Note that isKnownSymmetric and
// isKnownNonSymmetric are NOT negations of each other: both are false while
// symmetry is unknown.

// True iff both extents of `type` are statically known.
inline bool hasKnownShape(mlir::Type type) { return knownNumRows(type) && knownNumCols(type); }
inline bool hasKnownShape(mlir::Value value) { return hasKnownShape(value.getType()); }

// True iff `type` is known to be symmetric.
inline bool isKnownSymmetric(mlir::Type type) { return knownSymmetric(type).value_or(false); }
inline bool isKnownSymmetric(mlir::Value value) { return isKnownSymmetric(value.getType()); }

// True iff `type` is known to be non-symmetric (symmetry inferred as false).
inline bool isKnownNonSymmetric(mlir::Type type) {
    std::optional<bool> sym = knownSymmetric(type);
    return sym && !*sym;
}
inline bool isKnownNonSymmetric(mlir::Value value) { return isKnownNonSymmetric(value.getType()); }

} // namespace mlir::daphne

#endif // SRC_IR_DAPHNEIR_DATAPROPERTYACCESSORS_H
