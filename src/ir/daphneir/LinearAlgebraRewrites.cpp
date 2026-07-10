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

#include "ir/daphneir/LinearAlgebraRewrites.h"

#include "ir/daphneir/Daphne.h"
#include "ir/daphneir/DataPropertyAccessors.h"

#include "compiler/utils/CompilerUtils.h"

using namespace mlir;

namespace {

// Returns the element type of a matrix-typed value, or a null Type otherwise.
static Type getMatrixElementTypeOrNull(Value v) {
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(v.getType()))
        return mt.getElementType();
    return {};
}

/**
 * @brief Rewrites the trace idiom `sum(diag(X @ Y))` to `sum(X * t(Y))`.
 *
 * DAPHNE has no trace op, so the trace of a matrix product is spelled
 * `sum(diagVector(X @ Y))`. Using the identity
 *
 *     trace(X @ Y) = sum_i (X @ Y)[i, i]
 *                  = sum_i sum_j X[i, j] * Y[j, i]
 *                  = sum(X (elementwise*) t(Y)),
 *
 * this replaces an O(n^2 * m) matrix product, whose off-diagonal entries the
 * diagVector immediately discards, with an O(n * m) element-wise product.
 *
 * The rewrite fails closed, matching only a non-transposed product of operands
 * that share an element type and whose result is statically known to be square;
 * see the per-guard comments below for why each is needed.
 */
struct TraceIdiomPattern : public OpRewritePattern<daphne::AllAggSumOp> {
    using OpRewritePattern<daphne::AllAggSumOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::AllAggSumOp sumOp, PatternRewriter &rewriter) const override {
        auto diagOp = sumOp.getArg().getDefiningOp<daphne::DiagVectorOp>();
        if (!diagOp)
            return failure();

        auto matMulOp = diagOp.getArg().getDefiningOp<daphne::MatMulOp>();
        if (!matMulOp)
            return failure();

        // Only the plain, non-transposed product is the trace idiom we handle.
        if (CompilerUtils::constantOrDefault<bool>(matMulOp.getTransa(), false) ||
            CompilerUtils::constantOrDefault<bool>(matMulOp.getTransb(), false))
            return failure();

        Value x = matMulOp.getLhs();
        Value y = matMulOp.getRhs();

        // The emitted ewMul must be well-typed: X and Y need the same element type.
        Type elemX = getMatrixElementTypeOrNull(x);
        Type elemY = getMatrixElementTypeOrNull(y);
        if (!elemX || !elemY || elemX != elemY)
            return failure();

        // X @ Y must be square: X is m x n, Y is n x m. Fail closed on unknown
        // shape (the diagVector kernel would throw at runtime otherwise, so a
        // non-square case is a malformed program either way).
        std::optional<ssize_t> rowsX = daphne::knownNumRows(x);
        std::optional<ssize_t> colsX = daphne::knownNumCols(x);
        std::optional<ssize_t> rowsY = daphne::knownNumRows(y);
        std::optional<ssize_t> colsY = daphne::knownNumCols(y);
        if (!rowsX || !colsX || !rowsY || !colsY)
            return failure();
        if (*colsX != *rowsY || *rowsX != *colsY)
            return failure();

        // sum(X * t(Y)): the transpose gives an m x n matrix matching X, so the
        // element-wise product needs no broadcasting.
        Value transposedY = rewriter.create<daphne::TransposeOp>(matMulOp.getLoc(), y);
        Value ewMul = rewriter.create<daphne::EwMulOp>(sumOp.getLoc(), daphne::UnknownType::get(getContext()), x,
                                                       transposedY);
        rewriter.replaceOpWithNewOp<daphne::AllAggSumOp>(sumOp, sumOp.getResult().getType(), ewMul);
        return success();
    }
};

} // namespace

namespace mlir::daphne {

void populateLinearAlgebraRewritePatterns(RewritePatternSet &patterns) {
    MLIRContext *ctx = patterns.getContext();
    patterns.add<TraceIdiomPattern>(ctx);
}

} // namespace mlir::daphne
