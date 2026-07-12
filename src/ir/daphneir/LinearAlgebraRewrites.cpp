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

        // sum(X * t(Y)): t(Y) is an m x n matrix matching X, so the element-wise
        // product needs no broadcasting and has X's shape. Both created ops are
        // given concrete result types computed from the known extents: shape
        // inference has already run by the time this pass fires, so a shape left
        // unknown here would never be resolved and kernel dispatch would fail.
        auto yMatTy = llvm::cast<daphne::MatrixType>(y.getType());
        Value transposedY =
            rewriter.create<daphne::TransposeOp>(matMulOp.getLoc(), yMatTy.withShape(*colsY, *rowsY), y);
        Value ewMul = rewriter.create<daphne::EwMulOp>(sumOp.getLoc(), x.getType(), x, transposedY);
        rewriter.replaceOpWithNewOp<daphne::AllAggSumOp>(sumOp, sumOp.getResult().getType(), ewMul);
        return success();
    }
};

/**
 * @brief Rewrites the row-scaling idiom `diag(v) @ X` to `X * v`.
 *
 * Multiplying by a diagonal matrix on the left scales each row: with v an
 * n x 1 column vector and X an n x m matrix, `diag(v) @ X` is n x m and its
 * row i equals `v[i] * X[i, :]`. The element-wise product `X * v` broadcasts
 * the column vector down the rows and computes exactly that, so the rewrite
 * replaces an O(n^2 * m) matrix product against a materialized n x n diagonal
 * with an O(n * m) element-wise product (and, since diagMatrix is Pure, lets
 * DCE drop the now-dead diagonal).
 *
 * Operand order matters here: the element-wise kernel broadcasts a column
 * vector only when the matrix is the left operand, so the matrix must be the
 * lhs and the vector the rhs. That order is also safe from later reordering:
 * EwMul's canonicalizer only swaps a scalar lhs, and neither operand here is a
 * scalar.
 *
 * The rewrite fails closed, matching only a non-transposed product whose
 * diagonal argument is a statically-known column vector, whose operands share
 * an element type, and whose row extents statically agree (so the broadcast
 * branch is guaranteed to fire); see the per-guard comments below.
 */
struct RowScalePattern : public OpRewritePattern<daphne::MatMulOp> {
    using OpRewritePattern<daphne::MatMulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::MatMulOp matMulOp, PatternRewriter &rewriter) const override {
        auto diagOp = matMulOp.getLhs().getDefiningOp<daphne::DiagMatrixOp>();
        if (!diagOp)
            return failure();

        // Only the plain, non-transposed product carries the row-scaling identity.
        if (CompilerUtils::constantOrDefault<bool>(matMulOp.getTransa(), false) ||
            CompilerUtils::constantOrDefault<bool>(matMulOp.getTransb(), false))
            return failure();

        Value v = diagOp.getArg();
        Value x = matMulOp.getRhs();

        // The emitted ewMul must be well-typed: X and v need the same element type.
        Type elemV = getMatrixElementTypeOrNull(v);
        Type elemX = getMatrixElementTypeOrNull(x);
        if (!elemV || !elemX || elemV != elemX)
            return failure();

        // v must be a statically-known n x 1 column vector, and X's row extent
        // must statically match it, so the element-wise kernel is guaranteed to
        // take the column-vector broadcast branch (matrix rows == vector rows,
        // vector has one column). Fail closed on any unknown or mismatched
        // extent: a wrong broadcast would silently miscompute rather than throw.
        std::optional<ssize_t> rowsV = daphne::knownNumRows(v);
        std::optional<ssize_t> colsV = daphne::knownNumCols(v);
        std::optional<ssize_t> rowsX = daphne::knownNumRows(x);
        if (!rowsV || !colsV || !rowsX)
            return failure();
        if (*colsV != 1 || *rowsV != *rowsX)
            return failure();

        // X * v: matrix as lhs, column vector as rhs (the only order the
        // broadcast kernel handles, and the order EwMul's canonicalizer leaves
        // untouched since neither operand is a scalar).
        rewriter.replaceOpWithNewOp<daphne::EwMulOp>(matMulOp, matMulOp.getResult().getType(), x, v);
        return success();
    }
};

/**
 * @brief Rewrites the column-scaling idiom `X @ diag(v)` to `X * t(v)`.
 *
 * Multiplying by a diagonal matrix on the right scales each column: with v an
 * m x 1 column vector and X an n x m matrix, `X @ diag(v)` is n x m and its
 * column j equals `v[j] * X[:, j]`. Transposing v to the 1 x m row vector t(v)
 * and forming the element-wise product `X * t(v)` broadcasts that row vector
 * down the rows of X and computes exactly this, so the rewrite replaces an
 * O(n * m^2) matrix product against a materialized m x m diagonal with an
 * O(n * m) element-wise product plus a cheap transpose (and, since diagMatrix
 * is Pure, lets DCE drop the now-dead diagonal).
 *
 * The transpose is needed here: the element-wise kernel broadcasts across
 * columns only when the vector is a row vector (1 x m), so v (m x 1) must be
 * transposed. Operand order matters too: that broadcast branch is
 * anchored on the matrix being the left operand, so the matrix must be the lhs
 * and the row vector the rhs. The order is safe from later reordering:
 * EwMul's canonicalizer only swaps a scalar lhs, and neither operand here is a
 * scalar.
 *
 * The rewrite fails closed, matching only a non-transposed product whose
 * diagonal argument is a statically-known column vector, whose operands share
 * an element type, and whose column extent statically agrees with the vector
 * length (so the broadcast branch is guaranteed to fire); see the per-guard
 * comments below.
 */
struct ColScalePattern : public OpRewritePattern<daphne::MatMulOp> {
    using OpRewritePattern<daphne::MatMulOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::MatMulOp matMulOp, PatternRewriter &rewriter) const override {
        auto diagOp = matMulOp.getRhs().getDefiningOp<daphne::DiagMatrixOp>();
        if (!diagOp)
            return failure();

        // Only the plain, non-transposed product carries the column-scaling identity.
        if (CompilerUtils::constantOrDefault<bool>(matMulOp.getTransa(), false) ||
            CompilerUtils::constantOrDefault<bool>(matMulOp.getTransb(), false))
            return failure();

        Value x = matMulOp.getLhs();
        Value v = diagOp.getArg();

        // The emitted ewMul must be well-typed: X and v need the same element type.
        Type elemV = getMatrixElementTypeOrNull(v);
        Type elemX = getMatrixElementTypeOrNull(x);
        if (!elemV || !elemX || elemV != elemX)
            return failure();

        // v must be a statically-known m x 1 column vector, and X's column extent
        // must statically match its length, so t(v) is a 1 x m row vector whose
        // width equals X's and the element-wise kernel is guaranteed to take the
        // row-vector broadcast branch (matrix cols == vector cols, vector has one
        // row). Compare against X's columns here: the diagonal is on the right,
        // so it scales columns, unlike the row-scaling pattern. Fail closed on any
        // unknown or mismatched extent: a wrong broadcast would silently
        // miscompute rather than throw.
        std::optional<ssize_t> rowsV = daphne::knownNumRows(v);
        std::optional<ssize_t> colsV = daphne::knownNumCols(v);
        std::optional<ssize_t> colsX = daphne::knownNumCols(x);
        if (!rowsV || !colsV || !colsX)
            return failure();
        if (*colsV != 1 || *rowsV != *colsX)
            return failure();

        // X * t(v): matrix as lhs, row vector as rhs (the only order the broadcast
        // kernel handles, and the order EwMul's canonicalizer leaves untouched since
        // neither operand is a scalar). t(v) is given a concrete 1 x m result type:
        // shape inference has already run by the time this pass fires, so a shape
        // left unknown here would never be resolved and kernel dispatch would fail.
        auto vMatTy = llvm::cast<daphne::MatrixType>(v.getType());
        Value tv = rewriter.create<daphne::TransposeOp>(matMulOp.getLoc(), vMatTy.withShape(1, *rowsV), v);
        rewriter.replaceOpWithNewOp<daphne::EwMulOp>(matMulOp, matMulOp.getResult().getType(), x, tv);
        return success();
    }
};

} // namespace

namespace mlir::daphne {

void populateLinearAlgebraRewritePatterns(RewritePatternSet &patterns) {
    MLIRContext *ctx = patterns.getContext();
    patterns.add<TraceIdiomPattern, RowScalePattern, ColScalePattern>(ctx);
}

} // namespace mlir::daphne
