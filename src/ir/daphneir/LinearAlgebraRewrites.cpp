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
        // The element-wise product's sparsity and symmetry, however, are NOT X's:
        // an element-wise product is at most as dense as either factor, so
        // reusing X's type would over-state the density (and X's symmetry says
        // nothing about X * t(Y)). Reset both to unknown so a later inference pass
        // derives them, rather than baking in a wrong value that representation
        // selection would then consume.
        auto xMatTy = llvm::cast<daphne::MatrixType>(x.getType());
        auto yMatTy = llvm::cast<daphne::MatrixType>(y.getType());
        Value transposedY =
            rewriter.create<daphne::TransposeOp>(matMulOp.getLoc(), yMatTy.withShape(*colsY, *rowsY), y);
        Type ewMulTy = xMatTy.withSparsity(-1.0).withSymmetric(daphne::BoolOrUnknown::Unknown);
        Value ewMul = rewriter.create<daphne::EwMulOp>(sumOp.getLoc(), ewMulTy, x, transposedY);
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
        // untouched since neither operand is a scalar). The product's shape
        // matches diag(v) @ X, so reuse the matmul's extents, but reset sparsity
        // and symmetry to unknown: they describe the matrix product, not the
        // element-wise scaling, so leave them for a later inference pass to derive.
        auto resTy = llvm::cast<daphne::MatrixType>(matMulOp.getResult().getType());
        Type ewMulTy = resTy.withSparsity(-1.0).withSymmetric(daphne::BoolOrUnknown::Unknown);
        rewriter.replaceOpWithNewOp<daphne::EwMulOp>(matMulOp, ewMulTy, x, v);
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
        // The product's shape matches X @ diag(v), so reuse the matmul's extents,
        // but reset sparsity and symmetry to unknown: they describe the matrix
        // product, not the element-wise scaling, so leave them for a later
        // inference pass to derive.
        auto vMatTy = llvm::cast<daphne::MatrixType>(v.getType());
        Value tv = rewriter.create<daphne::TransposeOp>(matMulOp.getLoc(), vMatTy.withShape(1, *rowsV), v);
        auto resTy = llvm::cast<daphne::MatrixType>(matMulOp.getResult().getType());
        Type ewMulTy = resTy.withSparsity(-1.0).withSymmetric(daphne::BoolOrUnknown::Unknown);
        rewriter.replaceOpWithNewOp<daphne::EwMulOp>(matMulOp, ewMulTy, x, tv);
        return success();
    }
};

/**
 * @brief Rewrites the scalar-factor idiom `sum(s * X)` to `s * sum(X)`.
 *
 * A scalar factor distributes over a total sum: `sum(s * X) = s * sum(X)`. This
 * hoists the multiply out of the O(n * m) element-wise product into a single
 * scalar multiply after the aggregate (and, since the element-wise product is
 * Pure, lets DCE drop it when the sum was its only user).
 *
 * The rewrite fails closed. It matches only an element-wise product of exactly
 * one scalar and one matrix operand (in either order, since EwMul is commutative
 * and its canonicalizer may swap a scalar lhs to the rhs), and only when the matrix's
 * element type equals the sum's result value type; see the per-guard comments.
 */
struct SumScalarFactorPattern : public OpRewritePattern<daphne::AllAggSumOp> {
    using OpRewritePattern<daphne::AllAggSumOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(daphne::AllAggSumOp sumOp, PatternRewriter &rewriter) const override {
        auto mulOp = sumOp.getArg().getDefiningOp<daphne::EwMulOp>();
        if (!mulOp)
            return failure();

        // Require exactly one scalar and one matrix operand; accept either order
        // (EwMul is commutative). Fail closed otherwise.
        Value lhs = mulOp.getLhs();
        Value rhs = mulOp.getRhs();
        Value scalar;
        Value matrix;
        if (CompilerUtils::hasScaType(lhs) && getMatrixElementTypeOrNull(rhs)) {
            scalar = lhs;
            matrix = rhs;
        } else if (CompilerUtils::hasScaType(rhs) && getMatrixElementTypeOrNull(lhs)) {
            scalar = rhs;
            matrix = lhs;
        } else {
            return failure();
        }

        // Soundness guard: the aggregate accumulates in its result value type, so
        // sum(s * X) accumulates in the product's type while s * sum(X) accumulates
        // in X's. If s promotes X (e.g. s:f64 over X:si64) those differ and the
        // si64 accumulation can overflow, changing the result. Fire only when X's
        // element type already equals the sum's result type, so it stays unchanged.
        Type elemMatrix = getMatrixElementTypeOrNull(matrix);
        if (elemMatrix != sumOp.getResult().getType())
            return failure();

        // Give sum(X) X's element type (which the guard proved equals the result
        // type); an unknown type would never resolve, as inference has already run.
        // The outer s * sum(X) is a scalar-by-scalar product reusing that type.
        Value innerSum = rewriter.create<daphne::AllAggSumOp>(sumOp.getLoc(), elemMatrix, matrix);
        rewriter.replaceOpWithNewOp<daphne::EwMulOp>(sumOp, sumOp.getResult().getType(), scalar, innerSum);
        return success();
    }
};

/**
 * @brief Rewrites a row-wise aggregate of a single-column matrix to its input.
 *
 * A row-wise aggregate reduces each row to one value, so over an n x 1 matrix it
 * touches exactly one element per row: `sumRow(X) = X`, and likewise for min/max.
 * The rewrite drops the aggregate entirely (and, since these aggregates are Pure,
 * DCE removes it when it was the operand's only user).
 *
 * Only sum/min/max are instantiated: over a singleton each simply returns that
 * element, so the value is unchanged and there is no accumulation to overflow.
 * mean/var/stddev are excluded, since they promote to a floating-point result type
 * (and a singleton variance is 0, not the element), as are idxMin/idxMax,
 * whose result is a position, not the value.
 *
 * The rewrite fails closed: it fires only when the column count is statically
 * known to be 1 and the aggregate preserves the element type (which sum/min/max
 * do via ValueTypeFromFirstArg). It deliberately does not require the full matrix
 * type to match, only the element type: the aggregate result carries no sparsity
 * (it has no SparsityFromArg trait), so replacing it with the input yields an
 * equal-or-more-precise type its users already accept.
 */
template <class AggOp> struct RowAggDim1IdentityPattern : public OpRewritePattern<AggOp> {
    using OpRewritePattern<AggOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AggOp op, PatternRewriter &rewriter) const override {
        Value arg = op.getArg();

        std::optional<ssize_t> cols = daphne::knownNumCols(arg);
        if (!cols || *cols != 1)
            return failure();

        Type elemArg = getMatrixElementTypeOrNull(arg);
        Type elemRes = getMatrixElementTypeOrNull(op.getResult());
        if (!elemArg || elemArg != elemRes)
            return failure();

        rewriter.replaceOp(op, arg);
        return success();
    }
};

/**
 * @brief Rewrites a column-wise aggregate of a single-row matrix to its input.
 *
 * The column-wise counterpart of `RowAggDim1IdentityPattern`: a column-wise
 * aggregate reduces each column to one value, so over a 1 x m matrix it is the
 * identity, `sumCol(X) = X` (and min/max). Guards on a statically-known row count
 * of 1; see `RowAggDim1IdentityPattern` for the sum/min/max-only rationale and
 * the element-type-only match.
 */
template <class AggOp> struct ColAggDim1IdentityPattern : public OpRewritePattern<AggOp> {
    using OpRewritePattern<AggOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(AggOp op, PatternRewriter &rewriter) const override {
        Value arg = op.getArg();

        std::optional<ssize_t> rows = daphne::knownNumRows(arg);
        if (!rows || *rows != 1)
            return failure();

        Type elemArg = getMatrixElementTypeOrNull(arg);
        Type elemRes = getMatrixElementTypeOrNull(op.getResult());
        if (!elemArg || elemArg != elemRes)
            return failure();

        rewriter.replaceOp(op, arg);
        return success();
    }
};

} // namespace

namespace mlir::daphne {

void populateLinearAlgebraRewritePatterns(RewritePatternSet &patterns) {
    MLIRContext *ctx = patterns.getContext();
    patterns.add<TraceIdiomPattern, RowScalePattern, ColScalePattern, SumScalarFactorPattern>(ctx);
    patterns.add<RowAggDim1IdentityPattern<RowAggSumOp>, RowAggDim1IdentityPattern<RowAggMinOp>,
                 RowAggDim1IdentityPattern<RowAggMaxOp>, ColAggDim1IdentityPattern<ColAggSumOp>,
                 ColAggDim1IdentityPattern<ColAggMinOp>, ColAggDim1IdentityPattern<ColAggMaxOp>>(ctx);
}

} // namespace mlir::daphne
