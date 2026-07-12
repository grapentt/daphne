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

#include "ir/daphneir/DaphneAlgebraicTraitPatterns.h"

#include "ir/daphneir/Daphne.h"

#include "mlir/IR/Matchers.h"

using namespace mlir;

namespace {

// Returns the element type of a matrix-typed value, or a null Type otherwise.
static Type getMatrixElementTypeOrNull(Value v) {
    if (auto mt = llvm::dyn_cast<daphne::MatrixType>(v.getType()))
        return mt.getElementType();
    return {};
}

// Returns true when `v` has an integer element type (matrix element type for a
// matrix, the value type itself for a scalar). Absorbing/neutral rewrites that
// discard an operand are only valid over integers: IEEE floating point has
// NaN*0 = NaN, Inf*0 = NaN, and (-0.0)+0.0 = +0.0, none of which the algebraic
// identity preserves.
static bool hasIntegerElementType(Value v) {
    Type elemTy = getMatrixElementTypeOrNull(v);
    if (!elemTy)
        elemTy = v.getType();
    return llvm::isa<IntegerType>(elemTy);
}

// Returns true when `v` is a ConstantLike op whose scalar value is zero.
static bool isConstantZero(Value v) {
    return matchPattern(v, m_Zero()) || matchPattern(v, m_AnyZeroFloat());
}

// Returns true when `v` is a ConstantLike op whose scalar value is one.
static bool isConstantOne(Value v) { return matchPattern(v, m_One()) || matchPattern(v, m_OneFloat()); }

/**
 * @brief Collapses `f(f(x))` to `x` for any op tagged `Involutive`.
 *
 * Only checks that the outer op's result type matches the inner op's operand
 * type, so shape-swapping involutions (e.g. TransposeOp on rectangular
 * matrices) are handled correctly.
 */
struct InvolutivePattern : public RewritePattern {
    InvolutivePattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::Involutive>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Operation *inner = op->getOperand(0).getDefiningOp();
        if (!inner || inner->getName() != op->getName())
            return failure();
        if (inner->getNumOperands() != 1 || inner->getNumResults() != 1)
            return failure();

        Value innerInput = inner->getOperand(0);
        if (innerInput.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, innerInput);
        return success();
    }
};

/**
 * @brief Replaces `f(X)` with `X` when X's element type is an integer.
 */
struct IdentityOnIntegerElementTypePattern : public RewritePattern {
    IdentityOnIntegerElementTypePattern(MLIRContext *ctx)
        : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdentityOnIntegerElementType>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Value operand = op->getOperand(0);
        if (!hasIntegerElementType(operand))
            return failure();
        if (operand.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, operand);
        return success();
    }
};

/**
 * @brief Replaces `f(X)` with `X` when X's inferred symmetric property is true.
 *
 * Bails out on `BoolOrUnknown::Unknown` so the rewrite stays sound under
 * imprecise property inference.
 */
struct IdentityWhenSymmetricPattern : public RewritePattern {
    IdentityWhenSymmetricPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdentityWhenSymmetric>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Value operand = op->getOperand(0);
        auto mt = llvm::dyn_cast<daphne::MatrixType>(operand.getType());
        if (!mt || mt.getSymmetric() != BoolOrUnknown::True)
            return failure();
        if (operand.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, operand);
        return success();
    }
};

/**
 * @brief Replaces `f(x, 0)` with `x` for any op tagged `NeutralOnZeroRHS`.
 *
 * For `ewAdd` over a floating-point element type the rewrite is suppressed:
 * `(-0.0) + 0.0` is `+0.0`, so `x + 0.0 -> x` would wrongly preserve a negative
 * zero. Subtraction (`x - 0.0 = x`) is exact for all IEEE values, so `ewSub` is
 * left unguarded.
 */
struct NeutralOnZeroRHSPattern : public RewritePattern {
    NeutralOnZeroRHSPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::NeutralOnZeroRHS>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();
        if (!isConstantZero(op->getOperand(1)))
            return failure();
        if (llvm::isa<daphne::EwAddOp>(op) && !hasIntegerElementType(op->getResult(0)))
            return failure();

        Value lhs = op->getOperand(0);
        if (lhs.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, lhs);
        return success();
    }
};

/**
 * @brief Replaces `f(x, 1)` with `x` for any op tagged `NeutralOnOneRHS`.
 */
struct NeutralOnOneRHSPattern : public RewritePattern {
    NeutralOnOneRHSPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::NeutralOnOneRHS>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();
        if (!isConstantOne(op->getOperand(1)))
            return failure();

        Value lhs = op->getOperand(0);
        if (lhs.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, lhs);
        return success();
    }
};

/**
 * @brief Replaces `f(0, x)` with `0` for any op tagged `LeftAbsorbingOnZero`.
 *
 * Restricted to integer element types: over IEEE floating point the identity
 * `0 * x = 0` is false (NaN * 0 = NaN, Inf * 0 = NaN). Fires only when the zero
 * operand's type equals the op's result type, so broadcasting cases (scalar
 * zero * matrix) do not accidentally shrink the result to a scalar.
 */
struct LeftAbsorbingOnZeroPattern : public RewritePattern {
    LeftAbsorbingOnZeroPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::LeftAbsorbingOnZero>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();
        Value zero = op->getOperand(0);
        if (!isConstantZero(zero))
            return failure();
        if (!hasIntegerElementType(op->getResult(0)))
            return failure();
        if (zero.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, zero);
        return success();
    }
};

/**
 * @brief Replaces `f(x, 0)` with `0` for any op tagged `RightAbsorbingOnZero`.
 *
 * Restricted to integer element types for the same reason as
 * `LeftAbsorbingOnZeroPattern`.
 */
struct RightAbsorbingOnZeroPattern : public RewritePattern {
    RightAbsorbingOnZeroPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::RightAbsorbingOnZero>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();
        Value zero = op->getOperand(1);
        if (!isConstantZero(zero))
            return failure();
        if (!hasIntegerElementType(op->getResult(0)))
            return failure();
        if (zero.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, zero);
        return success();
    }
};

} // namespace

namespace mlir::daphne {

void populateAlgebraicTraitPatterns(RewritePatternSet &patterns) {
    MLIRContext *ctx = patterns.getContext();
    patterns.add<InvolutivePattern, IdentityOnIntegerElementTypePattern, IdentityWhenSymmetricPattern,
                 NeutralOnZeroRHSPattern, NeutralOnOneRHSPattern, LeftAbsorbingOnZeroPattern,
                 RightAbsorbingOnZeroPattern>(ctx);
}

} // namespace mlir::daphne
