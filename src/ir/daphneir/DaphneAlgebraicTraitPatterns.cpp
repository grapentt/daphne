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
 * @brief Collapses `f(f(x))` to `f(x)` for any op tagged `IdempotentUnary`.
 */
struct IdempotentUnaryPattern : public RewritePattern {
    IdempotentUnaryPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdempotentUnary>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Operation *inner = op->getOperand(0).getDefiningOp();
        if (!inner || inner->getName() != op->getName())
            return failure();
        if (inner->getNumOperands() != 1 || inner->getNumResults() != 1)
            return failure();

        rewriter.replaceOp(op, inner->getResult(0));
        return success();
    }
};

/**
 * @brief Collapses `f(x, x)` to `x` for any op tagged `IdempotentBinary`.
 */
struct IdempotentBinaryPattern : public RewritePattern {
    IdempotentBinaryPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdempotentBinary>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();

        Value lhs = op->getOperand(0);
        Value rhs = op->getOperand(1);
        if (lhs != rhs || lhs.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, lhs);
        return success();
    }
};

/**
 * @brief Collapses `f(f(x, y), y)` to `f(x, y)` for any op tagged `IdempotentOnSelf`.
 */
struct IdempotentOnSelfPattern : public RewritePattern {
    IdempotentOnSelfPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdempotentOnSelf>())
            return failure();
        if (op->getNumOperands() != 2 || op->getNumResults() != 1)
            return failure();

        Operation *inner = op->getOperand(0).getDefiningOp();
        if (!inner || inner->getName() != op->getName())
            return failure();
        if (inner->getNumOperands() != 2 || inner->getNumResults() != 1)
            return failure();
        if (inner->getOperand(1) != op->getOperand(1))
            return failure();

        rewriter.replaceOp(op, inner->getResult(0));
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
        Type elemTy = getMatrixElementTypeOrNull(operand);
        if (!elemTy)
            elemTy = operand.getType();
        if (!llvm::isa<IntegerType>(elemTy))
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
 * Fires only when the zero operand's type equals the op's result type, so
 * broadcasting cases (scalar zero * matrix) do not accidentally shrink the
 * result to a scalar.
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
        if (zero.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, zero);
        return success();
    }
};

/**
 * @brief Replaces `f(x, 0)` with `0` for any op tagged `RightAbsorbingOnZero`.
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
        if (zero.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, zero);
        return success();
    }
};

/**
 * @brief Replaces `f(X)` with `X` when X's row dimension is statically 1.
 *
 * Bails on unknown row dimension so the rewrite stays sound under imprecise
 * shape inference.
 */
struct IdentityOnSingletonRowDimPattern : public RewritePattern {
    IdentityOnSingletonRowDimPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdentityOnSingletonRowDim>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Value operand = op->getOperand(0);
        auto mt = llvm::dyn_cast<daphne::MatrixType>(operand.getType());
        if (!mt || mt.getNumRows() != 1)
            return failure();
        if (operand.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, operand);
        return success();
    }
};

/**
 * @brief Replaces `f(X)` with `X` when X's column dimension is statically 1.
 */
struct IdentityOnSingletonColDimPattern : public RewritePattern {
    IdentityOnSingletonColDimPattern(MLIRContext *ctx) : RewritePattern(MatchAnyOpTypeTag{}, /*benefit=*/1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const override {
        if (!op->hasTrait<OpTrait::IdentityOnSingletonColDim>())
            return failure();
        if (op->getNumOperands() != 1 || op->getNumResults() != 1)
            return failure();

        Value operand = op->getOperand(0);
        auto mt = llvm::dyn_cast<daphne::MatrixType>(operand.getType());
        if (!mt || mt.getNumCols() != 1)
            return failure();
        if (operand.getType() != op->getResult(0).getType())
            return failure();

        rewriter.replaceOp(op, operand);
        return success();
    }
};

} // namespace

namespace mlir::daphne {

void populateAlgebraicTraitPatterns(RewritePatternSet &patterns) {
    MLIRContext *ctx = patterns.getContext();
    patterns.add<InvolutivePattern, IdempotentUnaryPattern, IdempotentBinaryPattern, IdempotentOnSelfPattern,
                 IdentityOnIntegerElementTypePattern, IdentityWhenSymmetricPattern, NeutralOnZeroRHSPattern,
                 NeutralOnOneRHSPattern, LeftAbsorbingOnZeroPattern, RightAbsorbingOnZeroPattern,
                 IdentityOnSingletonRowDimPattern, IdentityOnSingletonColDimPattern>(ctx);
}

} // namespace mlir::daphne
