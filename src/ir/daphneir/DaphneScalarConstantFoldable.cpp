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

#include "ir/daphneir/DaphneScalarConstantFoldable.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::daphne {
#include <ir/daphneir/DaphneScalarConstantFoldable.cpp.inc>
} // namespace mlir::daphne

using namespace mlir;

namespace {

// -------------------------------------------------------------------------
// Attribute-kind probes
// -------------------------------------------------------------------------
//
// Kept as free helpers so the driver reads as a sequence of guarded attempts
// rather than a nested if-else pyramid.

bool isFloatAttrPair(Attribute a, Attribute b) { return llvm::isa<FloatAttr>(a) && llvm::isa<FloatAttr>(b); }

bool isIntAttrPair(Attribute a, Attribute b) { return llvm::isa<IntegerAttr>(a) && llvm::isa<IntegerAttr>(b); }

bool isBoolAttrPair(Attribute a, Attribute b) { return llvm::isa<BoolAttr>(a) && llvm::isa<BoolAttr>(b); }

// The legacy templates promoted the narrower of two integer/float operands to
// the width of the wider one. We recreate that step here so op lambdas never
// have to worry about mixed-width inputs.
//
// Returns a (possibly new) attribute pair whose values are ready to feed into
// the op's fold method; the caller must cast back to the concrete Attr kind.
template <typename AttrKind>
std::pair<AttrKind, AttrKind> promoteWidth(AttrKind lhs, AttrKind rhs, Location loc);

template <> std::pair<FloatAttr, FloatAttr> promoteWidth(FloatAttr lhs, FloatAttr rhs, Location /*loc*/) {
    // FloatAttr::getValueAsDouble already normalises to double for our purposes;
    // we return the originals since op lambdas operate on APFloat directly.
    return {lhs, rhs};
}

template <> std::pair<IntegerAttr, IntegerAttr> promoteWidth(IntegerAttr lhs, IntegerAttr rhs, Location loc) {
    Type lTy = lhs.getType();
    Type rTy = rhs.getType();
    if (!llvm::isa<IntegerType>(lTy) || !llvm::isa<IntegerType>(rTy))
        return {lhs, rhs};
    unsigned lw = lTy.getIntOrFloatBitWidth();
    unsigned rw = rTy.getIntOrFloatBitWidth();
    if (lw == rw)
        return {lhs, rhs};
    if (lw < rw) {
        auto ext = llvm::cast<IntegerType>(rTy).isUnsigned() ? lhs.getValue().zextOrTrunc(rw)
                                                             : lhs.getValue().sextOrTrunc(rw);
        return {IntegerAttr::getChecked(loc, rTy, ext), rhs};
    }
    auto ext =
        llvm::cast<IntegerType>(lTy).isUnsigned() ? rhs.getValue().zextOrTrunc(lw) : rhs.getValue().sextOrTrunc(lw);
    return {lhs, IntegerAttr::getChecked(loc, lTy, ext)};
}

} // namespace

namespace mlir::daphne {

// -------------------------------------------------------------------------
// Unary driver
// -------------------------------------------------------------------------

static Attribute foldScalarUnary(ScalarConstantFoldable op, Attribute operand, Type resultType, Location loc) {
    if (!operand)
        return {};

    if (auto f = llvm::dyn_cast<FloatAttr>(operand))
        if (auto v = op.foldScalarUnaryFloat(f.getValue()))
            return FloatAttr::getChecked(loc, resultType, *v);

    if (auto i = llvm::dyn_cast<IntegerAttr>(operand))
        if (auto v = op.foldScalarUnaryInt(i.getValue()))
            return IntegerAttr::getChecked(loc, resultType, *v);

    return {};
}

// -------------------------------------------------------------------------
// Binary driver — arithmetic (numeric in, numeric out)
// -------------------------------------------------------------------------

static Attribute foldScalarArithBinary(ScalarConstantFoldable op, Attribute lhs, Attribute rhs, Type resultType,
                                       Location loc) {
    if (isFloatAttrPair(lhs, rhs)) {
        auto [l, r] = promoteWidth(llvm::cast<FloatAttr>(lhs), llvm::cast<FloatAttr>(rhs), loc);
        if (auto v = op.foldScalarFloat(l.getValue(), r.getValue()))
            return FloatAttr::getChecked(loc, resultType, *v);
    }
    // BoolAttr checked before IntegerAttr: MLIR's BoolAttr is-a IntegerAttr of
    // type i1, so `isa<IntegerAttr>` matches bool pairs too. Dispatch to the
    // bool method first to preserve the legacy fold order.
    if (isBoolAttrPair(lhs, rhs)) {
        auto l = llvm::cast<BoolAttr>(lhs).getValue();
        auto r = llvm::cast<BoolAttr>(rhs).getValue();
        if (auto v = op.foldScalarBool(l, r))
            return BoolAttr::get(op->getContext(), *v);
    }
    if (isIntAttrPair(lhs, rhs)) {
        auto [l, r] = promoteWidth(llvm::cast<IntegerAttr>(lhs), llvm::cast<IntegerAttr>(rhs), loc);
        // Precedence matches the legacy signedness dispatch in Fold.cpp.
        if (auto intTy = llvm::dyn_cast<IntegerType>(resultType); intTy && intTy.isSigned()) {
            if (auto v = op.foldScalarSInt(l.getValue(), r.getValue()))
                return IntegerAttr::getChecked(loc, resultType, *v);
        } else if (auto intTy = llvm::dyn_cast<IntegerType>(resultType); intTy && intTy.isUnsigned()) {
            if (auto v = op.foldScalarUInt(l.getValue(), r.getValue()))
                return IntegerAttr::getChecked(loc, resultType, *v);
        }
        // Signless / Index intentionally not folded — matches the legacy
        // Fold.cpp templates, which key off isSignedInteger()/isUnsignedInteger().
    }
    return {};
}

// -------------------------------------------------------------------------
// Binary driver — comparison (numeric in, bool out)
// -------------------------------------------------------------------------

static Attribute foldScalarCmpBinary(ScalarConstantFoldable op, Attribute lhs, Attribute rhs, Type resultType,
                                     Location loc) {
    if (isFloatAttrPair(lhs, rhs)) {
        auto [l, r] = promoteWidth(llvm::cast<FloatAttr>(lhs), llvm::cast<FloatAttr>(rhs), loc);
        if (auto v = op.foldScalarCmpFloat(l.getValue(), r.getValue()))
            return IntegerAttr::getChecked(loc, resultType, *v);
    }
    if (isIntAttrPair(lhs, rhs)) {
        auto [l, r] = promoteWidth(llvm::cast<IntegerAttr>(lhs), llvm::cast<IntegerAttr>(rhs), loc);
        Type argTy = l.getType();
        // Comparison signedness follows the *input* type, unlike arithmetic
        // where the result type carries it.
        if (auto intTy = llvm::dyn_cast<IntegerType>(argTy); intTy && intTy.isUnsigned()) {
            if (auto v = op.foldScalarCmpUInt(l.getValue(), r.getValue()))
                return IntegerAttr::getChecked(loc, resultType, *v);
        } else {
            if (auto v = op.foldScalarCmpSInt(l.getValue(), r.getValue()))
                return IntegerAttr::getChecked(loc, resultType, *v);
        }
    }
    return {};
}

// -------------------------------------------------------------------------
// Entry point
// -------------------------------------------------------------------------

Attribute foldScalarOp(ScalarConstantFoldable op, ArrayRef<Attribute> operands, Type resultType, Location loc) {
    if (operands.empty())
        return {};

    if (operands.size() == 1)
        return foldScalarUnary(op, operands[0], resultType, loc);

    if (operands.size() != 2)
        return {}; // Ternary and higher not modelled by this interface.

    if (!operands[0] || !operands[1])
        return {};

    // Dispatch heuristic: an i1 result usually means a comparison (numeric in,
    // bool out), but bool-typed logical ops (EwAndOp/EwOrOp/EwXorOp on `bool`
    // operands) also produce i1. Both-bool-operand pairs go through the
    // arithmetic path so the op's `foldScalarBool` gets called.
    const bool isCmp = resultType.isSignlessInteger(1) && !isBoolAttrPair(operands[0], operands[1]);
    return isCmp ? foldScalarCmpBinary(op, operands[0], operands[1], resultType, loc)
                 : foldScalarArithBinary(op, operands[0], operands[1], resultType, loc);
}

} // namespace mlir::daphne
