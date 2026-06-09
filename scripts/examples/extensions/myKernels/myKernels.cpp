#include <runtime/local/datastructures/DataObjectFactory.h>
#include <runtime/local/datastructures/DenseMatrix.h>

#include <cmath>
#include <immintrin.h> // for the SIMD-enabled kernels
#include <iostream>
#include <stdexcept>

class DaphneContext;

extern "C" {
// **************************************
// Example of a kernel returning a scalar
// **************************************

// Custom sequential sum-kernel.
void mySumSeq(float *res, const DenseMatrix<float> *arg, int kernelId, DaphneContext *ctx) {
    std::cerr << "hello from mySumSeq()" << std::endl;

    const float *valuesArg = arg->getValues();
    *res = 0;
    for (size_t r = 0; r < arg->getNumRows(); r++) {
        for (size_t c = 0; c < arg->getNumCols(); c++)
            *res += valuesArg[c];
        valuesArg += arg->getRowSkip();
    }
}

// Custom SIMD-enabled sum-kernel.
void mySumSIMD(float *res, const DenseMatrix<float> *arg, int kernelId, DaphneContext *ctx) {
    std::cerr << "hello from mySumSIMD()" << std::endl;

    // Validation.
    const size_t numCells = arg->getNumRows() * arg->getNumCols();
    if (numCells % 8)
        throw std::runtime_error("for simplicity, the number of cells must be a multiple of 8");
    if (arg->getNumCols() != arg->getRowSkip())
        throw std::runtime_error("for simplicity, the argument must not be a column segment of another matrix");

    // SIMD accumulation (8x f32).
    const float *valuesArg = arg->getValues();
    __m256 acc = _mm256_setzero_ps();
    for (size_t i = 0; i < numCells / 8; i++) {
        acc = _mm256_add_ps(acc, _mm256_loadu_ps(valuesArg));
        valuesArg += 8;
    }

    // Summation of accumulator elements.
    *res = (reinterpret_cast<float *>(&acc))[0] + (reinterpret_cast<float *>(&acc))[1] +
           (reinterpret_cast<float *>(&acc))[2] + (reinterpret_cast<float *>(&acc))[3] +
           (reinterpret_cast<float *>(&acc))[4] + (reinterpret_cast<float *>(&acc))[5] +
           (reinterpret_cast<float *>(&acc))[6] + (reinterpret_cast<float *>(&acc))[7];
}

// **********************************************************************
// Example of a kernel returning a data object (DenseMatrix in this case)
// **********************************************************************

// Custom sequential squareroot-kernel.
void mySqrtSeq(DenseMatrix<float> **res_, const DenseMatrix<float> *arg, int kernelId, DaphneContext *ctx) {
    std::cout << "hello from mySqrtSeq()" << std::endl;

    // New variable for more convenient use (no double pointer).
    DenseMatrix<float> *&res = *res_;

    if (res == nullptr)
        res = DataObjectFactory::create<DenseMatrix<float>>(arg->getNumRows(), arg->getNumCols(), false);

    const float *valuesArg = arg->getValues();
    float *valuesRes = res->getValues();

    for (size_t r = 0; r < arg->getNumRows(); r++) {
        for (size_t c = 0; c < arg->getNumCols(); c++)
            valuesRes[c] = std::sqrt(valuesArg[c]);
        valuesArg += arg->getRowSkip();
        valuesRes += res->getRowSkip();
    }
}

// Custom SIMD-enabled squareroot-kernel.
void mySqrtSIMD(DenseMatrix<float> **res_, const DenseMatrix<float> *arg, int kernelId, DaphneContext *ctx) {
    std::cout << "hello from mySqrtSIMD()" << std::endl;

    // Validation.
    if (arg->getNumCols() % 8)
        throw std::runtime_error("for simplicity, the number of columns must be a multiple of 8");
    if (arg->getNumCols() != arg->getRowSkip())
        throw std::runtime_error("for simplicity, the argument must not be a column segment of another matrix");

    // New variable for more convenient use (no double pointer).
    DenseMatrix<float> *&res = *res_;

    if (res == nullptr)
        res = DataObjectFactory::create<DenseMatrix<float>>(arg->getNumRows(), arg->getNumCols(), false);

    const float *valuesArg = arg->getValues();
    float *valuesRes = res->getValues();

    // SIMD processing.
    for (size_t r = 0; r < arg->getNumRows(); r++)
        for (size_t c = 0; c < arg->getNumCols() / 8; c++) {
            _mm256_storeu_ps(valuesRes, _mm256_sqrt_ps(_mm256_loadu_ps(valuesArg)));
            valuesArg += 8;
            valuesRes += 8;
        }
}
}