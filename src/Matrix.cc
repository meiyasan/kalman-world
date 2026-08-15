/**
 * *********************************************
 *
 * \file Matrix.cc
 * \brief Implementation of the dense vector and matrix types
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Matrix.h"
#include "Kalman/Config.h"

#include <cmath>
#include <ostream>
#include <sstream>
#include <stdexcept>

#if KALMAN_WITH_BLAS
#  if defined(__APPLE__)
// Accelerate deprecated the classic 32-bit CBLAS prototypes in macOS 13.3; this
// selects the current headers, which keep the same `int` signatures we call with.
#    ifndef ACCELERATE_NEW_LAPACK
#      define ACCELERATE_NEW_LAPACK
#    endif
#    include <Accelerate/Accelerate.h>
#  else
#    include <cblas.h>
#  endif
#endif

namespace Kalman
{
    namespace
    {
        void RequireSame(std::size_t a, std::size_t b, const char *what)
        {
            if (a != b)
            {
                std::ostringstream os;
                os << "Kalman: " << what << " (" << a << " vs " << b << ").";
                throw std::invalid_argument(os.str());
            }
        }
    }

    // --- Vector --------------------------------------------------------------

    Vector::Vector(std::size_t n, double fill) : fData(n, fill) {}

    Vector::Vector(std::initializer_list<double> values) : fData(values) {}

    void Vector::Resize(std::size_t n, double fill) { fData.assign(n, fill); }

    Vector &Vector::operator+=(const Vector &other)
    {
        RequireSame(Size(), other.Size(), "vector sizes differ in +=");
        for (std::size_t i = 0; i < fData.size(); ++i) fData[i] += other(i);
        return *this;
    }

    Vector &Vector::operator-=(const Vector &other)
    {
        RequireSame(Size(), other.Size(), "vector sizes differ in -=");
        for (std::size_t i = 0; i < fData.size(); ++i) fData[i] -= other(i);
        return *this;
    }

    Vector &Vector::operator*=(double s)
    {
        for (double &v : fData) v *= s;
        return *this;
    }

    bool Vector::AllFinite() const
    {
        for (double v : fData) if (!std::isfinite(v)) return false;
        return true;
    }

    bool Vector::AnyNaN() const
    {
        for (double v : fData) if (std::isnan(v)) return true;
        return false;
    }

    void Vector::Print(std::ostream &os) const
    {
        os << "[";
        for (std::size_t i = 0; i < fData.size(); ++i)
            os << (i ? ", " : "") << fData[i];
        os << "]";
    }

    // --- Matrix --------------------------------------------------------------

    Matrix::Matrix(std::size_t rows, std::size_t cols, double fill)
        : fRows(rows), fCols(cols), fData(rows * cols, fill) {}

    Matrix Matrix::Identity(std::size_t n)
    {
        Matrix m(n, n, 0.);
        for (std::size_t i = 0; i < n; ++i) m(i, i) = 1.;
        return m;
    }

    Matrix Matrix::Diagonal(const Vector &diagonal)
    {
        const std::size_t n = diagonal.Size();
        Matrix m(n, n, 0.);
        for (std::size_t i = 0; i < n; ++i) m(i, i) = diagonal(i);
        return m;
    }

    void Matrix::Resize(std::size_t rows, std::size_t cols, double fill)
    {
        fRows = rows;
        fCols = cols;
        fData.assign(rows * cols, fill);
    }

    Matrix Matrix::Transpose() const
    {
        Matrix out(fCols, fRows);
        for (std::size_t i = 0; i < fRows; ++i)
            for (std::size_t j = 0; j < fCols; ++j)
                out(j, i) = (*this)(i, j);
        return out;
    }

    Matrix &Matrix::operator+=(const Matrix &other)
    {
        RequireSame(fRows, other.Rows(), "matrix rows differ in +=");
        RequireSame(fCols, other.Cols(), "matrix columns differ in +=");
        for (std::size_t i = 0; i < fData.size(); ++i) fData[i] += other.Data()[i];
        return *this;
    }

    Matrix &Matrix::operator-=(const Matrix &other)
    {
        RequireSame(fRows, other.Rows(), "matrix rows differ in -=");
        RequireSame(fCols, other.Cols(), "matrix columns differ in -=");
        for (std::size_t i = 0; i < fData.size(); ++i) fData[i] -= other.Data()[i];
        return *this;
    }

    Matrix &Matrix::operator*=(double s)
    {
        for (double &v : fData) v *= s;
        return *this;
    }

    bool Matrix::AllFinite() const
    {
        for (double v : fData) if (!std::isfinite(v)) return false;
        return true;
    }

    void Matrix::Print(std::ostream &os) const
    {
        for (std::size_t i = 0; i < fRows; ++i)
        {
            os << (i ? "\n [" : "[[");
            for (std::size_t j = 0; j < fCols; ++j)
                os << (j ? ", " : "") << (*this)(i, j);
            os << "]";
        }
        os << "]";
    }

    // --- free operators ------------------------------------------------------

    Vector operator+(Vector a, const Vector &b) { a += b; return a; }
    Vector operator-(Vector a, const Vector &b) { a -= b; return a; }
    Vector operator*(double s, Vector v) { v *= s; return v; }
    Vector operator*(Vector v, double s) { v *= s; return v; }

    Matrix operator+(Matrix a, const Matrix &b) { a += b; return a; }
    Matrix operator-(Matrix a, const Matrix &b) { a -= b; return a; }
    Matrix operator*(double s, Matrix m) { m *= s; return m; }

    Matrix operator*(const Matrix &a, const Matrix &b)
    {
        RequireSame(a.Cols(), b.Rows(), "inner dimensions differ in matrix product");
        Matrix out(a.Rows(), b.Cols(), 0.);

#if KALMAN_WITH_BLAS
        // Measured crossover on this class of hardware (double, square, hot cache):
        // the hand-rolled kernel wins below ~6 because a BLAS call costs ~20 ns of
        // argument handling, and loses badly above it (4x at n=16, 17x at n=32).
        // Dispatching on size takes the better of the two rather than betting on one.
        if (a.Rows() >= KALMAN_BLAS_MIN_DIM || a.Cols() >= KALMAN_BLAS_MIN_DIM ||
            b.Cols() >= KALMAN_BLAS_MIN_DIM)
        {
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        static_cast<int>(a.Rows()), static_cast<int>(b.Cols()),
                        static_cast<int>(a.Cols()), 1.0,
                        a.Data(), static_cast<int>(a.Cols()),
                        b.Data(), static_cast<int>(b.Cols()), 0.0,
                        out.Data(), static_cast<int>(b.Cols()));
            return out;
        }
#endif

        for (std::size_t i = 0; i < a.Rows(); ++i)
            for (std::size_t k = 0; k < a.Cols(); ++k)
            {
                const double aik = a(i, k);
                if (aik == 0.) continue;
                for (std::size_t j = 0; j < b.Cols(); ++j)
                    out(i, j) += aik * b(k, j);
            }
        return out;
    }

    Vector operator*(const Matrix &a, const Vector &x)
    {
        RequireSame(a.Cols(), x.Size(), "dimensions differ in matrix-vector product");
        Vector out(a.Rows(), 0.);
        for (std::size_t i = 0; i < a.Rows(); ++i)
        {
            double sum = 0.;
            for (std::size_t j = 0; j < a.Cols(); ++j) sum += a(i, j) * x(j);
            out(i) = sum;
        }
        return out;
    }

    std::ostream &operator<<(std::ostream &os, const Vector &v) { v.Print(os); return os; }
    std::ostream &operator<<(std::ostream &os, const Matrix &m) { m.Print(os); return os; }
}
