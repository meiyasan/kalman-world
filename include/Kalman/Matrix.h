/**
 * *********************************************
 *
 * \file Matrix.h
 * \brief Minimal dense vector and matrix types
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details libKalman has **no required dependency**: not ROOT, not Eigen, not
 * Torch. These two types are why. They are deliberately small and dumb —
 * row-major `double` storage, no expression templates, no aliasing analysis —
 * because the matrices a Kalman filter actually multiplies are tiny (state
 * dimensions of 2 to ~16), and at that size the cost is dominated by dispatch
 * and allocation rather than by arithmetic.
 *
 * That is also why the optional Torch backend does not replace them. A Torch op
 * dispatch costs more than a whole 6x6 predict/update; tensors earn their keep
 * on ensembles and fields, not here.
 *
 * *********************************************
 */

#ifndef KALMAN_MATRIX_H
#define KALMAN_MATRIX_H

#include <cstddef>
#include <initializer_list>
#include <iosfwd>
#include <vector>

namespace Kalman
{
    /**
     * \class Kalman::Vector
     * \brief A dense column vector of `double`.
     */
    class Vector
    {
        private:
            std::vector<double> fData;

        public:
            Vector() = default;
            explicit Vector(std::size_t n, double fill = 0.);
            Vector(std::initializer_list<double> values);

            static Vector Zero(std::size_t n) { return Vector(n, 0.); }

            std::size_t Size() const { return fData.size(); }
            bool Empty() const { return fData.empty(); }
            void Resize(std::size_t n, double fill = 0.);

            double &operator()(std::size_t i) { return fData[i]; }
            double operator()(std::size_t i) const { return fData[i]; }

            double *Data() { return fData.data(); }
            const double *Data() const { return fData.data(); }

            std::vector<double>::iterator begin() { return fData.begin(); }
            std::vector<double>::iterator end() { return fData.end(); }
            std::vector<double>::const_iterator begin() const { return fData.begin(); }
            std::vector<double>::const_iterator end() const { return fData.end(); }

            Vector &operator+=(const Vector &other);
            Vector &operator-=(const Vector &other);
            Vector &operator*=(double s);

            /// True when no element is NaN or infinite.
            bool AllFinite() const;
            /// True when no element is NaN (infinities are left alone).
            bool AnyNaN() const;

            void Print(std::ostream &os) const;
    };

    /**
     * \class Kalman::Matrix
     * \brief A dense row-major matrix of `double`.
     */
    class Matrix
    {
        private:
            std::size_t fRows = 0;
            std::size_t fCols = 0;
            std::vector<double> fData;

        public:
            Matrix() = default;
            Matrix(std::size_t rows, std::size_t cols, double fill = 0.);

            static Matrix Identity(std::size_t n);
            static Matrix Zero(std::size_t rows, std::size_t cols) { return Matrix(rows, cols, 0.); }
            /// Square matrix with \p diagonal on the diagonal and zeros elsewhere.
            static Matrix Diagonal(const Vector &diagonal);

            std::size_t Rows() const { return fRows; }
            std::size_t Cols() const { return fCols; }
            bool Empty() const { return fData.empty(); }
            bool IsSquare() const { return fRows == fCols; }
            void Resize(std::size_t rows, std::size_t cols, double fill = 0.);

            double &operator()(std::size_t i, std::size_t j) { return fData[i * fCols + j]; }
            double operator()(std::size_t i, std::size_t j) const { return fData[i * fCols + j]; }

            double *Data() { return fData.data(); }
            const double *Data() const { return fData.data(); }

            Matrix Transpose() const;

            Matrix &operator+=(const Matrix &other);
            Matrix &operator-=(const Matrix &other);
            Matrix &operator*=(double s);

            bool AllFinite() const;

            void Print(std::ostream &os) const;
    };

    Vector operator+(Vector a, const Vector &b);
    Vector operator-(Vector a, const Vector &b);
    Vector operator*(double s, Vector v);
    Vector operator*(Vector v, double s);

    Matrix operator+(Matrix a, const Matrix &b);
    Matrix operator-(Matrix a, const Matrix &b);
    Matrix operator*(double s, Matrix m);

    /// Matrix product `A B`.
    Matrix operator*(const Matrix &a, const Matrix &b);
    /// Matrix-vector product `A x`.
    Vector operator*(const Matrix &a, const Vector &x);

    std::ostream &operator<<(std::ostream &os, const Vector &v);
    std::ostream &operator<<(std::ostream &os, const Matrix &m);
}

#endif // KALMAN_MATRIX_H
