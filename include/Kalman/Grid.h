/**
 * *********************************************
 *
 * \file Grid.h
 * \brief A dense 2-D field of any scalar type — the state a solver marches.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Grid
 * \brief `(rows, cols)` cells of `T`, row-major, with the arithmetic a Runge–Kutta
 * step needs and nothing else.
 *
 * \details Kalman::Matrix is the small dense *algebra* type of the filters;
 * this is the *field* type of the solvers, which never multiply two grids but
 * add them, scale them and hand them to an FFT. `Grid<double>` is a real field
 * (`Kalman::Field`), `Grid<std::complex<double>>` its spectrum
 * (`Kalman::SpectralField`). Both feed the same generic RK step in
 * Kalman/Integrator.h, so any explicit integrator drives any solver.
 *
 * *********************************************
 */

#ifndef KALMAN_GRID_H
#define KALMAN_GRID_H

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace Kalman
{
    template <typename T>
    class Grid
    {
        private:
            std::size_t fRows = 0;
            std::size_t fCols = 0;
            std::vector<T> fData;

            void CheckSame(const Grid &other, const char *op) const
            {
                if (other.fRows != fRows || other.fCols != fCols)
                    throw std::runtime_error(std::string("Kalman::Grid: shape mismatch in ") + op);
            }

        public:
            Grid() = default;
            Grid(std::size_t rows, std::size_t cols, T fill = T())
                : fRows(rows), fCols(cols), fData(rows * cols, fill) {}

            std::size_t Rows() const { return fRows; }
            std::size_t Cols() const { return fCols; }
            std::size_t Size() const { return fData.size(); }
            bool Empty() const { return fData.empty(); }

            T &operator()(std::size_t i, std::size_t j) { return fData[i * fCols + j]; }
            const T &operator()(std::size_t i, std::size_t j) const { return fData[i * fCols + j]; }

            T *Data() { return fData.data(); }
            const T *Data() const { return fData.data(); }
            std::vector<T> &Raw() { return fData; }
            const std::vector<T> &Raw() const { return fData; }

            Grid &operator+=(const Grid &o) { CheckSame(o, "+="); for (std::size_t i = 0; i < fData.size(); ++i) fData[i] += o.fData[i]; return *this; }
            Grid &operator-=(const Grid &o) { CheckSame(o, "-="); for (std::size_t i = 0; i < fData.size(); ++i) fData[i] -= o.fData[i]; return *this; }
            Grid &operator*=(double s) { for (T &v : fData) v *= s; return *this; }

            /// Elementwise product (a mask, a taper, a wavenumber array).
            Grid &Multiply(const Grid &o) { CheckSame(o, "Multiply"); for (std::size_t i = 0; i < fData.size(); ++i) fData[i] *= o.fData[i]; return *this; }

            friend Grid operator+(Grid a, const Grid &b) { a += b; return a; }
            friend Grid operator-(Grid a, const Grid &b) { a -= b; return a; }
            friend Grid operator*(double s, Grid g) { g *= s; return g; }
            friend Grid operator*(Grid g, double s) { g *= s; return g; }
    };

    using Field = Grid<double>;
    using SpectralField = Grid<std::complex<double>>;

    /// The real part of a spectrum, cell by cell.
    inline Field Real(const SpectralField &s)
    {
        Field out(s.Rows(), s.Cols());
        for (std::size_t i = 0; i < s.Size(); ++i) out.Data()[i] = s.Data()[i].real();
        return out;
    }

    /// A real field promoted to a spectrum with zero imaginary part.
    inline SpectralField Complex(const Field &f)
    {
        SpectralField out(f.Rows(), f.Cols());
        for (std::size_t i = 0; i < f.Size(); ++i) out.Data()[i] = std::complex<double>(f.Data()[i], 0.);
        return out;
    }
}

#endif // KALMAN_GRID_H
