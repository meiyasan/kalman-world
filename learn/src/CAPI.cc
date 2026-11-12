/**
 * *********************************************
 *
 * \file CAPI.cc
 * \brief Implementation of the C interface of libKalmanLearn
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Learn/CAPI.h"
#include "Kalman/Learn/Solvers/Vorticity.h"

#include <exception>
#include <memory>
#include <stdexcept>
#include <string>

using Kalman::Learn::Solvers::Vorticity;

struct kalman_vorticity
{
    std::unique_ptr<Vorticity> solver;
    std::size_t size = 0;
    std::string device;
};

namespace
{
    thread_local std::string tError;

    template <typename F>
    int Guard(F &&body)
    {
        try { tError.clear(); return body(); }
        catch (const std::exception &e) { tError = e.what(); return -1; }
        catch (...) { tError = "unknown C++ exception"; return -1; }
    }

    std::vector<double> Copy(const double *data, std::size_t size, const char *what)
    {
        if (!data) throw std::invalid_argument(std::string("null ") + what);
        return std::vector<double>(data, data + size);
    }
}

const char *kalman_learn_last_error(void) { return tError.c_str(); }

int kalman_learn_has_torch(void) { return Vorticity::HasRuntime() ? 1 : 0; }

kalman_vorticity_t *kalman_vorticity_create(int ny, int nx, const double *kx, const double *ky,
                                            const double *k2, const double *k2inv, const double *dealias,
                                            double nu, double beta,
                                            const double *forcing_re, const double *forcing_im,
                                            const char *device)
{
    try
    {
        tError.clear();
        if (ny <= 0 || nx <= 0) throw std::invalid_argument("ny and nx must be positive");
        const std::size_t size = static_cast<std::size_t>(ny) * static_cast<std::size_t>(nx);
        Vorticity::Grid grid;
        grid.ny = static_cast<std::size_t>(ny);
        grid.nx = static_cast<std::size_t>(nx);
        grid.kx = Copy(kx, size, "kx");
        grid.ky = Copy(ky, size, "ky");
        grid.k2 = Copy(k2, size, "k2");
        grid.k2inv = Copy(k2inv, size, "k2inv");
        grid.dealias = Copy(dealias, size, "dealias");
        std::vector<double> fre, fim;
        if (forcing_re && forcing_im)
        {
            fre.assign(forcing_re, forcing_re + size);
            fim.assign(forcing_im, forcing_im + size);
        }
        auto handle = std::make_unique<kalman_vorticity>();
        handle->solver = std::make_unique<Vorticity>(grid, nu, beta, fre, fim, device ? device : "cpu");
        handle->size = size;
        handle->device = handle->solver->Device();
        return handle.release();
    }
    catch (const std::exception &e) { tError = e.what(); return nullptr; }
    catch (...) { tError = "unknown C++ exception"; return nullptr; }
}

void kalman_vorticity_destroy(kalman_vorticity_t *solver) { delete solver; }

const char *kalman_vorticity_device(const kalman_vorticity_t *solver)
{
    return solver ? solver->device.c_str() : nullptr;
}

int kalman_vorticity_advance(const kalman_vorticity_t *solver, const double *w, int batch, int nsteps,
                             double dt, int stages, const double *c, const double *A, const double *b,
                             double *out)
{
    return Guard([&] {
        if (!solver || !solver->solver) throw std::invalid_argument("null kalman_vorticity_t handle");
        if (batch <= 0 || nsteps < 0 || stages <= 0 || !c || !A || !b) throw std::invalid_argument("bad arguments");
        Vorticity::Tableau tableau;
        const std::size_t s = static_cast<std::size_t>(stages);
        tableau.c.assign(c, c + s);
        tableau.A.assign(A, A + s * s);
        tableau.b.assign(b, b + s);
        solver->solver->Advance(w, static_cast<std::size_t>(batch), static_cast<std::size_t>(nsteps), dt, tableau, out);
        return 0;
    });
}
