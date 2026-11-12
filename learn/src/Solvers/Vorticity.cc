/**
 * *********************************************
 *
 * \file Vorticity.cc
 * \brief Implementation of the Kalman::Learn::Solvers::Vorticity class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * libTorch is reached from this file and from Routers/NN/Torch.cc, nowhere else.
 */

#include "Kalman/Learn/Solvers/Vorticity.h"

#include <cstring>
#include <stdexcept>

// Set by learn/CMakeLists.txt when the runtime was found.
#ifndef KALMAN_WITH_TORCH
#  define KALMAN_WITH_TORCH 0
#endif

#if KALMAN_WITH_TORCH
#  include <torch/torch.h>
#endif

namespace Kalman::Learn::Solvers
{
#if KALMAN_WITH_TORCH

    struct Vorticity::Impl
    {
        torch::Device device{torch::kCPU};
        torch::Dtype real = torch::kFloat64;
        torch::Dtype complex = torch::kComplexDouble;
        std::string name = "cpu";
        long ny = 0, nx = 0;
        double nu = 0., beta = 0.;
        bool forced = false;

        torch::Tensor ikx, iky;        // i kx, i ky (complex)
        torch::Tensor k2, k2inv, dealias, forcing;

        torch::Tensor Real(const std::vector<double> &v) const
        {
            // Precision first, device second: Metal refuses a double-precision tensor.
            return torch::tensor(v, torch::kFloat64).reshape({ny, nx}).to(real).to(device);
        }

        // -u.grad(w) - beta v + nu lap(w) + F, in Fourier space.
        torch::Tensor Rhs(const torch::Tensor &wh) const
        {
            namespace fft = torch::fft;
            const torch::Tensor psih = wh * k2inv;
            const torch::Tensor u = torch::real(fft::ifft2(iky * psih));
            const torch::Tensor v = torch::real(fft::ifft2(-ikx * psih));
            const torch::Tensor wx = torch::real(fft::ifft2(ikx * wh));
            const torch::Tensor wy = torch::real(fft::ifft2(iky * wh));
            torch::Tensor product = u * wx + v * wy;
            if (beta != 0.) product = product + beta * v;
            torch::Tensor out = -(fft::fft2(product.to(complex)) * dealias) - nu * k2 * wh;
            if (forced) out = out + forcing;
            return out;
        }
    };

    bool Vorticity::HasRuntime() { return true; }

    Vorticity::Vorticity(const Grid &grid, double nu, double beta,
                         const std::vector<double> &forcingRe, const std::vector<double> &forcingIm,
                         const std::string &device)
        : fImpl(std::make_shared<Impl>())
    {
        const std::size_t size = grid.ny * grid.nx;
        if (size == 0) throw std::invalid_argument("Learn::Solvers::Vorticity: an empty grid.");
        for (const std::vector<double> *v : {&grid.kx, &grid.ky, &grid.k2, &grid.k2inv, &grid.dealias})
            if (v->size() != size)
                throw std::invalid_argument("Learn::Solvers::Vorticity: every grid operator must hold ny * nx values.");
        if (!forcingRe.empty() && (forcingRe.size() != size || forcingIm.size() != size))
            throw std::invalid_argument("Learn::Solvers::Vorticity: the forcing must hold ny * nx values.");

        Impl &m = *fImpl;
        m.ny = static_cast<long>(grid.ny);
        m.nx = static_cast<long>(grid.nx);
        m.nu = nu;
        m.beta = beta;

        if (device == "cuda" && torch::cuda::is_available())
        {
            m.device = torch::Device(torch::kCUDA);
            m.name = "cuda";
        }
        else if (device == "mps" && torch::mps::is_available())
        {
            m.device = torch::Device(torch::kMPS);
            m.name = "mps";
            m.real = torch::kFloat32;              // Metal has no double precision
            m.complex = torch::kComplexFloat;
        }

        const torch::Tensor kx = m.Real(grid.kx), ky = m.Real(grid.ky);
        const torch::Tensor zero = torch::zeros_like(kx);
        m.ikx = torch::complex(zero, kx);
        m.iky = torch::complex(zero, ky);
        m.k2 = m.Real(grid.k2);
        m.k2inv = m.Real(grid.k2inv);
        m.dealias = m.Real(grid.dealias);
        m.forced = !forcingRe.empty();
        if (m.forced) m.forcing = torch::complex(m.Real(forcingRe), m.Real(forcingIm));
    }

    std::string Vorticity::Device() const { return fImpl->name; }

    void Vorticity::Advance(const double *w, std::size_t batch, std::size_t nsteps, double dt,
                            const Tableau &tableau, double *out) const
    {
        if (!w || !out) throw std::invalid_argument("Learn::Solvers::Vorticity::Advance: null buffer.");
        const std::size_t s = tableau.Stages();
        if (s == 0 || tableau.c.size() != s || tableau.A.size() != s * s)
            throw std::invalid_argument("Learn::Solvers::Vorticity::Advance: malformed tableau.");

        const Impl &m = *fImpl;
        torch::NoGradGuard noGradients;
        const long B = static_cast<long>(batch);

        torch::Tensor field = torch::from_blob(const_cast<double *>(w), {B, m.ny, m.nx}, torch::kFloat64)
                                  .to(m.real).to(m.device);
        torch::Tensor wh = torch::fft::fft2(field);

        std::vector<torch::Tensor> k(s);
        for (std::size_t step = 0; step < nsteps; ++step)
        {
            for (std::size_t i = 0; i < s; ++i)
            {
                torch::Tensor stage = wh;
                if (i > 0)
                {
                    torch::Tensor acc = tableau.A[i * s] * k[0];
                    for (std::size_t j = 1; j < i; ++j) acc = acc + tableau.A[i * s + j] * k[j];
                    stage = wh + dt * acc;
                }
                k[i] = m.Rhs(stage);
            }
            torch::Tensor sum = tableau.b[0] * k[0];
            for (std::size_t i = 1; i < s; ++i) sum = sum + tableau.b[i] * k[i];
            wh = wh + dt * sum;
        }

        const torch::Tensor result = torch::real(torch::fft::ifft2(wh)).to(torch::kCPU).to(torch::kFloat64).contiguous();
        std::memcpy(out, result.data_ptr<double>(), sizeof(double) * batch * static_cast<std::size_t>(m.ny * m.nx));
    }

#else  // --- no libTorch: the class exists and says so ---------------------------

    struct Vorticity::Impl {};

    bool Vorticity::HasRuntime() { return false; }

    Vorticity::Vorticity(const Grid &, double, double, const std::vector<double> &,
                         const std::vector<double> &, const std::string &)
    {
        throw std::runtime_error("Kalman::Learn::Solvers::Vorticity: this build of libKalmanLearn has no "
                                 "libTorch (configure with -DKALMAN_USE_TORCH=ON).");
    }

    std::string Vorticity::Device() const { return "none"; }

    void Vorticity::Advance(const double *, std::size_t, std::size_t, double, const Tableau &, double *) const
    {
        throw std::runtime_error("Kalman::Learn::Solvers::Vorticity: no libTorch in this build.");
    }

#endif
}
