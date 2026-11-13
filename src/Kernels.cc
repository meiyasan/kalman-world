/**
 * *********************************************
 *
 * \file Kernels.cc
 * \brief The C kernels behind the Python engines, implemented by the C++ classes
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \details Each entry point builds the C++ objects the arithmetic belongs to and
 * runs them: a linear pass is a Kalman::Filter over Engines::Linear on a
 * Models::Linear assembled from the matrices, an update is Utils::Correct, the RTS
 * sweep is Engine::Smooth, the sigma-point moments are Utils::WeightedOuterSum,
 * resampling is Utils::SystematicResample. That is the point of the file: the
 * Python package used to carry its own C copy of this arithmetic (`native/kalman.c`),
 * and a second implementation of a Kalman step is a second place for it to be wrong.
 *
 * *********************************************
 */

#include "Kalman/CAPI.h"

#include "Kalman/Engines/Linear.h"
#include "Kalman/Filter.h"
#include "Kalman/Models/Linear.h"
#include "Kalman/Utils.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// The error slot shared with CAPI.cc (thread-local there); the kernels set it
// through the two hooks below so a void function can still leave a reason.
namespace Kalman::CAPI
{
    void SetError(const std::string &what);
    void ClearError();
}

namespace
{
    using Kalman::Matrix;
    using Kalman::Vector;

    Vector Vec(const double *p, std::size_t n)
    {
        Vector v(n);
        std::memcpy(v.Data(), p, n * sizeof(double));
        return v;
    }

    Matrix Mat(const double *p, std::size_t r, std::size_t c)
    {
        Matrix M(r, c);
        std::memcpy(M.Data(), p, r * c * sizeof(double));
        return M;
    }

    void Out(const Vector &v, double *p) { std::memcpy(p, v.Data(), v.Size() * sizeof(double)); }
    void Out(const Matrix &M, double *p) { std::memcpy(p, M.Data(), M.Rows() * M.Cols() * sizeof(double)); }

    void Fill(double *p, std::size_t count, double value)
    {
        if (p) for (std::size_t i = 0; i < count; ++i) p[i] = value;
    }

    template <typename F>
    int Guard(F &&body, std::initializer_list<std::pair<double *, std::size_t>> outputs)
    {
        try
        {
            Kalman::CAPI::ClearError();
            return body();
        }
        catch (const std::exception &e) { Kalman::CAPI::SetError(e.what()); }
        catch (...) { Kalman::CAPI::SetError("unknown C++ exception"); }
        for (const auto &o : outputs) Fill(o.first, o.second, std::numeric_limits<double>::quiet_NaN());
        return -1;
    }

    // A linear model assembled from the matrices a kernel is handed; a NULL B means
    // the control does not enter (l = 0), a NULL L means identity noise coupling.
    Kalman::Models::Linear Assemble(int n, int m, int l, const double *F, const double *H,
                                    const double *Q, const double *R, const double *B, const double *L)
    {
        const std::size_t nn = static_cast<std::size_t>(n), mm = static_cast<std::size_t>(m);
        const std::size_t ll = (B && l > 0) ? static_cast<std::size_t>(l) : 0;
        Kalman::Models::Linear model("kernel", nn, mm, ll);
        if (F) model.SetF(Mat(F, nn, nn));
        if (H) model.SetH(Mat(H, mm, nn));
        if (Q) model.SetQ(Mat(Q, nn, nn));
        if (R) model.SetR(Mat(R, mm, mm));
        if (ll) model.SetB(Mat(B, nn, ll));
        if (L) model.SetL(Mat(L, nn, nn));
        return model;
    }
}

extern "C" {

void kf_predict(int n, int l,
                const double *F, const double *x, const double *P,
                const double *B, const double *u,
                const double *L, const double *Q,
                double *x_out, double *P_out)
{
    const std::size_t nn = static_cast<std::size_t>(n);
    Guard([&] {
        // A one-row H keeps the model well-formed; the prediction never reads it.
        std::vector<double> H(nn, 0.);
        Kalman::Models::Linear model = Assemble(n, 1, l, F, H.data(), Q, nullptr, B, L);
        const Kalman::Engines::Linear engine;
        Vector uv;
        if (B && u && l > 0) uv = Vec(u, static_cast<std::size_t>(l));
        Vector xk;
        Matrix Pk, Fk;
        engine.Predict(model, Vec(x, nn), Mat(P, nn, nn), uv.Empty() ? nullptr : &uv, 0., xk, Pk, Fk);
        Out(xk, x_out);
        Out(Pk, P_out);
        return 0;
    }, {{x_out, nn}, {P_out, nn * nn}});
}

int kf_update(int n, int m,
              const double *H, const double *x, const double *P,
              const double *z, const double *R,
              double *x_out, double *P_out)
{
    const std::size_t nn = static_cast<std::size_t>(n), mm = static_cast<std::size_t>(m);
    return Guard([&] {
        Kalman::Models::Linear model = Assemble(n, m, 0, nullptr, H, nullptr, R, nullptr, nullptr);
        const Kalman::Engines::Linear engine;
        Vector xo;
        Matrix Po;
        const std::size_t used = engine.Update(model, Vec(x, nn), Mat(P, nn, nn), Vec(z, mm), 0., xo, Po);
        Out(xo, x_out);
        Out(Po, P_out);
        return static_cast<int>(used);
    }, {{x_out, nn}, {P_out, nn * nn}});
}

int kf_update_residual(int n, int m,
                        const double *H, const double *x, const double *P,
                        const double *y, const double *R,
                        double *x_out, double *P_out)
{
    const std::size_t nn = static_cast<std::size_t>(n), mm = static_cast<std::size_t>(m);
    return Guard([&] {
        const Vector yv = Vec(y, mm);
        const std::vector<std::size_t> keep = Kalman::Utils::ValidChannels(yv);
        const Vector xv = Vec(x, nn);
        const Matrix Pm = Mat(P, nn, nn);
        if (keep.empty())
        {
            Out(xv, x_out);
            Out(Pm, P_out);
            return 0;
        }
        Vector xo;
        Matrix Po;
        Kalman::Utils::Correct(Kalman::Utils::CompressRows(Mat(H, mm, nn), keep),
                               Kalman::Utils::CompressBoth(Mat(R, mm, mm), keep),
                               Kalman::Utils::Compress(yv, keep), xv, Pm, xo, Po);
        Out(xo, x_out);
        Out(Po, P_out);
        return static_cast<int>(keep.size());
    }, {{x_out, nn}, {P_out, nn * nn}});
}

void kf_forward_linear(int n, int m, int l, int N,
                        const double *F, const double *H,
                        const double *Q, const double *R,
                        const double *B, const double *L,
                        const double *zk, const double *uk,
                        const double *x0, const double *P0,
                        double *x_prior, double *P_prior,
                        double *x_post, double *P_post,
                        double *innovations)
{
    const std::size_t nn = static_cast<std::size_t>(n), mm = static_cast<std::size_t>(m);
    const std::size_t steps = N > 0 ? static_cast<std::size_t>(N) : 0;
    Guard([&] {
        if (N <= 0) throw std::invalid_argument("kf_forward_linear: N must be positive");
        Kalman::Models::Linear model = Assemble(n, m, l, F, H, Q, R, B, L);
        model.SetInitialState(Vec(x0, nn));
        model.SetInitialCovariance(Mat(P0, nn, nn));
        const std::size_t ll = model.NControl();

        std::vector<Vector> z(steps), u;
        for (std::size_t k = 0; k < steps; ++k) z[k] = Vec(zk + k * mm, mm);
        if (uk && ll > 0)
        {
            u.resize(steps);
            for (std::size_t k = 0; k < steps; ++k) u[k] = Vec(uk + k * ll, ll);
        }

        // The very filter a C++ program runs, on the very engine.
        Kalman::Engines::Linear engine;
        Kalman::Filter filter(&model, &engine);
        filter.Forward(z, u.empty() ? nullptr : &u);

        for (std::size_t k = 0; k < steps; ++k)
        {
            Out(filter.PriorStates()[k], x_prior + k * nn);
            Out(filter.PriorCovariances()[k], P_prior + k * nn * nn);
            Out(filter.PosteriorStates()[k], x_post + k * nn);
            Out(filter.PosteriorCovariances()[k], P_post + k * nn * nn);
            Out(filter.Innovations()[k], innovations + k * mm);
        }
        return 0;
    }, {{x_prior, steps * nn}, {P_prior, steps * nn * nn}, {x_post, steps * nn},
        {P_post, steps * nn * nn}, {innovations, steps * mm}});
}

void kf_backward_rts(int n, int N,
                      const double *x_post, const double *P_post,
                      const double *x_prior, const double *P_prior,
                      const double *F,
                      double *x_smooth, double *P_smooth)
{
    const std::size_t nn = static_cast<std::size_t>(n);
    const std::size_t steps = N > 0 ? static_cast<std::size_t>(N) : 0;
    Guard([&] {
        if (N <= 0) throw std::invalid_argument("kf_backward_rts: N must be positive");
        const Kalman::Engines::Linear engine;
        const Matrix Fm = Mat(F, nn, nn);
        // The last posterior is already the smoothed estimate; walk backwards from it.
        std::memcpy(x_smooth + (steps - 1) * nn, x_post + (steps - 1) * nn, nn * sizeof(double));
        std::memcpy(P_smooth + (steps - 1) * nn * nn, P_post + (steps - 1) * nn * nn, nn * nn * sizeof(double));
        for (std::size_t k = steps - 1; k-- > 0;)
        {
            Vector xs;
            Matrix Ps;
            engine.Smooth(Vec(x_post + k * nn, nn), Mat(P_post + k * nn * nn, nn, nn),
                          Vec(x_prior + (k + 1) * nn, nn), Mat(P_prior + (k + 1) * nn * nn, nn, nn),
                          Vec(x_smooth + (k + 1) * nn, nn), Mat(P_smooth + (k + 1) * nn * nn, nn, nn),
                          Fm, xs, Ps);
            Out(xs, x_smooth + k * nn);
            Out(Ps, P_smooth + k * nn * nn);
        }
        return 0;
    }, {{x_smooth, steps * nn}, {P_smooth, steps * nn * nn}});
}

void kf_backward_rts_steps(int n, int N,
                      const double *x_post, const double *P_post,
                      const double *x_prior, const double *P_prior,
                      const double *F,
                      double *x_smooth, double *P_smooth)
{
    const std::size_t nn = static_cast<std::size_t>(n);
    const std::size_t steps = N > 0 ? static_cast<std::size_t>(N) : 0;
    Guard([&] {
        if (N <= 0) throw std::invalid_argument("kf_backward_rts_steps: N must be positive");
        const Kalman::Engines::Linear engine;
        // The last posterior is already the smoothed estimate; walk backwards from it.
        std::memcpy(x_smooth + (steps - 1) * nn, x_post + (steps - 1) * nn, nn * sizeof(double));
        std::memcpy(P_smooth + (steps - 1) * nn * nn, P_post + (steps - 1) * nn * nn, nn * nn * sizeof(double));
        for (std::size_t k = steps - 1; k-- > 0;)
        {
            Vector xs;
            Matrix Ps;
            engine.Smooth(Vec(x_post + k * nn, nn), Mat(P_post + k * nn * nn, nn, nn),
                          Vec(x_prior + (k + 1) * nn, nn), Mat(P_prior + (k + 1) * nn * nn, nn, nn),
                          Vec(x_smooth + (k + 1) * nn, nn), Mat(P_smooth + (k + 1) * nn * nn, nn, nn),
                          Mat(F + (k + 1) * nn * nn, nn, nn), xs, Ps);
            Out(xs, x_smooth + k * nn);
            Out(Ps, P_smooth + k * nn * nn);
        }
        return 0;
    }, {{x_smooth, steps * nn}, {P_smooth, steps * nn * nn}});
}

void pf_weight_update(int n_particles, int m,
                       const double *z, const double *zp,
                       const double *Rinv, double *logw)
{
    const std::size_t np = static_cast<std::size_t>(n_particles), mm = static_cast<std::size_t>(m);
    Guard([&] {
        const Matrix Ri = Mat(Rinv, mm, mm);
        const Vector zv = Vec(z, mm);
        for (std::size_t i = 0; i < np; ++i)
        {
            Vector dz = zv;
            dz -= Vec(zp + i * mm, mm);
            logw[i] += -0.5 * Kalman::Utils::Mahalanobis(dz, Ri);
        }
        return 0;
    }, {});
}

void pf_systematic_resample(int n, const double *weights, double u0, int *indices)
{
    Guard([&] {
        if (n <= 0) throw std::invalid_argument("pf_systematic_resample: n must be positive");
        Kalman::Utils::SystematicResample(static_cast<std::size_t>(n), weights, u0, indices);
        return 0;
    }, {});
}

void ukf_weighted_outer_sum(int n_sigma, int n,
                             const double *Wc, const double *Y, const double *mean,
                             double *P_out)
{
    const std::size_t nn = static_cast<std::size_t>(n);
    Guard([&] {
        Kalman::Utils::WeightedOuterSum(static_cast<std::size_t>(n_sigma), nn, nn, Wc, Y, mean, Y, mean, P_out);
        return 0;
    }, {{P_out, nn * nn}});
}

void ukf_weighted_cross_sum(int n_sigma, int na, int nb,
                             const double *Wc, const double *A, const double *B,
                             double *C_out)
{
    const std::size_t a = static_cast<std::size_t>(na), b = static_cast<std::size_t>(nb);
    Guard([&] {
        // The rows are already centred, so the means are zero.
        const std::vector<double> za(a, 0.), zb(b, 0.);
        Kalman::Utils::WeightedOuterSum(static_cast<std::size_t>(n_sigma), a, b, Wc, A, za.data(), B, zb.data(), C_out);
        return 0;
    }, {{C_out, a * b}});
}

void diag_nis_batch(int N, int m,
                     const double *nu, const double *S, double *nis)
{
    const std::size_t steps = N > 0 ? static_cast<std::size_t>(N) : 0, mm = static_cast<std::size_t>(m);
    Guard([&] {
        for (std::size_t k = 0; k < steps; ++k)
        {
            const Vector v = Vec(nu + k * mm, mm);
            const Vector s = Kalman::Utils::Solve(Mat(S + k * mm * mm, mm, mm), v);
            double q = 0.;
            for (std::size_t j = 0; j < mm; ++j) q += v(j) * s(j);
            nis[k] = q;
        }
        return 0;
    }, {{nis, steps}});
}

void diag_nees_batch(int N, int n,
                      const double *e, const double *P, double *nees)
{
    const std::size_t steps = N > 0 ? static_cast<std::size_t>(N) : 0, nn = static_cast<std::size_t>(n);
    Guard([&] {
        for (std::size_t k = 0; k < steps; ++k)
            nees[k] = Kalman::Utils::Mahalanobis(Vec(e + k * nn, nn),
                                                 Kalman::Utils::Inverse(Mat(P + k * nn * nn, nn, nn)));
        return 0;
    }, {{nees, steps}});
}

int enkf_analysis(int n, int N, int p,
                  const double *X, const double *Y, const double *R, const double *z, const double *eps,
                  const double *rho_xy, const double *rho_yy, double *X_out)
{
    const std::size_t nn = static_cast<std::size_t>(n), NN = static_cast<std::size_t>(N), pp = static_cast<std::size_t>(p);
    return Guard([&] {
        if (n <= 0 || N < 2 || p <= 0) throw std::invalid_argument("enkf_analysis: need n, p > 0 and at least two members");
        const double norm = 1. / static_cast<double>(NN - 1);

        // Anomalies: X' (n x N) and Y'^T (N x p).
        Matrix Xa(nn, NN), YaT(NN, pp);
        for (std::size_t i = 0; i < nn; ++i)
        {
            double mean = 0.;
            for (std::size_t s = 0; s < NN; ++s) mean += X[i * NN + s];
            mean /= static_cast<double>(NN);
            for (std::size_t s = 0; s < NN; ++s) Xa(i, s) = X[i * NN + s] - mean;
        }
        for (std::size_t j = 0; j < pp; ++j)
        {
            double mean = 0.;
            for (std::size_t s = 0; s < NN; ++s) mean += Y[j * NN + s];
            mean /= static_cast<double>(NN);
            for (std::size_t s = 0; s < NN; ++s) YaT(s, j) = Y[j * NN + s] - mean;
        }

        Matrix PHt = Xa * YaT;                        // (n, p)
        PHt *= norm;
        Matrix S = YaT.Transpose() * YaT;             // (p, p)
        S *= norm;
        if (rho_xy)
            for (std::size_t i = 0; i < nn; ++i)
                for (std::size_t j = 0; j < pp; ++j) PHt(i, j) *= rho_xy[i * pp + j];
        for (std::size_t i = 0; i < pp; ++i)
            for (std::size_t j = 0; j < pp; ++j)
                S(i, j) = S(i, j) * (rho_yy ? rho_yy[i * pp + j] : 1.) + R[i * pp + j];

        const Matrix K = PHt * Kalman::Utils::Inverse(S);      // (n, p)

        // D = z + eps - Y (p x N), then X + K D.
        Matrix D(pp, NN);
        for (std::size_t j = 0; j < pp; ++j)
            for (std::size_t s = 0; s < NN; ++s) D(j, s) = z[j] + eps[j * NN + s] - Y[j * NN + s];
        const Matrix KD = K * D;
        for (std::size_t i = 0; i < nn; ++i)
            for (std::size_t s = 0; s < NN; ++s) X_out[i * NN + s] = X[i * NN + s] + KD(i, s);
        return 0;
    }, {{X_out, nn * NN}});
}

} // extern "C"
