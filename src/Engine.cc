/**
 * *********************************************
 *
 * \file Engine.cc
 * \brief Implementation of the shared engine machinery
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engine.h"
#include "Kalman/Utils.h"

namespace Kalman
{
    void Engine::Predict(const Model &model, const Vector &x, const Matrix &P,
                         const Vector *u, double t,
                         Vector &xk, Matrix &Pk, Matrix &F) const
    {
        Propagate(model, x, u, t, xk, F);

        Matrix Q = model.Q(t);
        const Matrix L = model.L(t);
        if (!L.Empty()) Q = Utils::Similarity(L, Q);      // L Q L^T

        Pk = Utils::Similarity(F, P);                     // F P F^T
        Pk += Q;
        Utils::Symmetrize(Pk);
    }

    std::size_t Engine::Update(const Model &model, const Vector &x, const Matrix &P,
                               const Vector &z, double t,
                               Vector &xOut, Matrix &POut) const
    {
        xOut = x;
        POut = P;

        const std::vector<std::size_t> keep = Utils::ValidChannels(z);
        if (keep.empty()) return 0;                       // predict-only step

        Vector zpred;
        Matrix H;
        Measure(model, x, t, zpred, H);

        // Residual against the true h(x) — not H x — so the same code is correct
        // for a nonlinear observation map.
        Vector y(z);
        y -= zpred;

        Utils::Correct(Utils::CompressRows(H, keep), Utils::CompressBoth(model.R(t), keep),
                       Utils::Compress(y, keep), x, P, xOut, POut);
        return keep.size();
    }

    void Engine::Smooth(const Vector &xPost, const Matrix &PPost,
                        const Vector &xPriorNext, const Matrix &PPriorNext,
                        const Vector &xSmoothNext, const Matrix &PSmoothNext,
                        const Matrix &F,
                        Vector &xSmooth, Matrix &PSmooth) const
    {
        const Matrix C = PPost * F.Transpose() * Utils::Inverse(PPriorNext);

        Vector dx(xSmoothNext);
        dx -= xPriorNext;

        xSmooth = xPost;
        xSmooth += C * dx;

        Matrix dP(PSmoothNext);
        dP -= PPriorNext;

        PSmooth = PPost;
        PSmooth += Utils::Similarity(C, dP);
        Utils::Symmetrize(PSmooth);
    }

    void Engine::Innovation(const Model &model, const Vector &x, const Matrix &P,
                            const Vector &z, double t, Vector &nu, Matrix &S) const
    {
        Vector zpred;
        Matrix H;
        Measure(model, x, t, zpred, H);

        nu = z;
        nu -= zpred;                                      // NaN survives where z was missing

        S = Utils::Similarity(H, P);
        S += model.R(t);
    }
}
