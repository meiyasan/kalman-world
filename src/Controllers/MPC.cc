/**
 * *********************************************
 *
 * \file MPC.cc
 * \brief Source of the Kalman::Controllers::MPC class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Controllers/MPC.h"
#include "Kalman/Algebra/Cholesky.h"
#include "Kalman/Algebra/Riccati.h"
#include "Kalman/Algebra/Symmetric.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace Kalman::Controllers
{
    namespace
    {
        const double kInf = std::numeric_limits<double>::infinity();

        Vector Limits(const Vector &given, std::size_t l, double none, const char *what)
        {
            if (given.Empty()) return Vector(l, none);
            if (given.Size() != l)
                throw std::invalid_argument(std::string("Controllers::MPC: ") + what + " has " + std::to_string(given.Size())
                                            + " entries, the command has " + std::to_string(l) + ".");
            return given;
        }
    }

    MPC::MPC(Model *model, std::size_t horizon, const Vector &uMin, const Vector &uMax, const Vector &duMax,
             std::size_t nIterations, const Vector &xRef, const Matrix &Qcost, const Matrix &Rcost)
        : Controller(model, xRef, Qcost, Rcost), fHorizon(horizon), fNIterations(std::max<std::size_t>(nIterations, 1))
    {
        if (fModel->NControl() == 0)
            throw std::invalid_argument("Controllers::MPC: the model has no control input (l = 0).");
        if (horizon == 0) throw std::invalid_argument("Controllers::MPC: the horizon must be positive.");
        const std::size_t l = NControl();
        fUMin = Limits(uMin, l, -kInf, "uMin");
        fUMax = Limits(uMax, l, kInf, "uMax");
        fDUMax = Limits(duMax, l, kInf, "duMax");
        for (std::size_t j = 0; j < l; ++j)
        {
            if (!(fUMin(j) <= fUMax(j))) throw std::invalid_argument("Controllers::MPC: uMin must not exceed uMax.");
            if (!(fDUMax(j) > 0.)) throw std::invalid_argument("Controllers::MPC: duMax must be positive.");
        }
        fPlan.assign(fHorizon, Vector(l, 0.));
    }

    void MPC::SetTerminalCost(const Matrix &Pf)
    {
        const std::size_t n = fModel->NState();
        if (Pf.Rows() != n || Pf.Cols() != n) throw std::invalid_argument("Controllers::MPC: the terminal cost must be (n, n).");
        fTerminal = Pf;
        fTerminalGiven = true;
    }

    void MPC::SetSolver(std::size_t maxSweeps, double tolerance)
    {
        if (maxSweeps == 0 || !(tolerance > 0.)) throw std::invalid_argument("Controllers::MPC: sweeps and tolerance must be positive.");
        fMaxSweeps = maxSweeps;
        fTolerance = tolerance;
    }

    void MPC::Reset()
    {
        Controller::Reset();
        fPlan.assign(fHorizon, Vector(NControl(), 0.));
        fPredicted.clear();
        fLast = Vector();
        fSweeps = fActive = 0;
    }

    // `x_next ~ f(x, u) + A (dx) + B (du)` at (x, u, t).
    void MPC::Linearise(const Vector &x, const Vector &u, double t, Matrix &A, Matrix &B) const
    {
        const std::size_t n = fModel->NState(), l = NControl();
        if (fModel->HasAnalyticJacobian())
        {
            // A model of matrices: its own F and B, exactly.
            A = fModel->F(t);
            B = fModel->B(t);
            if (B.Empty()) B = Matrix::Identity(n);         // the command enters the state directly
            return;
        }
        A = fModel->Jacobian(x, u, t);

        // df/du, differenced towards the inside of the limits: an f that clips its
        // command is flat outside them, and a plan resting on a limit would
        // otherwise see a command with no effect.
        B = Matrix(n, l, 0.);
        const Vector base = fModel->f(x, u, t);
        for (std::size_t j = 0; j < l; ++j)
        {
            double step = 1e-6 * std::max(1., std::fabs(u(j)));
            if (u(j) + step > fUMax(j)) step = -step;
            Vector moved(u);
            moved(j) += step;
            const Vector shifted = fModel->f(x, moved, t);
            for (std::size_t i = 0; i < n; ++i) B(i, j) = (shifted(i) - base(i)) / step;
        }
    }

    const Matrix &MPC::Terminal(double t)
    {
        if (!fTerminal.Empty()) return fTerminal;
        // The cost of running LQR for ever from the end of the horizon: with it, and
        // no limit in the way, the first command of the plan is the LQR command.
        try
        {
            Matrix A, B;
            Linearise(fXRef, Vector(NControl(), 0.), t, A, B);
            fTerminal = Algebra::SolveDARE(A, B, fQCost, fRCost);
        }
        catch (const std::exception &)
        {
            fTerminal = fQCost;                             // no stabilising solution: weigh the end as any step
        }
        return fTerminal;
    }

    void MPC::Solve(const Vector &x0, double t)
    {
        const std::size_t n = fModel->NState(), l = NControl(), N = fHorizon;
        const double dt = fModel->DeltaT();
        const Matrix &Pf = Terminal(t);

        // The path the current plan leads through, and the model linearised along it.
        std::vector<Vector> xbar(N + 1);
        std::vector<Matrix> A(N), B(N);
        xbar[0] = x0;
        for (std::size_t k = 0; k < N; ++k)
        {
            const double tk = t + static_cast<double>(k) * dt;
            Linearise(xbar[k], fPlan[k], tk, A[k], B[k]);
            xbar[k + 1] = fModel->f(xbar[k], fPlan[k], tk);
        }

        // S: how each command moves each later state. Block (k, j), for the state
        // after k + 1 steps and command j <= k, is A_k ... A_{j+1} B_j.
        Matrix S(n * N, l * N, 0.);
        for (std::size_t j = 0; j < N; ++j)
        {
            Matrix M = B[j];
            for (std::size_t k = j; k < N; ++k)
            {
                if (k > j) M = A[k] * M;
                for (std::size_t a = 0; a < n; ++a)
                    for (std::size_t b = 0; b < l; ++b) S(k * n + a, j * l + b) = M(a, b);
            }
        }

        // The states as a function of the commands U: X = xbar + S (U - Ubar), so the
        // error is E0 + S U with E0 = xbar - x_ref - S Ubar.
        Vector Ubar(l * N);
        for (std::size_t k = 0; k < N; ++k)
            for (std::size_t b = 0; b < l; ++b) Ubar(k * l + b) = fPlan[k](b);
        const Vector SU = S * Ubar;
        Vector E0(n * N);
        for (std::size_t k = 0; k < N; ++k)
            for (std::size_t a = 0; a < n; ++a) E0(k * n + a) = xbar[k + 1](a) - fXRef(a) - SU(k * n + a);

        // W S and W E0, with W = diag(Qcost, ..., Qcost, Pf).
        Matrix WS(n * N, l * N, 0.);
        Vector WE(n * N, 0.);
        for (std::size_t k = 0; k < N; ++k)
        {
            const Matrix &W = (k + 1 == N) ? Pf : fQCost;
            for (std::size_t a = 0; a < n; ++a)
            {
                double e = 0.;
                for (std::size_t c = 0; c < n; ++c) e += W(a, c) * E0(k * n + c);
                WE(k * n + a) = e;
                for (std::size_t col = 0; col < l * N; ++col)
                {
                    double v = 0.;
                    for (std::size_t c = 0; c < n; ++c) v += W(a, c) * S(k * n + c, col);
                    WS(k * n + a, col) = v;
                }
            }
        }

        // minimise  U^T H U / 2 + g^T U,   H = S^T W S + Rbar,   g = S^T W E0.
        const Matrix St = S.Transpose();
        Matrix H = St * WS;
        for (std::size_t k = 0; k < N; ++k)
            for (std::size_t a = 0; a < l; ++a)
                for (std::size_t b = 0; b < l; ++b) H(k * l + a, k * l + b) += fRCost(a, b);
        Algebra::Symmetrize(H);
        const Vector g = St * WE;
        const Matrix Hinv = Algebra::Inverse(H);

        // The limits, as rows  C U <= c.
        struct Row { std::size_t at; double sign; std::size_t before; bool chained; double bound; };
        std::vector<Row> rows;
        for (std::size_t k = 0; k < N; ++k)
            for (std::size_t b = 0; b < l; ++b)
            {
                const std::size_t i = k * l + b;
                if (std::isfinite(fUMax(b))) rows.push_back({i, 1., 0, false, fUMax(b)});
                if (std::isfinite(fUMin(b))) rows.push_back({i, -1., 0, false, -fUMin(b)});
                if (!std::isfinite(fDUMax(b))) continue;
                if (k > 0)
                {
                    rows.push_back({i, 1., i - l, true, fDUMax(b)});
                    rows.push_back({i, -1., i - l, true, fDUMax(b)});
                }
                else if (!fLast.Empty())
                {
                    rows.push_back({i, 1., 0, false, fDUMax(b) + fLast(b)});
                    rows.push_back({i, -1., 0, false, fDUMax(b) - fLast(b)});
                }
            }

        Vector U = Hinv * g;
        U *= -1.;                                           // the plan with no limit
        fSweeps = fActive = 0;

        const std::size_t nc = rows.size();
        if (nc > 0)
        {
            // Does the free plan already respect every limit?
            const auto lhs = [&rows](const Vector &V, std::size_t r) {
                const Row &row = rows[r];
                return row.sign * (V(row.at) - (row.chained ? V(row.before) : 0.));
            };
            bool feasible = true;
            for (std::size_t r = 0; r < nc && feasible; ++r) feasible = lhs(U, r) <= rows[r].bound + 1e-12;

            if (!feasible)
            {
                // C and C H^-1, row by row (each row of C has one or two entries).
                Matrix CHinv(nc, l * N, 0.);
                for (std::size_t r = 0; r < nc; ++r)
                    for (std::size_t col = 0; col < l * N; ++col)
                        CHinv(r, col) = rows[r].sign * (Hinv(rows[r].at, col) - (rows[r].chained ? Hinv(rows[r].before, col) : 0.));
                // M = C H^-1 C^T,   d = c + C H^-1 g.
                Matrix M(nc, nc, 0.);
                Vector d(nc);
                for (std::size_t r = 0; r < nc; ++r)
                {
                    for (std::size_t q = 0; q < nc; ++q)
                        M(r, q) = rows[q].sign * (CHinv(r, rows[q].at) - (rows[q].chained ? CHinv(r, rows[q].before) : 0.));
                    double cg = 0.;
                    for (std::size_t col = 0; col < l * N; ++col) cg += CHinv(r, col) * g(col);
                    d(r) = rows[r].bound + cg;
                }

                // Hildreth: one multiplier at a time, each set to its best value given the others.
                Vector lambda(nc, 0.);
                for (fSweeps = 1; fSweeps <= fMaxSweeps; ++fSweeps)
                {
                    double change = 0., size = 0.;
                    for (std::size_t r = 0; r < nc; ++r)
                    {
                        double w = d(r);
                        for (std::size_t q = 0; q < nc; ++q)
                            if (q != r) w += M(r, q) * lambda(q);
                        const double next = M(r, r) > 0. ? std::max(0., -w / M(r, r)) : 0.;
                        change = std::max(change, std::fabs(next - lambda(r)));
                        size = std::max(size, std::fabs(next));
                        lambda(r) = next;
                    }
                    if (change <= fTolerance * std::max(1., size)) break;
                }
                fSweeps = std::min(fSweeps, fMaxSweeps);

                // U = -H^-1 (g + C^T lambda).
                Vector rhs(g);
                for (std::size_t r = 0; r < nc; ++r)
                {
                    if (lambda(r) == 0.) continue;
                    ++fActive;
                    rhs(rows[r].at) += rows[r].sign * lambda(r);
                    if (rows[r].chained) rhs(rows[r].before) -= rows[r].sign * lambda(r);
                }
                U = Hinv * rhs;
                U *= -1.;
            }
        }

        // The dual solve stops within a tolerance: the limits are then met exactly,
        // the rate limit from the command applied last onwards.
        for (std::size_t k = 0; k < N; ++k)
            for (std::size_t b = 0; b < l; ++b)
            {
                double lo = fUMin(b), hi = fUMax(b);
                if (std::isfinite(fDUMax(b)) && (k > 0 || !fLast.Empty()))
                {
                    const double before = k > 0 ? fPlan[k - 1](b) : fLast(b);
                    lo = std::max(lo, before - fDUMax(b));
                    hi = std::min(hi, before + fDUMax(b));
                }
                fPlan[k](b) = std::min(hi, std::max(lo, U(k * l + b)));
            }

        // The states this plan leads through, by the model itself.
        fPredicted.assign(N + 1, x0);
        for (std::size_t k = 0; k < N; ++k)
            fPredicted[k + 1] = fModel->f(fPredicted[k], fPlan[k], t + static_cast<double>(k) * dt);
    }

    Vector MPC::Policy(const Vector &xHat, const Matrix &, double t, const Vector *)
    {
        if (xHat.Size() != fModel->NState())
            throw std::invalid_argument("Controllers::MPC: the estimate has the wrong size.");

        // Start from the plan of the previous step, moved one step on.
        if (!fLast.Empty())
        {
            for (std::size_t k = 0; k + 1 < fHorizon; ++k) fPlan[k] = fPlan[k + 1];
        }
        for (std::size_t pass = 0; pass < fNIterations; ++pass) Solve(xHat, t);

        fLast = fPlan.front();
        return fLast;
    }
}
