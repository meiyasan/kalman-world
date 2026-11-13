/**
 * *********************************************
 *
 * \file Ensemble.cc
 * \brief Implementation of the Kalman::Engines::Ensemble class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Engines/Ensemble.h"
#include "Kalman/Utils.h"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace Kalman::Engines
{
    namespace
    {
        bool AnyNonZero(const Matrix &M)
        {
            for (std::size_t i = 0; i < M.Rows() * M.Cols(); ++i)
                if (M.Data()[i] != 0.) return true;
            return false;
        }
    }

    // --- the localization function --------------------------------------------

    double GaspariCohn(double distance, double L)
    {
        // The same polynomials as engines/ensemble.py, term for term.
        const double r = distance / L;
        double c = 0.;
        if (r <= 1.)
            c = 1. - 5. / 3. * (r * r) + 5. / 8. * std::pow(r, 3) + 0.5 * std::pow(r, 4) - 0.25 * std::pow(r, 5);
        else if (r <= 2.)
            c = 4. - 5. * r + 5. / 3. * (r * r) + 5. / 8. * std::pow(r, 3) - 0.5 * std::pow(r, 4)
                + 1. / 12. * std::pow(r, 5) - 2. / (3. * r);
        return std::max(c, 0.);
    }

    // --- the hooks a model may offer -------------------------------------------

    void EnsembleModel::Forecast(const Model &model, std::vector<Vector> &members, const Vector &u, double t) const
    {
        for (Vector &x : members) x = model.f(x, u, t);
    }

    void EnsembleModel::Observe(const Model &model, const std::vector<Vector> &members, double t,
                                std::vector<Vector> &out) const
    {
        out.clear();
        out.reserve(members.size());
        for (const Vector &x : members) out.push_back(model.h(x, t));
    }

    bool EnsembleModel::Localization(Matrix &, Matrix &) const { return false; }

    bool EnsembleModel::ProcessNoiseVariances(double, Vector &) const { return false; }

    // --- the engine --------------------------------------------------------------

    Ensemble::Ensemble(std::size_t nMembers, double inflation, std::uint64_t seed)
        : fNMembers(nMembers), fInflation(inflation), fRandom(seed)
    {
        if (nMembers < 2) throw std::invalid_argument("Engines::Ensemble: at least two members.");
        if (!(inflation > 0.)) throw std::invalid_argument("Engines::Ensemble: inflation must be positive.");
    }

    void Ensemble::SetLocalization(const Matrix &rhoXY, const Matrix &rhoYY)
    {
        fRhoXY = rhoXY;
        fRhoYY = rhoYY;
    }

    void Ensemble::SetMembers(std::vector<Vector> members)
    {
        if (!members.empty())
        {
            if (members.size() != fNMembers)
            {
                std::ostringstream os;
                os << "Engines::Ensemble: " << members.size() << " members handed in, the engine has " << fNMembers << '.';
                throw std::invalid_argument(os.str());
            }
            for (const Vector &x : members)
                if (x.Size() != members.front().Size())
                    throw std::invalid_argument("Engines::Ensemble: the members handed in differ in size.");
        }
        fStart = std::move(members);
    }

    void Ensemble::SetNormals(std::vector<double> normals)
    {
        fNormals = std::move(normals);
        fNormalsUsed = 0;
        fHasNormals = true;
    }

    std::vector<Vector> Ensemble::Normals(std::size_t rows) const
    {
        const std::size_t N = fNMembers;
        std::vector<Vector> w(N, Vector(rows));
        if (!fHasNormals)
        {
            // The engine's own stream, one member after the other.
            std::normal_distribution<double> gauss(0., 1.);
            for (Vector &v : w)
                for (std::size_t i = 0; i < rows; ++i) v(i) = gauss(fRandom);
            return w;
        }
        if (fNormals.size() - fNormalsUsed < rows * N)
        {
            std::ostringstream os;
            os << "Engines::Ensemble: the draws handed in are used up (" << rows * N << " more needed, "
               << fNormals.size() - fNormalsUsed << " left).";
            throw std::runtime_error(os.str());
        }
        // numpy's (rows, N) block, row-major: member s reads column s.
        const double *block = fNormals.data() + fNormalsUsed;
        for (std::size_t i = 0; i < rows; ++i)
            for (std::size_t s = 0; s < N; ++s) w[s](i) = block[i * N + s];
        fNormalsUsed += rows * N;
        return w;
    }

    Vector Ensemble::Draw(const Matrix &factor) const
    {
        std::normal_distribution<double> gauss(0., 1.);
        Vector w(factor.Cols());
        for (std::size_t i = 0; i < w.Size(); ++i) w(i) = gauss(fRandom);
        return factor * w;
    }

    void Ensemble::Reset(const Model &model)
    {
        fModelRhoXY = Matrix();
        fModelRhoYY = Matrix();
        if (const auto *hooks = dynamic_cast<const EnsembleModel *>(&model))
            if (!hooks->Localization(fModelRhoXY, fModelRhoYY))
                fModelRhoXY = fModelRhoYY = Matrix();

        if (!fStart.empty())
        {
            if (fStart.front().Size() != model.NState())
                throw std::runtime_error("Engines::Ensemble: the members handed in do not have the model's n.");
            fMembers = fStart;
            return;
        }

        const Matrix factor = Utils::CholeskyFactor(model.P0(), true);
        if (factor.Empty())
            throw std::runtime_error("Engines::Ensemble: P0 has no Cholesky factor; cannot seed the members.");

        fMembers.assign(fNMembers, model.X0());
        for (Vector &x : fMembers) x += Draw(factor);
    }

    Vector Ensemble::Mean() const
    {
        if (fMembers.empty()) throw std::runtime_error("Engines::Ensemble: no members; call Reset() first.");
        Vector mean(fMembers.front().Size(), 0.);
        for (const Vector &x : fMembers) mean += x;
        mean *= 1. / static_cast<double>(fMembers.size());
        return mean;
    }

    void Ensemble::Estimate(Vector &mean, Matrix &cov) const
    {
        mean = Mean();
        const std::size_t n = mean.Size();
        const std::size_t N = fMembers.size();
        const double norm = 1. / static_cast<double>(std::max<std::size_t>(N - 1, 1));

        // (A A^T) / (N - 1) with A the anomalies, as one product (BLAS when it pays).
        Matrix A(n, N), AT(N, n);
        for (std::size_t s = 0; s < N; ++s)
            for (std::size_t i = 0; i < n; ++i)
            {
                const double a = fMembers[s](i) - mean(i);
                A(i, s) = a;
                AT(s, i) = a;
            }
        cov = A * AT;
        cov *= norm;
        Utils::Symmetrize(cov);
        for (std::size_t i = 0; i < n; ++i) cov(i, i) += 1e-12;
    }

    double Ensemble::Spread() const
    {
        const Vector mean = Mean();
        const std::size_t n = mean.Size();
        double total = 0.;
        for (std::size_t i = 0; i < n; ++i)
        {
            double v = 0.;
            for (const Vector &x : fMembers) v += (x(i) - mean(i)) * (x(i) - mean(i));
            total += std::sqrt(v / static_cast<double>(fMembers.size()));
        }
        return total / static_cast<double>(n);
    }

    Vector Ensemble::Forecast(const Model &model, const Vector *u, double t) const
    {
        if (fMembers.empty()) const_cast<Ensemble *>(this)->Reset(model);
        const std::size_t n = model.NState();

        Vector uu(model.NControl(), 0.);
        if (u && model.NControl() > 0 && u->Size() == model.NControl()) uu = *u;

        const auto *hooks = dynamic_cast<const EnsembleModel *>(&model);
        if (hooks) hooks->Forecast(model, fMembers, uu, t);
        else
            for (Vector &x : fMembers) x = model.f(x, uu, t);

        // The process noise, after every member has moved (the order of the draws
        // is the Python engine's: one (n, N) block).
        Vector variances;
        if (hooks && hooks->ProcessNoiseVariances(t, variances) && model.L(t).Empty())
        {
            if (variances.Size() != n) throw std::runtime_error("Engines::Ensemble: the process-noise variances are not n long.");
            bool any = false;
            for (std::size_t i = 0; i < n; ++i) any = any || variances(i) != 0.;
            if (any)
            {
                Vector sigma(n);
                for (std::size_t i = 0; i < n; ++i) sigma(i) = std::sqrt(variances(i));
                const std::vector<Vector> w = Normals(n);
                for (std::size_t s = 0; s < fMembers.size(); ++s)
                    for (std::size_t i = 0; i < n; ++i) fMembers[s](i) += sigma(i) * w[s](i);
            }
            return Mean();
        }

        Matrix Q = model.Q(t);
        const Matrix L = model.L(t);
        if (!L.Empty()) Q = Utils::Similarity(L, Q);
        if (AnyNonZero(Q))
        {
            const Matrix factor = Utils::CholeskyFactor(Q, true);
            const std::vector<Vector> w = Normals(factor.Cols());
            for (std::size_t s = 0; s < fMembers.size(); ++s) fMembers[s] += factor * w[s];
        }
        return Mean();
    }

    void Ensemble::Observations(const Model &model, double t, std::vector<Vector> &Y) const
    {
        if (const auto *hooks = dynamic_cast<const EnsembleModel *>(&model))
            hooks->Observe(model, fMembers, t, Y);
        else
        {
            Y.clear();
            Y.reserve(fMembers.size());
            for (const Vector &x : fMembers) Y.push_back(model.h(x, t));
        }
        if (Y.size() != fMembers.size()) throw std::runtime_error("Engines::Ensemble: Observe() must give one measurement per member.");
    }

    Matrix Ensemble::Analysis(const Matrix &X, const Matrix &Y, const Matrix &R, const Vector &z,
                              const Matrix &eps, const Matrix *rhoXY, const Matrix *rhoYY)
    {
        const std::size_t n = X.Rows(), N = X.Cols(), p = Y.Rows();
        if (N < 2 || Y.Cols() != N || eps.Rows() != p || eps.Cols() != N || R.Rows() != p || R.Cols() != p || z.Size() != p)
            throw std::invalid_argument("Engines::Ensemble::Analysis: inconsistent shapes.");
        const double norm = 1. / static_cast<double>(N - 1);

        // Anomalies: X' (n x N) and Y'^T (N x p).
        Matrix Xa(n, N), YaT(N, p);
        for (std::size_t i = 0; i < n; ++i)
        {
            double mean = 0.;
            for (std::size_t s = 0; s < N; ++s) mean += X(i, s);
            mean /= static_cast<double>(N);
            for (std::size_t s = 0; s < N; ++s) Xa(i, s) = X(i, s) - mean;
        }
        for (std::size_t j = 0; j < p; ++j)
        {
            double mean = 0.;
            for (std::size_t s = 0; s < N; ++s) mean += Y(j, s);
            mean /= static_cast<double>(N);
            for (std::size_t s = 0; s < N; ++s) YaT(s, j) = Y(j, s) - mean;
        }

        Matrix PHt = Xa * YaT;                        // (n, p)
        PHt *= norm;
        Matrix S = YaT.Transpose() * YaT;             // (p, p)
        S *= norm;
        if (rhoXY)
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < p; ++j) PHt(i, j) *= (*rhoXY)(i, j);
        for (std::size_t i = 0; i < p; ++i)
            for (std::size_t j = 0; j < p; ++j)
                S(i, j) = S(i, j) * (rhoYY ? (*rhoYY)(i, j) : 1.) + R(i, j);

        const Matrix K = PHt * Utils::Inverse(S);    // (n, p)

        // D = z + eps - Y (p x N), then X + K D.
        Matrix D(p, N);
        for (std::size_t j = 0; j < p; ++j)
            for (std::size_t s = 0; s < N; ++s) D(j, s) = z(j) + eps(j, s) - Y(j, s);
        Matrix out = K * D;
        out += X;
        return out;
    }

    Vector Ensemble::Analyze(const Model &model, const Vector &z, double t) const
    {
        if (fMembers.empty()) const_cast<Ensemble *>(this)->Reset(model);

        const std::vector<std::size_t> keep = Utils::ValidChannels(z);
        if (keep.empty()) return Mean();

        const std::size_t n = model.NState();
        const std::size_t m = model.NMeasure();
        const std::size_t p = keep.size();
        const std::size_t N = fMembers.size();

        // Inflate the anomalies before the analysis.
        const Vector mean = Mean();
        if (fInflation != 1.)
            for (Vector &x : fMembers)
                for (std::size_t i = 0; i < n; ++i) x(i) = mean(i) + fInflation * (x(i) - mean(i));

        // Observation-space ensemble, restricted to the channels that arrived.
        std::vector<Vector> Yfull;
        Observations(model, t, Yfull);

        Matrix X(n, N), Y(p, N);
        for (std::size_t s = 0; s < N; ++s)
        {
            for (std::size_t i = 0; i < n; ++i) X(i, s) = fMembers[s](i);
            if (Yfull[s].Size() != m) throw std::runtime_error("Engines::Ensemble: a member's measurement is not m long.");
            for (std::size_t j = 0; j < p; ++j) Y(j, s) = Yfull[s](keep[j]);
        }

        // The tapers: those set on the engine, else the model's.
        const Matrix &rhoXY = fRhoXY.Empty() && fRhoYY.Empty() ? fModelRhoXY : fRhoXY;
        const Matrix &rhoYY = fRhoXY.Empty() && fRhoYY.Empty() ? fModelRhoYY : fRhoYY;
        Matrix rxy, ryy;
        if (!rhoXY.Empty())
        {
            if (rhoXY.Rows() != n || rhoXY.Cols() != m)
                throw std::runtime_error("Engines::Ensemble: rho_xy must be (n, m).");
            rxy = Matrix(n, p);
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < p; ++j) rxy(i, j) = rhoXY(i, keep[j]);
        }
        if (!rhoYY.Empty())
        {
            if (rhoYY.Rows() != m || rhoYY.Cols() != m)
                throw std::runtime_error("Engines::Ensemble: rho_yy must be (m, m).");
            ryy = Matrix(p, p);
            for (std::size_t i = 0; i < p; ++i)
                for (std::size_t j = 0; j < p; ++j) ryy(i, j) = rhoYY(keep[i], keep[j]);
        }

        // The perturbations: one (p, N) block of draws through the factor of R.
        const Matrix R = Utils::CompressBoth(model.R(t), keep);
        const Matrix Rfac = Utils::CholeskyFactor(R, true);
        Matrix eps(p, N, 0.);
        if (!Rfac.Empty())
        {
            const std::vector<Vector> w = Normals(Rfac.Cols());
            for (std::size_t s = 0; s < N; ++s)
            {
                const Vector e = Rfac * w[s];
                for (std::size_t j = 0; j < p; ++j) eps(j, s) = e(j);
            }
        }

        const Matrix Xa = Analysis(X, Y, R, Utils::Compress(z, keep), eps,
                                   rxy.Empty() ? nullptr : &rxy, ryy.Empty() ? nullptr : &ryy);
        for (std::size_t s = 0; s < N; ++s)
            for (std::size_t i = 0; i < n; ++i) fMembers[s](i) = Xa(i, s);
        return Mean();
    }

    void Ensemble::Propagate(const Model &model, const Vector &x, const Vector *u,
                             double t, Vector &xk, Matrix &F) const
    {
        Vector uu(model.NControl(), 0.);
        if (u && model.NControl() > 0 && u->Size() == model.NControl()) uu = *u;
        xk = model.f(x, uu, t);
        F = Matrix::Identity(model.NState());
    }

    void Ensemble::Measure(const Model &model, const Vector &x, double t,
                           Vector &zpred, Matrix &H) const
    {
        zpred = model.h(x, t);
        H = model.JacobianH(x, t);
    }

    void Ensemble::Predict(const Model &model, const Vector &, const Matrix &,
                           const Vector *u, double t,
                           Vector &xk, Matrix &Pk, Matrix &F) const
    {
        Forecast(model, u, t);
        Estimate(xk, Pk);
        F = Matrix::Identity(model.NState());
    }

    std::size_t Ensemble::Update(const Model &model, const Vector &, const Matrix &,
                                 const Vector &z, double t,
                                 Vector &xOut, Matrix &POut) const
    {
        Analyze(model, z, t);
        Estimate(xOut, POut);
        return Utils::ValidChannels(z).size();
    }

    void Ensemble::Smooth(const Vector &, const Matrix &, const Vector &, const Matrix &,
                          const Vector &, const Matrix &, const Matrix &,
                          Vector &, Matrix &) const
    {
        throw std::runtime_error("Engines::Ensemble: an ensemble has no RTS smoother; use Forward() only.");
    }
}
