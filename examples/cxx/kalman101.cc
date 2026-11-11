/**
 * \file kalman101.cc
 * \brief The whole library in one program: simulate, filter, smooth, check.
 *
 * Generates a noisy constant-velocity track, filters it, smooths it, and reports
 * the consistency diagnostics. Nothing outside libKalman is used, so this also
 * demonstrates that the library stands alone: no ROOT, no Eigen, no Torch.
 *
 * The model is defined *here*, in the application, not in the library. libKalman
 * ships no physics — a constant-velocity tracker is a domain choice, and it lives
 * with the domain.
 *
 *     make build && make examples
 *     ./var/build/bin/kalman101 > track.tsv
 */
#include <Kalman/Kalman.h>

#include <cmath>
#include <cstdio>
#include <vector>

/// Constant velocity in 1-D with white-noise acceleration: state (position, velocity).
class ConstantVelocity : public Kalman::Models::Linear
{
    public:
        ConstantVelocity(double fs, double sigmaAcc, double sigmaMeas)
            : Kalman::Models::Linear("constant-velocity", 2, 1)
        {
            SetSampleRate(fs);

            const double dt = DeltaT();
            const double q = sigmaAcc * sigmaAcc;

            Kalman::Matrix F = Kalman::Matrix::Identity(2);
            F(0, 1) = dt;
            SetF(F);

            Kalman::Matrix H(1, 2, 0.);
            H(0, 0) = 1.;                       // only the position is measured
            SetH(H);

            // Discrete white-noise acceleration: an unmodelled acceleration of standard
            // deviation sigma_a [m/s^2], held constant over each step of length dt.
            // (The continuous-time version has dt^3/3, dt^2/2, dt and a spectral
            // density instead.) The off-diagonal terms matter: one acceleration moves
            // both the position and the velocity, so their errors are correlated, and
            // dropping these terms makes the filter inconsistent.
            Kalman::Matrix Q(2, 2, 0.);
            Q(0, 0) = q * dt * dt * dt * dt / 4.;
            Q(0, 1) = q * dt * dt * dt / 2.;
            Q(1, 0) = Q(0, 1);
            Q(1, 1) = q * dt * dt;
            SetQ(Q);

            Kalman::Matrix R(1, 1, 0.);
            R(0, 0) = sigmaMeas * sigmaMeas;
            SetR(R);
        }
};

int main()
{
    // --- the system ------------------------------------------------------
    ConstantVelocity model(/* fs */ 100., /* sigma_a */ 0.5, /* sigma_z */ 0.2);

    Kalman::Vector x0(2);
    x0(0) = 0.;                                 // start at the origin
    x0(1) = 1.;                                 // moving at 1 m/s
    model.SetInitialState(x0);

    std::fprintf(stderr, "%s\n", model.Describe().c_str());

    // --- the data --------------------------------------------------------
    Kalman::Sampler sampler(&model, 20260804);
    sampler.Generate(600, x0);

    // --- the estimate ----------------------------------------------------
    Kalman::Filters::KF filter(&model);
    const Kalman::Record forward = filter.Forward(sampler.Measurements());
    const Kalman::Record smoothed = filter.Backward();

    std::fprintf(stderr, "mean NIS  = %.3f   (consistent when close to m = %zu)\n",
                 filter.MeanNIS(), model.NMeasure());
    std::fprintf(stderr, "mean NEES = %.3f   (consistent when close to n = %zu)\n",
                 filter.MeanNEES(sampler.Truth()), model.NState());
    std::fprintf(stderr, "RMSE  raw = %.5f   filtered = %.5f   smoothed = %.5f\n",
                 [&] {
                     double e = 0.;
                     for (std::size_t k = 0; k < sampler.NSteps(); ++k)
                         e += std::pow(sampler.Measurements()[k](0) - sampler.Truth()[k](0), 2);
                     return std::sqrt(e / sampler.NSteps());
                 }(),
                 forward.RMSE(0, sampler.Truth()),
                 smoothed.RMSE(0, sampler.Truth()));

    // --- the data anyone can plot ----------------------------------------
    // Deliberately a plain table on stdout: the record holds the +/- sigma band,
    // and whether that becomes a TGraph, a matplotlib axis or a gnuplot file is
    // the caller's business, not the library's.
    const std::vector<double> sigma = forward.Sigma(0);
    std::printf("# t\ttruth\tmeasured\tfiltered\tsigma\tsmoothed\n");
    for (std::size_t k = 0; k < forward.NSteps(); ++k)
        std::printf("%.4f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                    forward.Time(k), sampler.Truth()[k](0), forward.Measurement(k)(0),
                    forward.State(k)(0), sigma[k], smoothed.State(k)(0));

    return 0;
}
