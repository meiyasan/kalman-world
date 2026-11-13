/**
 * \file 20_gravitational_apps.cc
 * \brief Kalman::Apps::Gravitational::Polarization: the Jones measurement, its
 * analytic Jacobian, and a filter that tracks a polarized carrier.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Apps/Apps.h"
#include "Check.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

using namespace Kalman;

int main()
{
    std::printf("20_gravitational_apps\n");
    auto &factory = Factory::Instance();
    CHECK(factory.Has("polarization"), "polarization: registered");
    std::unique_ptr<Model> model = factory.Create("polarization");
    auto *pol = dynamic_cast<Apps::Gravitational::Polarization *>(model.get());
    CHECK(pol != nullptr, "polarization: the factory gives a Polarization");
    if (!pol) return Check::Summary("20_gravitational_apps");

    CHECK(model->NState() == 8 && model->NMeasure() == 4 && model->NControl() == 0, "polarization: 8 states, 4 measurements");
    CHECK(model->StateComponents()[3].name == "phase_vel" && model->StateComponents()[3].unit == "rad/s"
              && model->MeasurementComponents()[2].name == "re2",
          "polarization: components are named, with units");
    CHECK(model->Structure() == StructureKind::Maps, "polarization: h is code");

    // The measurement is the Jones vector: |z|^2 = a^2, and z2/z1 = tan(theta/2) exp(i phi).
    Vector x(8, 0.);
    x(0) = std::log(2.);
    x(2) = 0.7;
    x(4) = 1.1;
    x(6) = -0.4;
    const Vector z = model->h(x, 0.);
    const double power = z(0) * z(0) + z(1) * z(1) + z(2) * z(2) + z(3) * z(3);
    CHECK_CLOSE(power, 4., 1e-12, "polarization: |z|^2 is the squared amplitude");
    const double ratio = std::sqrt((z(2) * z(2) + z(3) * z(3)) / (z(0) * z(0) + z(1) * z(1)));
    CHECK_CLOSE(ratio, std::tan(0.55), 1e-12, "polarization: |z2| / |z1| is tan(theta/2)");

    // The analytic H against central differences of h.
    const Matrix H = model->JacobianH(x, 0.);
    const Matrix numeric = Differentiators::Jacobian(
        DifferentiatorKind::Central, [&model](const Vector &p) { return model->h(p, 0.); }, x, 1e-6);
    double worst = 0.;
    for (std::size_t i = 0; i < H.Rows(); ++i)
        for (std::size_t j = 0; j < H.Cols(); ++j) worst = std::max(worst, std::fabs(H(i, j) - numeric(i, j)));
    CHECK_CLOSE(worst, 0., 1e-8, "polarization: analytic H agrees with differences of h");

    // The transition follows the sample rate: dt = 1/fs on each pair.
    model->SetSampleRate(50.);
    CHECK_CLOSE(model->F(0.)(2, 3), 0.02, 0., "polarization: dt = 1/fs");
    const Vector stepped = model->f(x, Vector(), 0.);
    CHECK_CLOSE(stepped(2), 0.7, 0., "polarization: a phase at rest stays");
    model->SetSampleRate(100.);

    // A carrier at 1 Hz with a fixed polarization, sigma_n = 0.3: the filter
    // locks onto the frequency and the amplitude, and its innovations are white
    // with the size the model says (NIS ~ m).
    model->SetParameter("sigma_n", 0.3);
    model->SetParameter("sigma_psi", 0.6);
    model->SetParameter("sigma_a", 0.05);
    model->SetParameter("sigma_theta", 0.05);
    model->SetParameter("sigma_phi", 0.05);
    const double fs = 100., f0 = 1.;
    const std::size_t N = 3000;
    std::mt19937 rng(11);
    std::normal_distribution<double> noise(0., 0.3);
    std::vector<Vector> zs;
    Vector truth(8, 0.);
    truth(0) = 0.;
    truth(4) = 1.2;
    truth(6) = 2.0;
    for (std::size_t k = 0; k < N; ++k)
    {
        truth(2) = 2. * M_PI * f0 * (k + 1) / fs;
        Vector zk = model->h(truth, 0.);
        for (std::size_t i = 0; i < 4; ++i) zk(i) += noise(rng);
        zs.push_back(zk);
    }
    Vector x0(8, 0.);
    x0(3) = 2. * M_PI * f0;
    x0(4) = 1.0;
    x0(6) = 1.8;
    model->SetInitialState(x0);
    Matrix P0(8, 8, 0.);
    for (std::size_t i = 0; i < 8; ++i) P0(i, i) = 0.1;
    model->SetInitialCovariance(P0);
    Filters::EKF ekf(model.get());
    ekf.Forward(zs);
    const std::vector<double> steps = ekf.NIS();
    double nis = 0., freq = 0., amp = 0.;
    std::size_t counted = 0;
    for (std::size_t k = N / 2; k < N; ++k)
    {
        nis += steps[k];
        freq += ekf.PosteriorStates()[k](3) / (2. * M_PI);
        amp += std::exp(ekf.PosteriorStates()[k](0));
        ++counted;
    }
    nis /= counted;
    freq /= counted;
    amp /= counted;
    CHECK_CLOSE(freq, f0, 0.01, "polarization: the carrier frequency is found");
    CHECK_CLOSE(amp, 1., 0.05, "polarization: the amplitude is found");
    CHECK(nis > 3. && nis < 5.5, "polarization: mean NIS near m = 4");
    std::printf("    (mean NIS %.2f, frequency %.4f Hz, amplitude %.4f)\n", nis, freq, amp);

    return Check::Summary("20_gravitational_apps");
}
