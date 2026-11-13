/**
 * \file 18_gps_apps.cc
 * \brief Kalman::Apps::GPS::Receiver, and the named arrays it is configured with.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Apps/Apps.h"
#include "Check.h"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Kalman;

namespace
{
    template <class F>
    std::string Thrown(F &&body)
    {
        try { body(); }
        catch (const std::exception &e) { return e.what(); }
        return "";
    }
}

int main()
{
    std::printf("18_gps_apps\n");
    auto &factory = Factory::Instance();
    CHECK(factory.Has("gps"), "gps: registered");
    std::unique_ptr<Model> model = factory.Create("gps");
    auto *receiver = dynamic_cast<Apps::GPS::Receiver *>(model.get());
    CHECK(receiver != nullptr, "gps: the factory gives a Receiver");
    if (!receiver) return Check::Summary("18_gps_apps");

    CHECK(model->NState() == 5 && model->NMeasure() == 8, "gps: 5 states, 8 pseudoranges by default");
    CHECK(model->Dynamics() == DynamicsKind::Continuous && model->Differentiator() == DifferentiatorKind::Analytic,
          "gps: stated in continuous time, analytic Jacobians");
    CHECK(model->StateComponents()[3].name == "clock_bias" && model->MeasurementComponents()[7].name == "pseudorange_7",
          "gps: components are named");
    CHECK(model->ArrayNames().size() == 6 && model->Array("eccentricity").size() == 8, "gps: six arrays, one entry per satellite");

    // Every satellite of the default set is on its orbit and above the receiver's horizon.
    const Vector x0 = model->X0();
    bool orbit = true, above = true;
    for (std::size_t i = 0; i < receiver->NSatellites(); ++i)
    {
        double s[3], r = 0., dot = 0.;
        receiver->Position(i, 0., s);
        for (std::size_t k = 0; k < 3; ++k)
        {
            r += s[k] * s[k];
            dot += (s[k] - x0(k)) * x0(k);
        }
        orbit = orbit && std::fabs(std::sqrt(r) / Apps::GPS::Receiver::kOrbitRadius - 1.) < 0.011;   // e = 0.01
        above = above && dot > 0.;
    }
    CHECK(orbit, "gps: satellites at the orbit radius, within the eccentricity");
    CHECK(above, "gps: the default satellites are above the horizon");

    // The analytic H against central differences of h.
    Vector x = x0;
    x(0) += 12.;
    x(3) = 4.;
    const Matrix H = model->JacobianH(x, 300.);
    const Matrix numeric = Differentiators::Jacobian(
        DifferentiatorKind::Central, [&model](const Vector &p) { return model->h(p, 300.); }, x, 1e-3);
    double worst = 0.;
    for (std::size_t i = 0; i < H.Rows(); ++i)
        for (std::size_t j = 0; j < H.Cols(); ++j) worst = std::max(worst, std::fabs(H(i, j) - numeric(i, j)));
    // A range of 2e7 m carries 4e-9 m of rounding, so the step is a large one (1e-3 of each component).
    CHECK_CLOSE(worst, 0., 1e-5, "gps: analytic H agrees with differences of h");
    CHECK_CLOSE(model->F(0.)(3, 4), 1., 0., "gps: the clock bias follows the drift over one second");

    // Fewer satellites: the model resizes, and arrays that disagree are said.
    model->SetArray("eccentricity", std::vector<double>(4, 0.01));
    CHECK(!Thrown([&model] { model->CheckShapes(); }).empty(), "arrays: a disagreement is said by CheckShapes()");
    for (const char *name : {"semi_major_axis", "inclination", "raan", "arg_perigee", "mean_anomaly"})
    {
        std::vector<double> values = model->Array(name);
        values.resize(4);
        model->SetArray(name, values);
    }
    CHECK(model->NMeasure() == 4 && model->R(0.).Rows() == 4 && model->h(x, 0.).Size() == 4, "arrays: the model resized to 4 satellites");
    CHECK(Thrown([&model] { model->CheckShapes(); }).empty(), "arrays: consistent again");
    CHECK(Thrown([&model] { model->SetArray("orbits", {1.}); }).find("it declares") != std::string::npos,
          "arrays: an unknown name lists the known ones");

    // A run: the receiver starts 60 m off and comes back.
    std::vector<Vector> zk;
    for (int k = 0; k < 120; ++k) zk.push_back(model->h(x0, static_cast<double>(k)));
    Vector start = x0;
    start(0) += 60.;
    model->SetInitialState(start);
    Filter filter(model.get());
    const IO::Record record = filter.Forward(zk);
    CHECK_CLOSE(record.State(119)(0), x0(0), 0.5, "gps: 60 m off at the start, within 0.5 m after 120 s");

    return Check::Summary("18_gps_apps");
}
