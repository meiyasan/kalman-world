/**
 * \file 19_tracking_apps.cc
 * \brief Kalman::Apps::Tracking::DriftChamber: registered, propagates, and is found tracks in.
 */
#include "Kalman/Kalman.h"
#include "Kalman/Apps/Apps.h"
#include "Check.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace Kalman;

int main()
{
    std::printf("19_tracking_apps\n");
    auto &factory = Factory::Instance();
    CHECK(factory.Has("drift-chamber"), "drift chamber: registered");
    std::unique_ptr<Model> model = factory.Create("drift-chamber");
    auto *chamber = dynamic_cast<Apps::Tracking::DriftChamber *>(model.get());
    CHECK(chamber != nullptr, "drift chamber: the factory gives a DriftChamber");
    if (!chamber) return Check::Summary("19_tracking_apps");

    CHECK(model->NState() == 4 && model->NMeasure() == 2 && model->DefaultEngine() == EngineKind::Combinatorial,
          "drift chamber: 4 states, 2 measured, combinatorial engine");
    CHECK(model->StateComponents()[2].name == "px" && model->StateComponents()[2].unit == "GeV/c",
          "drift chamber: components are named");
    CHECK(chamber->Radii().size() == 8 && chamber->Radii().front() == 20. && chamber->Radii().back() == 104.,
          "drift chamber: 8 layers from 20 to 104 cm");

    // A 1 GeV/c track from the origin: it lands on each layer, and |p| is conserved.
    Vector s(4, 0.);
    s(2) = 0.8;
    s(3) = 0.6;
    bool onLayer = true, momentum = true;
    for (const double r : chamber->Radii())
    {
        s = model->f(s, Vector(), r);
        onLayer = onLayer && std::fabs(std::hypot(s(0), s(1)) - r) < 1e-9;
        momentum = momentum && std::fabs(std::hypot(s(2), s(3)) - 1.) < 1e-9;
    }
    CHECK(onLayer, "drift chamber: each propagation ends on its layer (1e-9 cm)");
    CHECK(momentum, "drift chamber: the field conserves |p| (1e-9)");

    // The radius of curvature: R [cm] = p / (kKappa q Bz).
    const double R = 1. / (Apps::Tracking::DriftChamber::kKappa * 1. * 1.5);
    // Centre of the circle through the origin with momentum (0.8, 0.6), turning clockwise for q > 0.
    const double cx = R * 0.6, cy = -R * 0.8;
    CHECK_CLOSE(std::hypot(s(0) - cx, s(1) - cy), R, 1e-6, "drift chamber: the arc is the circle of radius p / (0.003 q B)");

    Vector slow(4, 0.);
    slow(2) = 0.02;
    CHECK(std::isnan(model->f(slow, Vector(), 104.)(0)), "drift chamber: a track that curls back gives NaN");

    model->SetArray("radii", {10., 20., 30.});
    CHECK(chamber->Radii().size() == 3, "drift chamber: the layers are an array");

    Vector x0;
    Matrix P0;
    Vector hit(2);
    hit(0) = 12.;
    hit(1) = 16.;
    chamber->SeedState(hit, x0, P0);
    CHECK_CLOSE(x0(2), 0.6, 1e-12, "drift chamber: a seed points away from the origin");

    return Check::Summary("19_tracking_apps");
}
