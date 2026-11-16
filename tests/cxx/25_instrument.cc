/**
 * \file 25_instrument.cc
 * \brief Kalman::Instrument: a set of sensors with one name, one rate and one calibration.
 */
#include "Kalman/Kalman.h"
#include "Check.h"

#include <cmath>
#include <memory>
#include <stdexcept>

using namespace Kalman;

namespace
{
    template <class Fn>
    bool Throws(Fn fn)
    {
        try { fn(); }
        catch (const std::exception &) { return true; }
        return false;
    }

    /// A room: its temperature and its CO2 level, both slowly wandering. Its own
    /// sensor reads the CO2 only.
    std::shared_ptr<Models::Linear> Room()
    {
        auto room = std::make_shared<Models::Linear>("room", 2, 1);
        room->SetSampleRate(1.);
        Matrix H(1, 2, 0.); H(0, 1) = 1.;
        Matrix Q(2, 2, 0.); Q(0, 0) = 1e-4; Q(1, 1) = 1e-2;
        Matrix R(1, 1, 0.); R(0, 0) = 4.;
        room->SetF(Matrix::Identity(2)); room->SetH(H); room->SetQ(Q); room->SetR(R);
        Vector x0(2); x0(0) = 21.; x0(1) = 450.;
        Matrix P0(2, 2, 0.); P0(0, 0) = 4.; P0(1, 1) = 100.;
        room->SetInitialState(x0);
        room->SetInitialCovariance(P0);
        return room;
    }

    Sensor Reads(const std::string &name, std::size_t component, double variance)
    {
        Matrix G(1, 2, 0.); G(0, component) = 1.;
        Matrix R(1, 1, 0.); R(0, 0) = variance;
        return Sensor(name, G, R);
    }
}

int main()
{
    const Sensor probe = Reads("probe", 0, 0.04), humidity = Reads("humidity", 1, 1.);

    // --- composing sensors ------------------------------------------------------------
    {
        const Instrument both = probe | humidity;
        CHECK(both.name == "probe+humidity" && both.sensors.size() == 2, "s1 | s2: an instrument named after both");
        CHECK((both | Reads("third", 0, 1.)).sensors.size() == 3, "instrument | s3: one more sensor");
        const Instrument station = Compose({probe, humidity}, "station", 0.5);
        CHECK(station.name == "station" && station.Channels() == 2, "Compose: a name, and the channels of its sensors");
        const auto sensors = station.Sensors();
        CHECK(sensors[0].instrument == "station" && sensors[1].instrument == "station", "its sensors carry its name");
        CHECK(sensors[0].rate == 0.5 && !sensors[0].calibrated, "and its rate; no calibration unless asked");
        Sensor fast = probe; fast.rate = 2.;
        CHECK(Compose({fast, humidity}, "station", 0.5).Sensors()[0].rate == 2., "a sensor's own rate is kept");
        CHECK(Throws([&] { Compose({probe, probe}, "twice").Check(2); }), "refused: two sensors of one name");
        CHECK(Throws([&] { Instrument("", {probe}).Check(2); }), "refused: no name");
        CHECK(Throws([&] { Instrument("empty", {}).Check(2); }), "refused: no sensor");
        CHECK(Throws([&] { Compose({probe}, "narrow").Check(3); }), "refused: a sensor that does not fit the state");
    }

    // --- a model read by an instrument -------------------------------------------------
    auto room = Room();
    Instrument station = Compose({probe, humidity}, "station");
    {
        const std::unique_ptr<Model> plain = room->AddInstrument(station);
        CHECK(plain->NMeasure() == 3 && plain->NState() == 2, "AddInstrument: m grows by its channels");
        CHECK(plain->Structure() == room->Structure() && plain->DefaultEngine() == EngineKind::Linear,
              "uncalibrated matrices: still a model of matrices");
        CHECK(plain->ParameterNames() == room->ParameterNames(), "uncalibrated: no parameter added");
        const auto groups = plain->Instruments();
        CHECK(groups.size() == 2 && groups[0].name == "room" && groups[0].sensors.size() == 1,
              "Instruments: the model's own sensor, an instrument of one");
        CHECK(groups[1].name == "station" && groups[1].sensors.size() == 2, "Instruments: then the station and its two");
    }

    station.Calibrate(2., 1.5);
    const std::shared_ptr<Model> read = AddInstrument(room, station);
    CHECK(read->HasParameter("station.gain") && read->HasParameter("station.offset"),
          "calibrated: the model declares <name>.gain and <name>.offset");
    CHECK(read->Parameter("station.gain") == 2. && read->Parameter("station.offset") == 1.5, "with the values given");
    CHECK(read->DefaultEngine() == EngineKind::Extended && read->Structure() == StructureKind::Maps,
          "calibrated: gain * G x + offset is no matrix H");
    Vector x(2); x(0) = 20.; x(1) = 400.;
    {
        const Vector z = read->h(x, 0.);
        CHECK(z(0) == 400., "h: the model's own sensor is not calibrated");
        CHECK(z(1) == 2. * 20. + 1.5 && z(2) == 2. * 400. + 1.5, "h: every sensor of the instrument is");
        const Matrix H = read->JacobianH(x, 0.);
        CHECK(H(1, 0) == 2. && H(2, 1) == 2. && H(0, 1) == 1., "JacobianH: the rows carry the gain");
        read->SetParameter("station.offset", -1.);
        const Vector moved = read->h(x, 0.);
        CHECK(moved(1) == 39. && moved(2) == 799., "SetParameter: one value for all its sensors");
        const auto sensors = read->Sensors();
        CHECK(sensors[1].offset == -1. && sensors[2].offset == -1. && sensors[1].Read(x)(0) == 39.,
              "Sensors: each reads through the calibration");
        CHECK(read->Instruments()[1].calibrated && read->Instruments()[1].offset == -1., "Instruments: and reports it");
        read->SetParameter("station.offset", 1.5);
        CHECK(Throws([&] { read->SetParameter("station.bias", 0.); }), "refused: a parameter it does not declare");
    }

    // --- calibrating while measuring ------------------------------------------------------
    // One thermometer with an unknown offset: the offset and the temperature cannot
    // be told apart. A second, calibrated one settles it.
    Instrument thermo = Compose({probe}, "thermo").Calibrate(1., 0.);
    const std::shared_ptr<Model> alone = AddInstrument(Room(), thermo);
    const std::shared_ptr<Model> blind = Joint(alone, {{"thermo.offset", 2., 0.}});
    {
        const Observability seen(*blind, 50);
        CHECK(seen.NWeak() == 1, "one thermometer: one direction the data do not determine");
        CHECK(seen.Weakness(0) > 0.1 && seen.Weakness(2) > 0.1 && seen.Weakness(1) < 1e-6,
              "it mixes the temperature and the offset, not the CO2");
    }
    const std::shared_ptr<Model> checked = AddSensor(alone, Reads("reference", 0, 0.01));
    const std::shared_ptr<Model> model = Joint(checked, {{"thermo.offset", 2., 0.}});
    CHECK(Observability(*model, 50).IsObservable(), "with a reference thermometer: observable");
    CHECK(model->StateComponents()[2].name == "thermo.offset", "the offset is a named component of the state");

    const double offset = 1.5;
    const std::shared_ptr<Model> truth =
        AddSensor(AddInstrument(Room(), Compose({probe}, "thermo").Calibrate(1., offset)), Reads("reference", 0, 0.01));
    Sampler sampler(truth.get(), 20261117);
    const std::size_t N = 400;
    sampler.Generate(N, truth->X0());

    Filters::EKF ekf(model.get());
    const Record rec = ekf.Forward(sampler.Measurements());
    const double estimate = rec.State(N - 1)(2), sigma = std::sqrt(rec.Covariance(N - 1)(2, 2));
    std::printf("  thermo.offset: estimate %.4f +/- %.4f, truth %.2f; mean NIS %.3f\n", estimate, sigma, offset, ekf.MeanNIS());
    CHECK(std::fabs(estimate - offset) < 3. * sigma, "EKF: the offset is within 3 sigma of the truth");
    CHECK(sigma < 0.05, "EKF: and the data pin it down (sigma < 0.05 from 2)");
    CHECK(ekf.MeanNIS() > 2.4 && ekf.MeanNIS() < 3.6, "EKF: mean NIS consistent with m = 3");
    CHECK(checked->Parameter("thermo.offset") == 0., "EKF: the inner model is left as it was found");

    return Check::Summary("25_instrument");
}
