/**
 * \file toy_sensor.cc
 * \brief The toy model in C++: simulate a test sensor, filter it, save both.
 *
 * Loads `libKalmanPlugin_toy` (toy_plugin.cc), which defines the cooling-coffee
 * model, and never sees its class: the model is found by name in the factory.
 * The program then
 *
 *   1. simulates a **test sensor** — half an hour of thermometer readings of a
 *      cup that starts at 90 degC and cools at k = 0.002 /s — with the library's
 *      sampler and a fixed seed;
 *   2. runs the extended Kalman filter over the readings;
 *   3. writes `toy_sensor.csv` (time, true T, true k, reading) and
 *      `toy_estimate.csv` (time, estimated T and k with their sigmas).
 *
 * `examples/python/toy_sensor.py` reads those two files, runs the *same C++
 * model* from Python on the *same readings*, and compares.
 *
 *     make build && make examples
 *     ./var/build/bin/toy_sensor [output directory] [plugin library]
 */
#include <Kalman/Kalman.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#ifndef KALMAN_TOY_PLUGIN
#define KALMAN_TOY_PLUGIN "libKalmanPlugin_toy"
#endif

int main(int argc, char **argv)
{
    const std::string outdir = argc > 1 ? argv[1] : ".";
    const std::string plugin = argc > 2 ? argv[2] : KALMAN_TOY_PLUGIN;

    // --- the model: loaded, not compiled in ------------------------------------
    Kalman::Factory &factory = Kalman::Factory::Instance();
    factory.LoadPlugin(plugin);
    std::unique_ptr<Kalman::Model> coffee = factory.Create("toy-coffee");
    coffee->SetParameter("room", 21.);                 // degC; a named parameter of the model
    std::printf("model   : %s\n", coffee->Describe().c_str());

    // --- the test sensor ---------------------------------------------------------
    // The truth starts somewhere the filter does not know: hotter, and cooling
    // twice as fast as its initial guess.
    const std::size_t steps = 1800;                    // 30 minutes at 1 Hz
    Kalman::Vector truth0(2);
    truth0(0) = 90.;
    truth0(1) = 2e-3;
    Kalman::Sampler sensor(coffee.get(), 2026);        // fixed seed: the same readings every run
    sensor.Generate(steps, truth0);

    // --- the filter --------------------------------------------------------------
    Kalman::Filters::EKF ekf(coffee.get());
    const Kalman::Record estimate = ekf.Forward(sensor.Measurements());

    // --- save the readings and the estimate ---------------------------------------
    const std::string sensorFile = outdir + "/toy_sensor.csv";
    const std::string estimateFile = outdir + "/toy_estimate.csv";
    std::FILE *fs = std::fopen(sensorFile.c_str(), "w");
    std::FILE *fe = std::fopen(estimateFile.c_str(), "w");
    if (!fs || !fe)
    {
        std::fprintf(stderr, "cannot write into '%s'\n", outdir.c_str());
        return 1;
    }
    std::fprintf(fs, "time,true_T,true_k,reading\n");
    std::fprintf(fe, "time,T,k,sigma_T,sigma_k\n");
    const double dt = coffee->DeltaT();
    for (std::size_t i = 0; i < steps; ++i)
    {
        const Kalman::Vector &x = ekf.PosteriorStates()[i];
        const Kalman::Matrix &P = ekf.PosteriorCovariances()[i];
        std::fprintf(fs, "%.17g,%.17g,%.17g,%.17g\n", i * dt, sensor.Truth()[i](0), sensor.Truth()[i](1),
                     sensor.Measurements()[i](0));
        std::fprintf(fe, "%.17g,%.17g,%.17g,%.17g,%.17g\n", i * dt, x(0), x(1), std::sqrt(P(0, 0)),
                     std::sqrt(P(1, 1)));
    }
    std::fclose(fs);
    std::fclose(fe);

    // --- what the filter achieved -----------------------------------------------------
    double raw = 0.;
    for (std::size_t i = 0; i < steps; ++i)
        raw += std::pow(sensor.Measurements()[i](0) - sensor.Truth()[i](0), 2);
    raw = std::sqrt(raw / static_cast<double>(steps));

    const Kalman::Vector &last = ekf.PosteriorStates().back();
    const Kalman::Matrix &lastP = ekf.PosteriorCovariances().back();
    std::printf("sensor  : %zu readings, thermometer error %.3f degC (RMS)\n", steps, raw);
    std::printf("filter  : temperature error %.3f degC (RMS), mean NIS %.2f (1 is ideal)\n",
                estimate.RMSE(0, sensor.Truth()), ekf.MeanNIS());
    std::printf("learned : k = %.6f +/- %.6f /s   (truth %.6f, first guess %.6f)\n", last(1),
                std::sqrt(lastP(1, 1)), sensor.Truth().back()(1), coffee->X0()(1));
    std::printf("wrote   : %s\n          %s\n", sensorFile.c_str(), estimateFile.c_str());
    return 0;
}
