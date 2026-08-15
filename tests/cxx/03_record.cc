/**
 * \file 03_record.cc
 * \brief The record's derived views, the prior pass, and error reporting.
 */
#include "Kalman/Kalman.h"
#include "Check.h"
#include "Fixtures.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

using namespace Kalman;

int main()
{
    Fixture::ConstantVelocity model(1, 100.);
    model.SetProcessNoise(0.5);
    model.SetMeasurementNoise(0.2);

    Sampler sampler(&model, 31337);
    sampler.Generate(400);

    Filters::KF filter(&model);
    const Record rec = filter.Forward(sampler.Measurements());

    // --- derived series -------------------------------------------------------
    {
        const std::vector<double> t = rec.Times();
        CHECK(t.size() == 400, "record: one time tag per step");
        CHECK_CLOSE(t.front(), 0., 1e-12, "record: starts at t0");
        CHECK_CLOSE(t.back(), 3.99, 1e-9, "record: ends at (N-1)/fs");

        const std::vector<double> sigma = rec.Sigma(0);
        CHECK_CLOSE(sigma[10], std::sqrt(rec.Covariance(10)(0, 0)), 1e-12,
                    "record: sigma is sqrt(P_ii)");

        const std::vector<double> z = rec.MeasurementSeries(0);
        CHECK(z.size() == 400, "record: measurement series spans the run");
        CHECK_CLOSE(z[7], rec.Measurement(7)(0), 1e-12, "record: measurement series matches");

        CHECK(rec.RMSE(0, sampler.Truth()) > 0., "record: RMSE against truth is positive");
    }

    // --- the prior record is a separate, wider estimate -----------------------
    {
        const Record prior = filter.PriorRecord();
        CHECK(prior.Flavor() == StateFlavor::Prior, "prior: record tagged Prior");
        CHECK(prior.NSteps() == rec.NSteps(), "prior: same length as the posterior");
        CHECK(prior.Covariance(200)(0, 0) > rec.Covariance(200)(0, 0),
              "prior: predicted covariance exceeds the corrected one");
    }

    // --- printing ------------------------------------------------------------
    {
        std::ostringstream os;
        os << rec;
        const std::string text = os.str();
        CHECK(text.find("Posterior") != std::string::npos, "record: prints its flavour");
        CHECK(text.find("400") != std::string::npos, "record: prints its length");
    }

    // --- errors are exceptions with a readable message -------------------------
    {
        bool threw = false;
        try
        {
            const Matrix bad(3, 3, 0.);
            model.SetF(bad);
        }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "errors: a wrong-shaped F throws");

        threw = false;
        try { rec.State(10000); }
        catch (const std::out_of_range &) { threw = true; }
        CHECK(threw, "errors: an out-of-range step throws");

        threw = false;
        try
        {
            Fixture::ConstantVelocity fresh(1, 100.);
            Filters::KF kf(&fresh);
            kf.Backward();                          // no forward pass yet
        }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "errors: Backward() before Forward() throws");

        threw = false;
        try
        {
            Fixture::ConstantVelocity fresh(1, 100.);
            Filters::KF kf(&fresh);
            std::vector<Vector> wrong(3, Vector(5, 0.));   // 5 channels, model has 1
            kf.Forward(wrong);
        }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "errors: a wrong-width measurement throws");

        threw = false;
        try { Filter f(nullptr); }
        catch (const std::runtime_error &) { threw = true; }
        CHECK(threw, "errors: a null model throws");
    }

    return Check::Summary("record");
}
