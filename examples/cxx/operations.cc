/**
 * \file operations.cc
 * \brief The model operations, one after the other, on a cart.
 *
 * An operation takes models and gives back a model; the filter is always the
 * ordinary one. This is the C++ side of `examples/python/operations.py`: the
 * same names, the same cart, the same five steps.
 *
 *     AddSensor      one more sensor on the same state
 *     Lag            the last states carried along: a sensor that reports late
 *     AddInstrument  sensors under one name, with a calibration
 *     Joint          a parameter estimated with the state
 *     Compose        two systems side by side
 *     Route          one system, two regimes
 *
 *     make build && make examples
 *     ./var/build/bin/operations
 */
#include <Kalman/Kalman.h>
#include <Kalman/Operations.h>

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace Kalman;

namespace
{
    const double FS = 10.;

    /// Position and velocity; a noisy position sensor. \p q is the variance of the unmodelled acceleration.
    std::shared_ptr<Models::Linear> Cart(const std::string &name = "cart", double q = 1.)
    {
        auto cart = std::make_shared<Models::Linear>(name, 2, 1);
        cart->SetSampleRate(FS);
        const double dt = cart->DeltaT();

        Matrix F = Matrix::Identity(2);
        F(0, 1) = dt;
        cart->SetF(F);

        Matrix H(1, 2, 0.);
        H(0, 0) = 1.;
        cart->SetH(H);

        Matrix Q(2, 2, 0.);
        Q(0, 0) = q * dt * dt * dt * dt / 4.;
        Q(0, 1) = Q(1, 0) = q * dt * dt * dt / 2.;
        Q(1, 1) = q * dt * dt;
        cart->SetQ(Q);

        Matrix R(1, 1, 0.);
        R(0, 0) = 0.25;
        cart->SetR(R);

        Vector x0(2);
        x0(0) = 0.;
        x0(1) = 1.;
        cart->SetInitialState(x0);
        cart->SetInitialCovariance(Matrix::Identity(2));
        return cart;
    }

    /// A sensor that reads the position, with variance \p variance.
    Sensor Position(const std::string &name, double variance)
    {
        Matrix G(1, 2, 0.);
        G(0, 0) = 1.;
        Matrix R(1, 1, 0.);
        R(0, 0) = variance;
        return Sensor(name, G, R);
    }
}

int main()
{
    const std::size_t N = 1000;

    // --- AddSensor, then Lag: a precise camera that reports 0.3 s late -----------------------
    // A late sensor reads a past state, so the model has to carry it: Lag(model, 3)
    // stacks the last three states under the present one.
    {
        Sensor camera = Position("camera", 0.01);
        camera.delay = 0.3;
        const std::shared_ptr<Models::Lag> late = Lag(AddSensor(Cart(), camera), 3);

        Sampler sampler(late.get(), 1);
        sampler.Generate(N, late->X0());
        Filters::KF both(late.get());
        const Record record = both.Forward(sampler.Measurements());

        // the same readings without the camera: the first channel only, on the plain cart
        auto cart = Cart();
        std::vector<Vector> first;
        for (const Vector &z : sampler.Measurements())
        {
            Vector one(1);
            one(0) = z(0);
            first.push_back(one);
        }
        Filters::KF alone(cart.get());
        const Record without = alone.Forward(first);

        std::printf("AddSensor + Lag:  position error %.3f with the late camera, %.3f without\n",
                    record.RMSE(0, sampler.Truth()), without.RMSE(0, sampler.Truth()));
        std::printf("                  the position three steps ago is known to %.3f (fixed-lag smoothing)\n",
                    record.RMSE(6, sampler.Truth()));
    }

    // --- AddInstrument, then Joint: the offset of an uncalibrated gauge ----------------------------
    // An instrument is sensors under one name. Calibrate() makes it carry a gain and
    // an offset, which the model then declares as parameters; Joint estimates one.
    {
        const Sensor probe = Position("probe", 0.04);
        const std::shared_ptr<Model> biased = AddInstrument(Cart(), Compose({probe}, "gauge").Calibrate(1., 0.8));
        Sampler sampler(biased.get(), 2);
        sampler.Generate(N, biased->X0());

        const std::shared_ptr<Model> assumed = AddInstrument(Cart(), Compose({probe}, "gauge").Calibrate(1., 0.));
        const std::shared_ptr<Models::Joint> model = Joint(assumed, {{"gauge.offset", 2., 0.}});
        Filters::EKF ekf(model.get());
        const Record record = ekf.Forward(sampler.Measurements());
        std::printf("Joint:            gauge.offset = %.3f +/- %.3f (truth 0.8)\n",
                    record.State(N - 1)(2), record.Sigma(2)[N - 1]);
    }

    // --- Compose: two carts in one filter ----------------------------------------------------------------
    {
        const std::shared_ptr<Models::Composite> pair = Compose({Cart("left"), Cart("right")});
        Sampler sampler(pair.get(), 3);
        sampler.Generate(N, pair->X0());
        Filters::KF kf(pair.get());
        const Record record = kf.Forward(sampler.Measurements());
        std::printf("Compose:          n = %zu, m = %zu, parts %s and %s; position errors %.3f and %.3f\n",
                    pair->NState(), pair->NMeasure(), pair->PartName(0).c_str(), pair->PartName(1).c_str(),
                    record.RMSE(0, sampler.Truth()), record.RMSE(2, sampler.Truth()));
    }

    // --- Route: a cart that is quiet, then shaken -----------------------------------------------------------
    // The same state seen by two models; the interacting engine runs both and keeps
    // the probability of each. Filter picks that engine from the model.
    {
        auto quiet = Cart("quiet", 0.01), shaken = Cart("shaken", 100.);
        Sampler first(quiet.get(), 4);
        first.Generate(N / 2, quiet->X0());
        Sampler second(shaken.get(), 5);
        second.Generate(N / 2, first.Truth().back());
        std::vector<Vector> zk = first.Measurements();
        zk.insert(zk.end(), second.Measurements().begin(), second.Measurements().end());

        Matrix Pi(2, 2, 0.02);
        Pi(0, 0) = Pi(1, 1) = 0.98;
        const std::shared_ptr<Models::Router> modes = Route({quiet, shaken}, Pi);
        Filter filter(modes.get());
        filter.Forward(zk);

        const Matrix mu = dynamic_cast<const Engines::Interacting &>(*filter.GetEngine()).ModeHistory();
        double before = 0., after = 0.;
        for (std::size_t k = 0; k < N / 2; ++k)
        {
            before += mu(k, 1);
            after += mu(N / 2 + k, 1);
        }
        std::printf("Route:            probability of the shaken regime: %.2f in the first half, %.2f in the second\n",
                    before / (N / 2), after / (N / 2));
    }

    return 0;
}
