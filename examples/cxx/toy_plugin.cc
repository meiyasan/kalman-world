/**
 * \file toy_plugin.cc
 * \brief A toy model written once in C++, for toy_sensor.cc and toy_sensor.py.
 *
 * A cup of coffee cools toward the temperature of the room (Newton's law of
 * cooling), and nobody knows how fast: the cooling rate is part of the state.
 *
 *     state        x = [T, k]      temperature [degC], cooling rate [1/s]
 *     dynamics     T' = -k (T - T_room),   k' = 0
 *     measurement  z = T           a thermometer
 *
 * The product k (T - T_room) makes it nonlinear, so it runs on the extended
 * engine. The room temperature is a *named parameter*: whoever loads the model
 * (the C++ program, the Python script) can set it without recompiling.
 *
 * This file is built as a shared library, `libKalmanPlugin_toy`. It is the
 * only place the physics is written; the two programs below load it.
 *
 *     make build && make examples
 *     ./var/build/bin/toy_sensor                 # C++: simulate the sensor, filter, write CSV
 *     python examples/python/toy_sensor.py       # Python: the same model on the same readings
 */
#include <Kalman/Factory.h>
#include <Kalman/Models/Extended.h>

#include <cmath>

namespace
{
    class CoolingCoffee : public Kalman::Models::Extended
    {
        private:
            double fRoom = 21.;   // degC

        public:
            CoolingCoffee() : Kalman::Models::Extended("toy-coffee", 2, 1)
            {
                DeclareParameter("room", 21.);
                ParametersChanged();
                SetSampleRate(1.);                       // one thermometer reading a second

                // The model is nearly exact: a little slack on T, and a very slow
                // random walk on k (the cup's cooling rate barely changes).
                Kalman::Matrix Q(2, 2, 0.);
                Q(0, 0) = 1e-4;
                Q(1, 1) = 1e-12;
                SetQ(Q);

                Kalman::Matrix R(1, 1, 0.5 * 0.5);       // thermometer noise: 0.5 degC
                SetR(R);

                // What the filter believes before the first reading: hot coffee,
                // and a cooling rate that is only a guess.
                Kalman::Vector x0(2);
                x0(0) = 80.;
                x0(1) = 1e-3;
                SetInitialState(x0);
                Kalman::Matrix P0(2, 2, 0.);
                P0(0, 0) = 25.;
                P0(1, 1) = 1e-6;
                SetInitialCovariance(P0);
            }

            const char *Title() const override { return "Cooling Coffee (toy)"; }
            const char *Description() const override
            { return "Newton's law of cooling with an unknown rate; a thermometer reads the temperature."; }

            // The exact solution over one step, so no integrator is needed.
            Kalman::Vector f(const Kalman::Vector &x, const Kalman::Vector &, double) const override
            {
                Kalman::Vector out(2);
                out(0) = fRoom + (x(0) - fRoom) * std::exp(-x(1) * DeltaT());
                out(1) = x(1);
                return out;
            }

            Kalman::Vector h(const Kalman::Vector &x, double) const override
            {
                Kalman::Vector z(1);
                z(0) = x(0);
                return z;
            }

        protected:
            void ParametersChanged() override { fRoom = Parameter("room"); }
    };
}

KALMAN_REGISTER_MODEL(CoolingCoffee, "toy-coffee");
