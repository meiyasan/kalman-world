/**
 * *********************************************
 *
 * \file KolmogorovFlow.h
 * \brief Header of the Kalman::Apps::Weather::KolmogorovFlow class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Apps::Weather::KolmogorovFlow
 * \brief A 2-D vorticity field observed at sparse probes — the twin of
 * `kalman.apps.weather.KolmogorovFlowModel`.
 *
 * \details The state is the whole vorticity field on a periodic `grid x grid`
 * mesh (n = grid^2, row-major: the component `w(i,j)` is row i, along y, and
 * column j, along x). One step of the filter is one assimilation cycle:
 * `cycle` steps of Kalman::Solvers::Vorticity,
 *
 * \code{.unparsed}
 *   dw/dt = -u.grad(w) - beta v + nu lap(w) + F
 * \endcode
 *
 * The measurement is the field at the probes (`w(i,j)` per probe), or with
 * `gradient` set the two derivatives there (`dw/dx(i,j)` for every probe,
 * then `dw/dy(i,j)`): a linear map of the field either way, which H() states.
 *
 * It is an ensemble model. Besides `f` and `h` it implements
 * Engines::EnsembleModel: the forecast of the whole ensemble (the members in
 * parallel, `threads` at a time), its observation, the Gaspari-Cohn tapers
 * from the torus distances between cells and probes (`loc_radius`, in grid
 * cells) and the process noise as a diagonal, so that a cycle forms nothing
 * of size n x n.
 *
 * | Parameter    | Meaning                                                 |
 * |--------------|---------------------------------------------------------|
 * | `grid`       | cells along each side; n = grid^2                        |
 * | `nu`         | viscosity                                               |
 * | `beta`       | planetary-vorticity gradient (0: Kolmogorov flow)        |
 * | `dt`         | solver time step                                        |
 * | `cycle`      | solver steps per filter step                            |
 * | `length_y`, `length_x` | the periodic box                              |
 * | `integrator` | Kalman::IntegratorKind of the time stepping (1: RK4)     |
 * | `loc_radius` | Gaspari-Cohn length, grid cells (zero beyond twice it)   |
 * | `gradient`   | 0: the field at the probes, 1: its two derivatives       |
 * | `threads`    | members advanced at once (0: one per hardware thread)    |
 *
 * | Array              | Meaning                                         |
 * |--------------------|-------------------------------------------------|
 * | `probes`           | the probed cells, as flat indices i * grid + j   |
 * | `forcing_re`, `forcing_im` | the transform of the forcing F (n each; empty: none) |
 * | `process_variance` | the diagonal of Q (n values, one for all, or none) |
 *
 * The defaults are those of the Python class (64 x 64, nu = 1/150, the
 * forcing 2 cos(4 y), 30 steps of 0.005), except the probes: the Python class
 * draws its 100 with numpy and pushes them, the registered model observes a
 * regular 10 x 10 lattice. The initial ensemble is not drawn here: the Python
 * class takes it from its climatology and hands it to the engine
 * (Engines::Ensemble::SetMembers()).
 *
 * *********************************************
 */

#ifndef KALMAN_APPS_WEATHER_KOLMOGOROVFLOW_H
#define KALMAN_APPS_WEATHER_KOLMOGOROVFLOW_H

#include "Kalman/Apps/Model.h"

#include <Kalman/Engines/Ensemble.h>
#include <Kalman/Solvers/Vorticity.h>

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Kalman::Apps::Weather
{
    class KolmogorovFlow : public Kalman::Apps::Model, public Kalman::Engines::EnsembleModel
    {
        protected:
            std::size_t fGrid = 0;
            std::size_t fCycle = 30;
            bool fGradient = false;
            double fLocRadius = 8.;
            std::size_t fThreads = 0;
            std::string fUnit;                                  // of the vorticity ("" when non-dimensional)
            std::vector<std::size_t> fProbes;
            Vector fVariance;                                   // diagonal of Q
            std::shared_ptr<const Kalman::Solvers::Vorticity> fSolver;

            // Built on first use after a change: the observation operator and the tapers.
            mutable std::mutex fCacheMutex;
            mutable Matrix fObserve;
            mutable Matrix fRhoXY, fRhoYY;

        public:
            /// The Kolmogorov flow of the Python class's defaults.
            KolmogorovFlow();

            const char *Title() const override { return "Kolmogorov Flow"; }
            const char *Description() const override
            { return "2D Navier-Stokes vorticity field (Kolmogorov forcing) with sparse point probes."; }
            const char *Version() const override { return "1.0"; }

            std::size_t Grid() const { return fGrid; }
            const std::vector<std::size_t> &Probes() const { return fProbes; }
            const Kalman::Solvers::Vorticity &Solver() const { return *fSolver; }

            // --- the model ------------------------------------------------------
            /// One cycle of the solver.
            Vector f(const Vector &x, const Vector &u, double t = 0.) const override;
            /// The field (or its gradient) at the probes.
            Vector h(const Vector &x, double t = 0.) const override;
            /// The identity: the chaotic flow has no usable dense Jacobian (as in Python).
            Matrix F(double t = 0.) const override;
            /// The observation operator, a linear map of the field.
            Matrix H(double t = 0.) const override;
            /// The diagonal `process_variance` as a matrix.
            Matrix Q(double t = 0.) const override;

            // --- what the ensemble engine asks of a field --------------------------
            void Forecast(const Kalman::Model &model, std::vector<Vector> &members,
                          const Vector &u, double t) const override;
            void Observe(const Kalman::Model &model, const std::vector<Vector> &members, double t,
                         std::vector<Vector> &out) const override;
            bool Localization(Matrix &rhoXY, Matrix &rhoYY) const override;
            bool ProcessNoiseVariances(double t, Vector &variances) const override;

            /// Distance between two cells on the periodic grid, in cells.
            double TorusDistance(std::size_t a, std::size_t b) const;

            /// The probes, the arrays and R against the grid and the measurement.
            void CheckShapes() const override;

        protected:
            /// The Kolmogorov defaults under another registered name.
            explicit KolmogorovFlow(const std::string &name);
            void ParametersChanged() override;
            /// The physics' defaults, for a subclass (the barotropic flow) to restate.
            void Configure(double nu, double beta, double dt, std::size_t cycle, double ly, double lx,
                           bool forcing);

        private:
            Field AsField(const Vector &x) const;
            void Measure(const Field &w, Vector &z) const;
            std::size_t Workers(std::size_t jobs) const;
    };
}

#endif // KALMAN_APPS_WEATHER_KOLMOGOROVFLOW_H
