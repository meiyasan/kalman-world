/**
 * *********************************************
 *
 * \file Solver.cc
 * \brief Implementation of the Kalman::Solver base and the kind resolver
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Solver.h"
#include "Kalman/Solvers/FiniteDifference.h"
#include "Kalman/Solvers/Spectral.h"

#include <sstream>
#include <stdexcept>

namespace Kalman
{
    Solver::Solver(std::shared_ptr<Integrator> integrator)
        : fIntegrator(integrator ? std::move(integrator) : Integrators::Make(IntegratorKind::RK4))
    {
    }

    void Solver::SetIntegrator(std::shared_ptr<Integrator> integrator)
    {
        if (!integrator) throw std::runtime_error("Kalman::Solver: the integrator must not be null.");
        fIntegrator = std::move(integrator);
    }

    void Solver::SetIntegrator(IntegratorKind kind)
    {
        fIntegrator = Integrators::Make(kind);
    }

    Tableau Solver::ExplicitTableau() const
    {
        const Tableau tab = fIntegrator->GetTableau();
        if (tab.Empty())
            throw std::runtime_error(std::string("Kalman::Solver: the ") + fIntegrator->Title()
                                     + " has no explicit tableau; a field solver needs an explicit "
                                       "Runge-Kutta integrator (Euler / RK4 / DormandPrince).");
        return tab;
    }

    namespace Solvers
    {
        std::unique_ptr<Solver> Make(SolverKind kind, std::size_t n, double dt)
        {
            switch (kind)
            {
                case SolverKind::Spectral:         return std::make_unique<Spectral>(n, n, dt);
                case SolverKind::FiniteDifference: return std::make_unique<FiniteDifference>(n, n, dt);
                case SolverKind::FiniteVolume:
                case SolverKind::FiniteElement:    break;
            }
            std::ostringstream os;
            os << "Kalman::Solvers::Make: '" << ToString(kind) << "' is reserved ("
               << (kind == SolverKind::FiniteVolume ? "flux" : "mesh/basis")
               << " discretization); use Spectral or FiniteDifference.";
            throw std::runtime_error(os.str());
        }
    }
}
