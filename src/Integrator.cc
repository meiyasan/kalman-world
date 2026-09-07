/**
 * *********************************************
 *
 * \file Integrator.cc
 * \brief Implementation of the Kalman::Integrator base and the kind resolver
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Integrator.h"
#include "Kalman/Integrators/DormandPrince.h"
#include "Kalman/Integrators/Euler.h"
#include "Kalman/Integrators/RK4.h"

#include <sstream>
#include <stdexcept>

namespace Kalman
{
    Vector Integrator::Integrate(const Derivative &f, const Vector &x, const Vector &u,
                                 double t, double dt, std::size_t n) const
    {
        Vector out = x;
        for (std::size_t i = 0; i < n; ++i)
        {
            out = Step(f, out, u, t, dt);
            t += dt;
        }
        return out;
    }

    namespace Integrators
    {
        std::unique_ptr<Integrator> Make(IntegratorKind kind)
        {
            switch (kind)
            {
                case IntegratorKind::Euler:         return std::make_unique<Euler>();
                case IntegratorKind::RK4:           return std::make_unique<RK4>();
                case IntegratorKind::DormandPrince: return std::make_unique<DormandPrince>();
                case IntegratorKind::ImplicitEuler: break;
            }
            std::ostringstream os;
            os << "Kalman::Integrators::Make: '" << ToString(kind) << "' is reserved "
               << "(backward Euler needs a per-step Newton solve); use RK4 or DormandPrince.";
            throw std::runtime_error(os.str());
        }
    }
}
