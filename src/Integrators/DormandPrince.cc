/**
 * *********************************************
 *
 * \file DormandPrince.cc
 * \brief Implementation of the adaptive RK45 step
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Integrators/DormandPrince.h"

#include <algorithm>
#include <cmath>

namespace Kalman::Integrators
{
    namespace
    {
        const double kC[7] = {0., 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1.};
        const double kA[7][6] = {
            {},
            {1. / 5.},
            {3. / 40., 9. / 40.},
            {44. / 45., -56. / 15., 32. / 9.},
            {19372. / 6561., -25360. / 2187., 64448. / 6561., -212. / 729.},
            {9017. / 3168., -355. / 33., 46732. / 5247., 49. / 176., -5103. / 18656.},
            {35. / 384., 0., 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84.},
        };
        const double kB5[7] = {35. / 384., 0., 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84., 0.};
        const double kB4[7] = {5179. / 57600., 0., 7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40.};
    }

    Tableau DormandPrince::GetTableau() const
    {
        Tableau tab;
        tab.c.assign(kC, kC + 7);
        tab.b.assign(kB5, kB5 + 7);
        tab.a.resize(7);
        for (std::size_t i = 0; i < 7; ++i) tab.a[i].assign(kA[i], kA[i] + i);
        return tab;
    }

    Vector DormandPrince::Step(const Derivative &f, const Vector &x0, const Vector &u,
                               double t0, double dt) const
    {
        Vector x = x0;
        const double target = t0 + dt;
        double t = t0, h = dt;
        const double sign = dt < 0. ? -1. : 1.;

        Vector k[7];
        for (std::size_t iter = 0; iter < fMaxSubsteps; ++iter)
        {
            if ((h > 0. && t >= target) || (h < 0. && t <= target)) break;
            h = sign * std::min(std::abs(h), std::abs(target - t));

            for (std::size_t i = 0; i < 7; ++i)
            {
                Vector xi = x;
                for (std::size_t j = 0; j < i; ++j)
                    if (kA[i][j] != 0.) xi += (h * kA[i][j]) * k[j];
                k[i] = f(xi, u, t + kC[i] * h);
            }

            Vector x5 = x, x4 = x;
            for (std::size_t i = 0; i < 7; ++i)
            {
                if (kB5[i] != 0.) x5 += (h * kB5[i]) * k[i];
                if (kB4[i] != 0.) x4 += (h * kB4[i]) * k[i];
            }

            double err = 0.;
            for (std::size_t i = 0; i < x.Size(); ++i)
            {
                const double scale = fAtol + fRtol * std::max(std::abs(x(i)), std::abs(x5(i)));
                const double e = (x5(i) - x4(i)) / scale;
                err += e * e;
            }
            err = std::sqrt(err) / std::max(std::sqrt(static_cast<double>(x.Size())), 1.);

            if (err <= 1. || std::abs(h) <= 1e-14 * std::abs(dt))
            {
                x = x5;
                t += h;
            }
            h *= std::min(5., std::max(0.2, 0.9 * std::pow(1. / std::max(err, 1e-12), 0.2)));
        }
        return x;
    }
}
