/**
 * *********************************************
 *
 * \file Controller.cc
 * \brief Implementation of the Kalman::Controller base
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Controller.h"

#include <algorithm>
#include <stdexcept>

namespace Kalman
{
    Controller::Controller(Model *model, const Vector &xRef, const Matrix &Qcost, const Matrix &Rcost)
        : fModel(model)
    {
        if (!fModel) throw std::runtime_error("Kalman::Controller: a model is required.");
        const std::size_t n = fModel->NState(), l = NControl();

        fXRef = xRef.Empty() ? Vector(n, 0.) : xRef;
        if (fXRef.Size() != n) throw std::runtime_error("Kalman::Controller: x_ref must have n entries.");

        fQCost = Qcost.Empty() ? Matrix::Identity(n) : Qcost;
        if (fQCost.Rows() != n || fQCost.Cols() != n) throw std::runtime_error("Kalman::Controller: Qcost must be (n, n).");

        fRCost = Rcost.Empty() ? Matrix::Identity(l) : Rcost;
        if (fRCost.Rows() != l || fRCost.Cols() != l) throw std::runtime_error("Kalman::Controller: Rcost must be (l, l).");
    }

    std::size_t Controller::NControl() const
    {
        return std::max<std::size_t>(fModel->NControl(), 1);
    }

    void Controller::SetXRef(const Vector &xRef)
    {
        if (xRef.Size() != fModel->NState()) throw std::runtime_error("Kalman::Controller: x_ref must have n entries.");
        fXRef = xRef;
    }

    Vector Controller::Command(const Vector &xHat, const Matrix &P, double t, const Vector *nu)
    {
        Vector u = Policy(xHat, P, t, nu);
        if (u.Size() != NControl())
            throw std::runtime_error("Kalman::Controller: the policy returned " + std::to_string(u.Size())
                                     + " commands, expected " + std::to_string(NControl()) + ".");

        Update(xHat, P, t, nu);

        // Cost against what the actuator delivers, not the raw ask (see the header).
        const Vector applied = fModel->ClipControl(u);
        Vector e = xHat;
        e -= fXRef;
        double cost = 0.;
        for (std::size_t i = 0; i < e.Size(); ++i)
            for (std::size_t j = 0; j < e.Size(); ++j) cost += e(i) * fQCost(i, j) * e(j);
        for (std::size_t i = 0; i < applied.Size(); ++i)
            for (std::size_t j = 0; j < applied.Size(); ++j) cost += applied(i) * fRCost(i, j) * applied(j);

        fCostLog.emplace_back(t, cost);
        fControlLog.emplace_back(t, u);
        return u;
    }

    void Controller::Reset()
    {
        fCostLog.clear();
        fControlLog.clear();
    }

    double Controller::TotalCost() const
    {
        double total = 0.;
        for (const auto &kv : fCostLog) total += kv.second;
        return total;
    }
}
