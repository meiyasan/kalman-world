/**
 * *********************************************
 *
 * \file Tabulated.cc
 * \brief Implementation of the Kalman::Models::Tabulated class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Models/Tabulated.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Kalman::Models
{
    const Matrix *Tabulated::At(const char *which, double t) const
    {
        const auto it = fTables.find(which);
        if (it == fTables.end() || it->second.empty()) return nullptr;
        const double k = std::round((t - T0()) * SampleRate());
        const double last = static_cast<double>(it->second.size() - 1);
        return &it->second[static_cast<std::size_t>(std::min(std::max(k, 0.), last))];
    }

    void Tabulated::SetTable(const std::string &which, std::vector<Matrix> table)
    {
        if (which != "F" && which != "H" && which != "Q" && which != "R")
            throw std::invalid_argument("Kalman::Models::Tabulated: a table is for F, H, Q or R, not '" + which + "'.");
        if (table.empty()) throw std::invalid_argument("Kalman::Models::Tabulated: the table of " + which + " is empty.");
        const std::size_t rows = (which == "F" || which == "Q") ? NState() : NMeasure();
        const std::size_t cols = (which == "R") ? NMeasure() : NState();
        for (const auto &M : table)
            if (M.Rows() != rows || M.Cols() != cols)
                throw std::invalid_argument("Kalman::Models::Tabulated: the table of " + which + " needs matrices of "
                                            + std::to_string(rows) + " x " + std::to_string(cols) + ".");
        fTables[which] = std::move(table);
        SetStructure(StructureKind::TimeVarying);
    }

    const std::vector<Matrix> &Tabulated::Table(const std::string &which) const
    {
        static const std::vector<Matrix> none;
        const auto it = fTables.find(which);
        return it == fTables.end() ? none : it->second;
    }

    std::size_t Tabulated::Steps() const
    {
        std::size_t steps = 0;
        for (const auto &t : fTables) steps = std::max(steps, t.second.size());
        return steps;
    }

    void Tabulated::SetComponent(bool state, std::size_t i, const std::string &name, const std::string &unit)
    {
        std::vector<Component> all = state ? StateComponents() : MeasurementComponents();
        if (i >= all.size())
            throw std::out_of_range("Kalman::Models::Tabulated: there is no component " + std::to_string(i) + ".");
        all[i] = Component{name, unit};
        if (state) DeclareState(std::move(all));
        else DeclareMeasurement(std::move(all));
    }

    void Tabulated::SetComponents(std::vector<Component> state, std::vector<Component> measurement)
    {
        if (!state.empty()) DeclareState(std::move(state));
        if (!measurement.empty()) DeclareMeasurement(std::move(measurement));
    }

    Matrix Tabulated::F(double t) const { const Matrix *M = At("F", t); return M ? *M : Linear::F(t); }
    Matrix Tabulated::H(double t) const { const Matrix *M = At("H", t); return M ? *M : Linear::H(t); }
    Matrix Tabulated::Q(double t) const { const Matrix *M = At("Q", t); return M ? *M : Linear::Q(t); }
    Matrix Tabulated::R(double t) const { const Matrix *M = At("R", t); return M ? *M : Linear::R(t); }
}
