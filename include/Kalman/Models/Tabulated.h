/**
 * *********************************************
 *
 * \file Tabulated.h
 * \brief Header of the Kalman::Models::Tabulated class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Models::Tabulated
 * \brief A linear model given by its matrices, constant or one per sample.
 *
 * \details What a model file is read into (Kalman::IO::ModelFile). Each of
 * F, H, Q and R is either one matrix, or a table with one matrix per sample:
 * sample k is the one at time `T0() + k / fs`, and a time before the first or
 * after the last sample reads the first or the last entry.
 *
 * With no table it is a plain linear model of constant matrices.
 *
 * *********************************************
 */

#ifndef KALMAN_MODELS_TABULATED_H
#define KALMAN_MODELS_TABULATED_H

#include "Kalman/Models/Linear.h"

#include <map>
#include <string>
#include <vector>

namespace Kalman::Models
{
    class Tabulated : public Linear
    {
        private:
            std::map<std::string, std::vector<Matrix>> fTables;   // "F", "H", "Q", "R"

            const Matrix *At(const char *which, double t) const;

        public:
            Tabulated(std::string name, std::size_t n, std::size_t m, std::size_t l = 0)
                : Linear(std::move(name), n, m, l) {}

            const char *Title() const override { return "Tabulated Linear Model"; }
            const char *Description() const override
            { return "Linear system given by its matrices, constant or one per sample."; }

            /**
             * \brief Give one matrix per sample for \p which ("F", "H", "Q" or "R").
             * \throws std::invalid_argument for another name, an empty table, or a matrix of the wrong shape.
             */
            void SetTable(const std::string &which, std::vector<Matrix> table);
            /// The table of \p which, empty when that matrix is constant.
            const std::vector<Matrix> &Table(const std::string &which) const;
            /// Samples covered by the tables (0 when every matrix is constant).
            std::size_t Steps() const;

            /// Name component \p i of the state (\p state true) or of the measurement.
            /// \throws std::out_of_range when \p i is past the last component.
            void SetComponent(bool state, std::size_t i, const std::string &name, const std::string &unit);
            /// Name the components (a model read from a file takes them from the file).
            void SetComponents(std::vector<Component> state, std::vector<Component> measurement);

            Matrix F(double t = 0.) const override;
            Matrix H(double t = 0.) const override;
            Matrix Q(double t = 0.) const override;
            Matrix R(double t = 0.) const override;
    };
}

#endif // KALMAN_MODELS_TABULATED_H
