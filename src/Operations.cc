/**
 * *********************************************
 *
 * \file Operations.cc
 * \brief The model operations: the functions, and the methods of Kalman::Model.
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/Operations.h"

#include <utility>

namespace Kalman
{
    namespace
    {
        /// A shared pointer that does not own: the model outlives what borrows it.
        std::shared_ptr<Model> Borrow(Model *model) { return std::shared_ptr<Model>(model, [](Model *) {}); }
    }

    std::shared_ptr<Models::Fusion> AddSensor(std::shared_ptr<Model> model, Sensor sensor)
    { return std::make_shared<Models::Fusion>(std::move(model), std::move(sensor)); }

    std::shared_ptr<Models::Joint> Joint(std::shared_ptr<Model> model, std::vector<EstimatedParameter> parameters)
    { return std::make_shared<Models::Joint>(std::move(model), std::move(parameters)); }

    std::unique_ptr<Model> Model::AddSensor(const Sensor &sensor)
    { return std::make_unique<Models::Fusion>(Borrow(this), sensor); }

    std::unique_ptr<Model> Model::Joint(const std::vector<EstimatedParameter> &parameters)
    { return std::make_unique<Models::Joint>(Borrow(this), parameters); }
}
