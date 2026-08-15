/**
 * *********************************************
 *
 * \file Collection.cc
 * \brief Implementation of the Kalman::IO::Collection class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/IO/Collection.h"

#include <ostream>

namespace Kalman::IO
{
    std::size_t Collection::TotalSteps() const
    {
        std::size_t total = 0;
        for (const Record &record : fRecords) total += record.NSteps();
        return total;
    }

    void Collection::Print(std::ostream &os) const
    {
        os << fName << " — " << fRecords.size() << " record"
           << (fRecords.size() == 1 ? "" : "s") << ", " << TotalSteps() << " steps total";
        for (const Record &record : fRecords)
        {
            os << "\n  ";
            record.Print(os);
        }
    }

    std::ostream &operator<<(std::ostream &os, const Collection &collection)
    {
        collection.Print(os);
        return os;
    }
}
