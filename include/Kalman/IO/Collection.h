/**
 * *********************************************
 *
 * \file Collection.h
 * \brief Header of the Kalman::IO::Collection class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::IO::Collection
 * \brief Several records handled as one — the output of a multi-target pass.
 *
 * \details The counterpart of `KalmanRecordCollection`. One filtered trajectory
 * is a Kalman::IO::Record; a track finder, a multi-target tracker or a bank of
 * per-regime filters produces *many*, and they travel together.
 *
 * It is a thin sequence rather than a container with opinions: iteration,
 * indexing and a summary line. What it adds over a bare `std::vector<Record>` is
 * a name and a place to hang collection-level reporting, so callers do not each
 * invent their own.
 *
 * *********************************************
 */

#ifndef KALMAN_IO_COLLECTION_H
#define KALMAN_IO_COLLECTION_H

#include "Kalman/IO/Record.h"

#include <iosfwd>
#include <string>
#include <vector>

namespace Kalman::IO
{
    class Collection
    {
        private:
            std::string fName = "records";
            std::vector<Record> fRecords;

        public:
            Collection() = default;
            explicit Collection(std::string name) : fName(std::move(name)) {}

            const std::string &Name() const { return fName; }
            void SetName(std::string name) { fName = std::move(name); }

            void Add(const Record &record) { fRecords.push_back(record); }
            void Clear() { fRecords.clear(); }
            void Reserve(std::size_t n) { fRecords.reserve(n); }

            std::size_t Size() const { return fRecords.size(); }
            bool Empty() const { return fRecords.empty(); }

            Record &operator[](std::size_t i) { return fRecords.at(i); }
            const Record &operator[](std::size_t i) const { return fRecords.at(i); }

            std::vector<Record>::iterator begin() { return fRecords.begin(); }
            std::vector<Record>::iterator end() { return fRecords.end(); }
            std::vector<Record>::const_iterator begin() const { return fRecords.begin(); }
            std::vector<Record>::const_iterator end() const { return fRecords.end(); }

            /// Total number of steps across every record.
            std::size_t TotalSteps() const;

            void Print(std::ostream &os) const;
    };

    std::ostream &operator<<(std::ostream &os, const Collection &collection);
}

namespace Kalman
{
    /// `Kalman::RecordCollection` is the short spelling of Kalman::IO::Collection.
    using RecordCollection = IO::Collection;
}

#endif // KALMAN_IO_COLLECTION_H
