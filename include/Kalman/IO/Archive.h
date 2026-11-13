/**
 * *********************************************
 *
 * \file Archive.h
 * \brief Header of the Kalman::IO::Archive class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::IO::Archive
 * \brief Several named byte strings in one file: a zip archive, entries stored.
 *
 * \details What keeps the parts of a model file together. The file is a
 * standard zip archive, so `unzip -l`, Python's `zipfile` and any file manager
 * open it. Every entry carries a CRC-32 that is checked on reading.
 *
 * Entries whose name ends in `.json` are always stored as they are: they are
 * the readable part. The others (tables) are compressed with zlib when the
 * library was built with it (KALMAN_WITH_ZLIB) and it makes them smaller;
 * otherwise they are stored. A build without zlib reads stored entries only,
 * and says so when it meets a compressed one.
 *
 * *********************************************
 */

#ifndef KALMAN_IO_ARCHIVE_H
#define KALMAN_IO_ARCHIVE_H

#include <string>
#include <utility>
#include <vector>

namespace Kalman::IO
{
    class Archive
    {
        public:
            using Entry = std::pair<std::string, std::string>;   // name, bytes

        private:
            std::vector<Entry> fEntries;

        public:
            /// Add an entry (replacing one of the same name).
            void Set(const std::string &name, std::string bytes);
            bool Has(const std::string &name) const;
            /// \throws std::out_of_range naming the missing entry and the present ones.
            const std::string &Get(const std::string &name) const;
            const std::vector<Entry> &Entries() const { return fEntries; }

            /// The archive as bytes.
            std::string Bytes() const;
            /// \throws std::runtime_error when \p bytes is not a zip archive of stored entries, or a CRC fails.
            static Archive FromBytes(const std::string &bytes);

            /// \throws std::runtime_error when the file cannot be written.
            void Write(const std::string &path) const;
            /// \throws std::runtime_error when the file cannot be read, or as FromBytes().
            static Archive Read(const std::string &path);

            /// True when this build compresses and reads compressed entries (zlib).
            static bool Compresses();

            /// True when \p bytes starts like a zip archive.
            static bool Looks(const std::string &bytes);
    };
}

#endif // KALMAN_IO_ARCHIVE_H
