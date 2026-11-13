/**
 * *********************************************
 *
 * \file Archive.cc
 * \brief Implementation of the Kalman::IO::Archive class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/IO/Archive.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace Kalman::IO
{
    namespace
    {
        std::uint32_t Crc32(const std::string &bytes)
        {
            static const std::array<std::uint32_t, 256> table = [] {
                std::array<std::uint32_t, 256> t{};
                for (std::uint32_t i = 0; i < 256; ++i)
                {
                    std::uint32_t c = i;
                    for (int k = 0; k < 8; ++k) c = (c & 1U) ? (0xEDB88320U ^ (c >> 1)) : (c >> 1);
                    t[i] = c;
                }
                return t;
            }();
            std::uint32_t c = 0xFFFFFFFFU;
            for (const unsigned char b : bytes) c = table[(c ^ b) & 0xFFU] ^ (c >> 8);
            return c ^ 0xFFFFFFFFU;
        }

        // Little-endian fields, whatever the host.
        void Put16(std::string &out, std::uint32_t v)
        {
            out += static_cast<char>(v & 0xFFU);
            out += static_cast<char>((v >> 8) & 0xFFU);
        }

        void Put32(std::string &out, std::uint32_t v)
        {
            Put16(out, v & 0xFFFFU);
            Put16(out, (v >> 16) & 0xFFFFU);
        }

        std::uint32_t Get16(const std::string &in, std::size_t at)
        {
            if (at + 2 > in.size()) throw std::runtime_error("Kalman::IO::Archive: the archive is cut short.");
            return static_cast<std::uint32_t>(static_cast<unsigned char>(in[at]))
                 | (static_cast<std::uint32_t>(static_cast<unsigned char>(in[at + 1])) << 8);
        }

        std::uint32_t Get32(const std::string &in, std::size_t at) { return Get16(in, at) | (Get16(in, at + 2) << 16); }

        const std::uint32_t kLocal = 0x04034b50U, kCentral = 0x02014b50U, kEnd = 0x06054b50U;
        const std::uint32_t kDosDate = (1U << 5) | 1U;   // 1980-01-01: the same model gives the same bytes
    }

    void Archive::Set(const std::string &name, std::string bytes)
    {
        if (name.empty()) throw std::invalid_argument("Kalman::IO::Archive: an entry needs a name.");
        for (auto &e : fEntries)
            if (e.first == name) { e.second = std::move(bytes); return; }
        fEntries.emplace_back(name, std::move(bytes));
    }

    bool Archive::Has(const std::string &name) const
    {
        for (const auto &e : fEntries)
            if (e.first == name) return true;
        return false;
    }

    const std::string &Archive::Get(const std::string &name) const
    {
        for (const auto &e : fEntries)
            if (e.first == name) return e.second;
        std::string known;
        for (const auto &e : fEntries) known += (known.empty() ? "" : ", ") + e.first;
        throw std::out_of_range("Kalman::IO::Archive: no entry '" + name + "' (the archive holds: " + known + ").");
    }

    std::string Archive::Bytes() const
    {
        if (fEntries.size() > 0xFFFEU) throw std::runtime_error("Kalman::IO::Archive: too many entries.");
        std::string out, central;
        for (const auto &e : fEntries)
        {
            if (e.second.size() > 0xFFFFFFFEULL || out.size() > 0xFFFFFFFEULL)
                throw std::runtime_error("Kalman::IO::Archive: '" + e.first + "' passes the 4 GB of a plain zip archive.");
            const std::uint32_t crc = Crc32(e.second), size = static_cast<std::uint32_t>(e.second.size());
            const std::uint32_t offset = static_cast<std::uint32_t>(out.size());

            Put32(out, kLocal);
            Put16(out, 20); Put16(out, 0); Put16(out, 0);      // version needed, flags, method 0 = stored
            Put16(out, 0); Put16(out, kDosDate);                // time, date
            Put32(out, crc); Put32(out, size); Put32(out, size);
            Put16(out, static_cast<std::uint32_t>(e.first.size())); Put16(out, 0);
            out += e.first;
            out += e.second;

            Put32(central, kCentral);
            Put16(central, 20); Put16(central, 20); Put16(central, 0); Put16(central, 0);
            Put16(central, 0); Put16(central, kDosDate);
            Put32(central, crc); Put32(central, size); Put32(central, size);
            Put16(central, static_cast<std::uint32_t>(e.first.size()));
            Put16(central, 0); Put16(central, 0); Put16(central, 0); Put16(central, 0);
            Put32(central, 0); Put32(central, offset);
            central += e.first;
        }
        const std::uint32_t start = static_cast<std::uint32_t>(out.size());
        out += central;
        Put32(out, kEnd);
        Put16(out, 0); Put16(out, 0);
        Put16(out, static_cast<std::uint32_t>(fEntries.size())); Put16(out, static_cast<std::uint32_t>(fEntries.size()));
        Put32(out, static_cast<std::uint32_t>(central.size())); Put32(out, start);
        Put16(out, 0);
        return out;
    }

    bool Archive::Looks(const std::string &bytes)
    { return bytes.size() >= 4 && bytes[0] == 'P' && bytes[1] == 'K' && (bytes[2] == 3 || bytes[2] == 5); }

    Archive Archive::FromBytes(const std::string &bytes)
    {
        if (!Looks(bytes)) throw std::runtime_error("Kalman::IO::Archive: not a zip archive.");
        // The end record is the last thing in the file, before an optional comment.
        std::size_t end = std::string::npos;
        for (std::size_t back = 22; back <= bytes.size() && back <= 22 + 0xFFFF; ++back)
            if (Get32(bytes, bytes.size() - back) == kEnd) { end = bytes.size() - back; break; }
        if (end == std::string::npos) throw std::runtime_error("Kalman::IO::Archive: no end record; the archive is cut short.");

        const std::uint32_t count = Get16(bytes, end + 10);
        std::size_t at = Get32(bytes, end + 16);
        Archive archive;
        for (std::uint32_t i = 0; i < count; ++i)
        {
            if (Get32(bytes, at) != kCentral) throw std::runtime_error("Kalman::IO::Archive: damaged directory.");
            const std::uint32_t method = Get16(bytes, at + 10), crc = Get32(bytes, at + 16);
            const std::uint32_t packed = Get32(bytes, at + 20), size = Get32(bytes, at + 24);
            const std::uint32_t nameLength = Get16(bytes, at + 28), extra = Get16(bytes, at + 30),
                                comment = Get16(bytes, at + 32);
            const std::size_t local = Get32(bytes, at + 42);
            if (at + 46 + nameLength > bytes.size()) throw std::runtime_error("Kalman::IO::Archive: damaged directory.");
            const std::string name = bytes.substr(at + 46, nameLength);
            at += 46 + nameLength + extra + comment;

            if (!name.empty() && name.back() == '/') continue;   // a folder
            if (method != 0)
                throw std::runtime_error("Kalman::IO::Archive: entry '" + name + "' is compressed; this reader takes "
                                         "stored entries only. Repack the archive without compression (zip -0).");
            if (Get32(bytes, local) != kLocal) throw std::runtime_error("Kalman::IO::Archive: damaged entry '" + name + "'.");
            const std::size_t data = local + 30 + Get16(bytes, local + 26) + Get16(bytes, local + 28);
            if (packed != size || data + size > bytes.size())
                throw std::runtime_error("Kalman::IO::Archive: entry '" + name + "' is cut short.");
            std::string content = bytes.substr(data, size);
            if (Crc32(content) != crc)
                throw std::runtime_error("Kalman::IO::Archive: entry '" + name + "' fails its checksum; the file is damaged.");
            archive.Set(name, std::move(content));
        }
        return archive;
    }

    void Archive::Write(const std::string &path) const
    {
        const std::string bytes = Bytes();
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        if (!out) throw std::runtime_error("Kalman::IO::Archive: cannot write '" + path + "'.");
    }

    Archive Archive::Read(const std::string &path)
    {
        std::ifstream in(path, std::ios::binary);
        if (!in) throw std::runtime_error("Kalman::IO::Archive: cannot read '" + path + "'.");
        std::ostringstream buffer;
        buffer << in.rdbuf();
        return FromBytes(buffer.str());
    }
}
