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

#include "Kalman/Config.h"

#if KALMAN_WITH_ZLIB
#include <zlib.h>
#endif

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

#if KALMAN_WITH_ZLIB
        // Raw deflate, as a zip entry holds it. "" when it would not be smaller.
        std::string Deflate(const std::string &bytes)
        {
            z_stream stream{};
            if (deflateInit2(&stream, Z_BEST_COMPRESSION, Z_DEFLATED, -15, 8, Z_DEFAULT_STRATEGY) != Z_OK) return "";
            std::string out(deflateBound(&stream, static_cast<uLong>(bytes.size())), '\0');
            stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(bytes.data()));
            stream.avail_in = static_cast<uInt>(bytes.size());
            stream.next_out = reinterpret_cast<Bytef *>(out.data());
            stream.avail_out = static_cast<uInt>(out.size());
            const int status = deflate(&stream, Z_FINISH);
            const std::size_t size = stream.total_out;
            deflateEnd(&stream);
            if (status != Z_STREAM_END || size >= bytes.size()) return "";
            out.resize(size);
            return out;
        }

        std::string Inflate(const std::string &packed, std::size_t size, const std::string &name)
        {
            std::string out(size, '\0');
            z_stream stream{};
            if (inflateInit2(&stream, -15) != Z_OK)
                throw std::runtime_error("Kalman::IO::Archive: zlib could not start on '" + name + "'.");
            stream.next_in = reinterpret_cast<Bytef *>(const_cast<char *>(packed.data()));
            stream.avail_in = static_cast<uInt>(packed.size());
            stream.next_out = reinterpret_cast<Bytef *>(out.data());
            stream.avail_out = static_cast<uInt>(out.size());
            const int status = inflate(&stream, Z_FINISH);
            const std::size_t got = stream.total_out;
            inflateEnd(&stream);
            if (status != Z_STREAM_END || got != size)
                throw std::runtime_error("Kalman::IO::Archive: entry '" + name + "' does not decompress; the file is damaged.");
            return out;
        }
#endif

        bool Readable(const std::string &name)
        { return name.size() >= 5 && name.compare(name.size() - 5, 5, ".json") == 0; }

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

            // The readable part stays as it is; the rest is compressed when that helps.
            std::string deflated;
#if KALMAN_WITH_ZLIB
            if (!Readable(e.first)) deflated = Deflate(e.second);
#endif
            const std::uint32_t method = deflated.empty() ? 0U : 8U;
            const std::string &data = deflated.empty() ? e.second : deflated;
            const std::uint32_t packed = static_cast<std::uint32_t>(data.size());

            Put32(out, kLocal);
            Put16(out, 20); Put16(out, 0); Put16(out, method);  // version needed, flags, 0 stored / 8 deflated
            Put16(out, 0); Put16(out, kDosDate);                // time, date
            Put32(out, crc); Put32(out, packed); Put32(out, size);
            Put16(out, static_cast<std::uint32_t>(e.first.size())); Put16(out, 0);
            out += e.first;
            out += data;

            Put32(central, kCentral);
            Put16(central, 20); Put16(central, 20); Put16(central, 0); Put16(central, method);
            Put16(central, 0); Put16(central, kDosDate);
            Put32(central, crc); Put32(central, packed); Put32(central, size);
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

    bool Archive::Compresses() { return KALMAN_WITH_ZLIB != 0; }

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
            if (method != 0 && (method != 8 || !Compresses()))
                throw std::runtime_error("Kalman::IO::Archive: entry '" + name + "' is compressed"
                                         + (method == 8 ? ", and this build of libKalman has no zlib (KALMAN_USE_ZLIB). "
                                                          "Repack the archive without compression (zip -0), or use a "
                                                          "build with zlib."
                                                        : " with a method other than deflate."));
            if (Get32(bytes, local) != kLocal) throw std::runtime_error("Kalman::IO::Archive: damaged entry '" + name + "'.");
            const std::size_t data = local + 30 + Get16(bytes, local + 26) + Get16(bytes, local + 28);
            if ((method == 0 && packed != size) || data + packed > bytes.size())
                throw std::runtime_error("Kalman::IO::Archive: entry '" + name + "' is cut short.");
            std::string content = bytes.substr(data, packed);
#if KALMAN_WITH_ZLIB
            if (method == 8) content = Inflate(content, size, name);
#endif
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
