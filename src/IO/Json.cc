/**
 * *********************************************
 *
 * \file Json.cc
 * \brief Implementation of the Kalman::IO::Json value
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 */

#include "Kalman/IO/Json.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace Kalman::IO
{
    namespace
    {
        [[noreturn]] void Wrong(const std::string &what, const char *expected)
        { throw std::runtime_error("Kalman::IO::Json: " + what + " is not " + expected + "."); }

        const Json kNull;

        void Escape(std::string &out, const std::string &s)
        {
            out += '"';
            for (const unsigned char c : s)
            {
                switch (c)
                {
                    case '"':  out += "\\\""; break;
                    case '\\': out += "\\\\"; break;
                    case '\n': out += "\\n"; break;
                    case '\r': out += "\\r"; break;
                    case '\t': out += "\\t"; break;
                    default:
                        if (c < 0x20)
                        {
                            char buffer[8];
                            std::snprintf(buffer, sizeof buffer, "\\u%04x", c);
                            out += buffer;
                        }
                        else out += static_cast<char>(c);
                }
            }
            out += '"';
        }

        void Number(std::string &out, double v, bool integer)
        {
            if (!std::isfinite(v)) { out += "null"; return; }   // JSON has no NaN or infinity
            if (integer && std::abs(v) < 1e15)
            {
                char buffer[32];
                std::snprintf(buffer, sizeof buffer, "%.0f", v);
                out += buffer;
                return;
            }
            if (v == std::floor(v) && std::abs(v) < 1e15)
            {
                char buffer[32];
                std::snprintf(buffer, sizeof buffer, "%.1f", v);
                out += buffer;
                return;
            }
            // The shortest text that reads back as the same double.
            char buffer[40];
            for (int digits = 15; digits <= 17; ++digits)
            {
                std::snprintf(buffer, sizeof buffer, "%.*g", digits, v);
                if (std::strtod(buffer, nullptr) == v) break;
            }
            out += buffer;
        }

        struct Reader
        {
            const std::string &text;
            std::size_t at = 0;

            [[noreturn]] void Fail(const std::string &message) const
            {
                std::size_t line = 1, column = 1;
                for (std::size_t i = 0; i < at && i < text.size(); ++i)
                {
                    if (text[i] == '\n') { ++line; column = 1; }
                    else ++column;
                }
                throw std::runtime_error("Kalman::IO::Json: " + message + " at line " + std::to_string(line)
                                         + ", column " + std::to_string(column) + ".");
            }

            void Skip()
            {
                while (at < text.size() && (text[at] == ' ' || text[at] == '\n' || text[at] == '\t' || text[at] == '\r'))
                    ++at;
            }

            bool Take(const char *word)
            {
                const std::string w(word);
                if (text.compare(at, w.size(), w) != 0) return false;
                at += w.size();
                return true;
            }

            std::string String()
            {
                ++at;   // the opening quote
                std::string out;
                while (true)
                {
                    if (at >= text.size()) Fail("unterminated string");
                    const char c = text[at++];
                    if (c == '"') return out;
                    if (c != '\\') { out += c; continue; }
                    if (at >= text.size()) Fail("unterminated escape");
                    const char e = text[at++];
                    switch (e)
                    {
                        case '"': out += '"'; break;
                        case '\\': out += '\\'; break;
                        case '/': out += '/'; break;
                        case 'b': out += '\b'; break;
                        case 'f': out += '\f'; break;
                        case 'n': out += '\n'; break;
                        case 'r': out += '\r'; break;
                        case 't': out += '\t'; break;
                        case 'u':
                        {
                            if (at + 4 > text.size()) Fail("short \\u escape");
                            const unsigned long code = std::strtoul(text.substr(at, 4).c_str(), nullptr, 16);
                            at += 4;
                            // UTF-8, basic plane (a surrogate pair is kept as two code units).
                            if (code < 0x80) out += static_cast<char>(code);
                            else if (code < 0x800)
                            {
                                out += static_cast<char>(0xC0 | (code >> 6));
                                out += static_cast<char>(0x80 | (code & 0x3F));
                            }
                            else
                            {
                                out += static_cast<char>(0xE0 | (code >> 12));
                                out += static_cast<char>(0x80 | ((code >> 6) & 0x3F));
                                out += static_cast<char>(0x80 | (code & 0x3F));
                            }
                            break;
                        }
                        default: Fail("unknown escape");
                    }
                }
            }

            Json Value(int depth)
            {
                if (depth > 200) Fail("nested too deeply");
                Skip();
                if (at >= text.size()) Fail("unexpected end of text");
                const char c = text[at];
                if (c == '{')
                {
                    ++at;
                    Json object = Json::Object();
                    Skip();
                    if (at < text.size() && text[at] == '}') { ++at; return object; }
                    while (true)
                    {
                        Skip();
                        if (at >= text.size() || text[at] != '"') Fail("expected a member name");
                        const std::string key = String();
                        Skip();
                        if (at >= text.size() || text[at] != ':') Fail("expected ':'");
                        ++at;
                        object.Set(key, Value(depth + 1));
                        Skip();
                        if (at < text.size() && text[at] == ',') { ++at; continue; }
                        if (at < text.size() && text[at] == '}') { ++at; return object; }
                        Fail("expected ',' or '}'");
                    }
                }
                if (c == '[')
                {
                    ++at;
                    Json array = Json::Array();
                    Skip();
                    if (at < text.size() && text[at] == ']') { ++at; return array; }
                    while (true)
                    {
                        array.Add(Value(depth + 1));
                        Skip();
                        if (at < text.size() && text[at] == ',') { ++at; continue; }
                        if (at < text.size() && text[at] == ']') { ++at; return array; }
                        Fail("expected ',' or ']'");
                    }
                }
                if (c == '"') return Json(String());
                if (Take("true")) return Json(true);
                if (Take("false")) return Json(false);
                if (Take("null")) return Json();
                const char *begin = text.c_str() + at;
                char *end = nullptr;
                const double v = std::strtod(begin, &end);
                if (end == begin) Fail("unexpected character");
                at += static_cast<std::size_t>(end - begin);
                return Json(v);
            }
        };
    }

    Json Json::From(const Vector &v)
    {
        Json out = Array();
        for (std::size_t i = 0; i < v.Size(); ++i) out.Add(v(i));
        return out;
    }

    Json Json::From(const Matrix &M)
    {
        Json out = Array();
        for (std::size_t i = 0; i < M.Rows(); ++i)
        {
            Json row = Array();
            for (std::size_t j = 0; j < M.Cols(); ++j) row.Add(M(i, j));
            out.Add(std::move(row));
        }
        return out;
    }

    double Json::AsNumber(const std::string &what) const
    {
        if (fType != Type::Number) Wrong(what, "a number");
        return fNumber;
    }

    bool Json::AsBoolean(const std::string &what) const
    {
        if (fType != Type::Boolean) Wrong(what, "true or false");
        return fBoolean;
    }

    const std::string &Json::AsString(const std::string &what) const
    {
        if (fType != Type::String) Wrong(what, "a string");
        return fString;
    }

    const std::vector<Json> &Json::Items(const std::string &what) const
    {
        if (fType != Type::Array) Wrong(what, "an array");
        return fItems;
    }

    const std::vector<std::pair<std::string, Json>> &Json::Members(const std::string &what) const
    {
        if (fType != Type::Object) Wrong(what, "an object");
        return fMembers;
    }

    Vector Json::AsVector(const std::string &what) const
    {
        const auto &items = Items(what);
        Vector v(items.size());
        for (std::size_t i = 0; i < items.size(); ++i) v(i) = items[i].AsNumber(what + "[" + std::to_string(i) + "]");
        return v;
    }

    Matrix Json::AsMatrix(const std::string &what) const
    {
        const auto &rows = Items(what);
        if (rows.empty()) return Matrix();
        const std::size_t cols = rows[0].Items(what + "[0]").size();
        Matrix M(rows.size(), cols, 0.);
        for (std::size_t i = 0; i < rows.size(); ++i)
        {
            const auto &row = rows[i].Items(what + "[" + std::to_string(i) + "]");
            if (row.size() != cols)
                throw std::runtime_error("Kalman::IO::Json: " + what + " has rows of different lengths.");
            for (std::size_t j = 0; j < cols; ++j) M(i, j) = row[j].AsNumber(what);
        }
        return M;
    }

    bool Json::Has(const std::string &key) const
    {
        if (fType != Type::Object) return false;
        for (const auto &m : fMembers)
            if (m.first == key) return !m.second.IsNull();
        return false;
    }

    const Json &Json::operator[](const std::string &key) const
    {
        if (fType == Type::Object)
            for (const auto &m : fMembers)
                if (m.first == key) return m.second;
        return kNull;
    }

    Json &Json::Set(const std::string &key, Json value)
    {
        if (fType == Type::Null) fType = Type::Object;
        if (fType != Type::Object) Wrong("the value", "an object");
        for (auto &m : fMembers)
            if (m.first == key) { m.second = std::move(value); return *this; }
        fMembers.emplace_back(key, std::move(value));
        return *this;
    }

    Json &Json::Add(Json value)
    {
        if (fType == Type::Null) fType = Type::Array;
        if (fType != Type::Array) Wrong("the value", "an array");
        fItems.push_back(std::move(value));
        return *this;
    }

    void Json::Write(std::string &out, int indent, int depth) const
    {
        const auto newline = [&out, indent](int d)
        {
            if (indent <= 0) return;
            out += '\n';
            out.append(static_cast<std::size_t>(indent * d), ' ');
        };
        switch (fType)
        {
            case Type::Null: out += "null"; break;
            case Type::Boolean: out += fBoolean ? "true" : "false"; break;
            case Type::Number: Number(out, fNumber, fInteger); break;
            case Type::String: Escape(out, fString); break;
            case Type::Array:
            {
                // A row of numbers or strings reads best on one line; an array of rows, one row per line.
                bool flat = true;
                for (const auto &item : fItems) flat = flat && !item.IsArray() && !item.IsObject();
                out += '[';
                for (std::size_t i = 0; i < fItems.size(); ++i)
                {
                    if (i) out += flat ? ", " : ",";
                    if (!flat) newline(depth + 1);
                    fItems[i].Write(out, flat ? 0 : indent, depth + 1);
                }
                if (!flat && !fItems.empty()) newline(depth);
                out += ']';
                break;
            }
            case Type::Object:
            {
                bool flat = indent <= 0 || fMembers.size() <= 2;   // {"name": .., "unit": ..} on one line
                for (const auto &m : fMembers) flat = flat && !m.second.IsArray() && !m.second.IsObject();
                out += '{';
                for (std::size_t i = 0; i < fMembers.size(); ++i)
                {
                    if (i) out += flat ? ", " : ",";
                    if (!flat) newline(depth + 1);
                    Escape(out, fMembers[i].first);
                    out += ": ";
                    fMembers[i].second.Write(out, flat ? 0 : indent, depth + 1);
                }
                if (!flat && !fMembers.empty()) newline(depth);
                out += '}';
                break;
            }
        }
    }

    std::string Json::Dump(int indent) const
    {
        std::string out;
        Write(out, indent, 0);
        return out;
    }

    Json Json::Parse(const std::string &text)
    {
        Reader reader{text};
        Json value = reader.Value(0);
        reader.Skip();
        if (reader.at != text.size()) reader.Fail("text after the value");
        return value;
    }
}
