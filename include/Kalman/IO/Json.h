/**
 * *********************************************
 *
 * \file Json.h
 * \brief Header of the Kalman::IO::Json value
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::IO::Json
 * \brief A small JSON value: enough to write and read a model file.
 *
 * \details The core has no dependency, so it carries its own reader and
 * writer. It is not a general JSON library: it holds null, booleans, numbers
 * (as double), strings, arrays and objects (which keep the order their keys
 * were set in), reads standard JSON and writes numbers with 17 significant
 * digits, so a double goes to a file and back unchanged.
 *
 * libKalmanNet uses nlohmann_json for the servers; this is for the files the
 * core itself reads and writes (Kalman::IO::ModelFile).
 *
 * *********************************************
 */

#ifndef KALMAN_IO_JSON_H
#define KALMAN_IO_JSON_H

#include "Kalman/Matrix.h"

#include <string>
#include <utility>
#include <vector>

namespace Kalman::IO
{
    class Json
    {
        public:
            enum class Type { Null, Boolean, Number, String, Array, Object };

        private:
            Type fType = Type::Null;
            bool fBoolean = false;
            double fNumber = 0.;
            bool fInteger = false;    // written without a decimal point (a count, a version)
            std::string fString;
            std::vector<Json> fItems;                           // array
            std::vector<std::pair<std::string, Json>> fMembers;  // object, in insertion order

        public:
            Json() = default;
            Json(bool value) : fType(Type::Boolean), fBoolean(value) {}
            Json(double value) : fType(Type::Number), fNumber(value) {}
            Json(int value) : fType(Type::Number), fNumber(value), fInteger(true) {}
            Json(std::size_t value) : fType(Type::Number), fNumber(static_cast<double>(value)), fInteger(true) {}
            Json(const char *value) : fType(Type::String), fString(value) {}
            Json(std::string value) : fType(Type::String), fString(std::move(value)) {}

            static Json Array() { Json j; j.fType = Type::Array; return j; }
            static Json Object() { Json j; j.fType = Type::Object; return j; }
            /// A vector as an array of numbers.
            static Json From(const Vector &v);
            /// A matrix as an array of rows.
            static Json From(const Matrix &M);

            Type GetType() const { return fType; }
            bool IsNull() const { return fType == Type::Null; }
            bool IsObject() const { return fType == Type::Object; }
            bool IsArray() const { return fType == Type::Array; }
            bool IsString() const { return fType == Type::String; }
            bool IsNumber() const { return fType == Type::Number; }

            // --- reading: each throws std::runtime_error naming \p what on the wrong type
            double AsNumber(const std::string &what = "value") const;
            bool AsBoolean(const std::string &what = "value") const;
            const std::string &AsString(const std::string &what = "value") const;
            const std::vector<Json> &Items(const std::string &what = "value") const;
            const std::vector<std::pair<std::string, Json>> &Members(const std::string &what = "value") const;
            /// An array of numbers as a vector.
            Vector AsVector(const std::string &what = "value") const;
            /// An array of equal-length rows as a matrix.
            Matrix AsMatrix(const std::string &what = "value") const;

            /// True when this is an object with a non-null member \p key.
            bool Has(const std::string &key) const;
            /// Member \p key of an object, or a null value when absent.
            const Json &operator[](const std::string &key) const;

            // --- building
            /// Set member \p key of an object (replacing it when present). Returns *this.
            Json &Set(const std::string &key, Json value);
            /// Append to an array. Returns *this.
            Json &Add(Json value);

            /// The text. \p indent > 0 gives one member per line; short arrays of numbers stay on one line.
            std::string Dump(int indent = 2) const;

            /// \throws std::runtime_error with the position of the first error.
            static Json Parse(const std::string &text);

        private:
            void Write(std::string &out, int indent, int depth) const;
    };
}

#endif // KALMAN_IO_JSON_H
