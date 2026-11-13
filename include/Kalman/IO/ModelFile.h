/**
 * *********************************************
 *
 * \file ModelFile.h
 * \brief Header of the Kalman::IO::ModelFile class
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::IO::ModelFile
 * \brief A model written to one file, and read back.
 *
 * \details A `.kalman` file is one archive (Kalman::IO::Archive, a zip) that
 * holds everything a model file is made of, so its parts cannot be separated
 * or mixed with another model's:
 *
 * - `model.json`: the readable part. What the model is, its settings, and a
 *   `spec` in the form the JSON API takes (`kalman.api`, `kalman-serve`);
 * - `tables/<name>.f64`: what does not belong in text, raw little-endian
 *   float64, described (shape, samples) in `model.json`.
 *
 * The same code writes any model, through the Kalman::Model interface. What
 * the file can hold depends on what the model is made of (Model::Structure()):
 *
 * | Structure | File (`kind`) | Reads back |
 * |---|---|---|
 * | Matrices | `complete`: F, H, B, L, Q, R, x0, P0, fs (or A and Qc when stated in continuous time) | anywhere |
 * | TimeVarying | `complete` over a stated number of samples: the matrices that change, as tables | anywhere |
 * | Maps | `reference`: the registered name, its plugin, parameters, noise, initial state | where the plugin is installed |
 * | any, on request | `linearization`: the Jacobians at one state, as a Matrices model | anywhere, valid near that state |
 *
 * `model.json` alone (no tables) may also be saved as plain JSON, with the
 * extension `.kalman.json`. A file with tables is refused that way: its JSON
 * would not be the whole model.
 *
 * *********************************************
 */

#ifndef KALMAN_IO_MODELFILE_H
#define KALMAN_IO_MODELFILE_H

#include "Kalman/IO/Archive.h"
#include "Kalman/IO/Json.h"
#include "Kalman/Model.h"

#include <memory>
#include <string>

namespace Kalman::IO
{
    class ModelFile
    {
        public:
            static constexpr const char *kFormat = "kalman-model";
            static constexpr int kVersion = 1;
            static constexpr const char *kExtension = ".kalman";
            static constexpr const char *kJsonExtension = ".kalman.json";
            static constexpr const char *kDocument = "model.json";

            struct Options
            {
                std::string registered;   ///< the factory name: what a reference refers to
                std::string plugin;       ///< the library that registered it, by name
                bool linearize = false;   ///< write the Jacobians at one state as a Matrices model
                Vector at;                ///< that state (the model's initial state when empty)
                std::size_t steps = 0;    ///< samples to tabulate for a TimeVarying model
                Json origin;              ///< where the model came from, kept as given when writing a file read earlier
            };

        private:
            Json fDocument;
            Archive fTables;    // "tables/<name>.f64" entries only

        public:
            ModelFile() = default;

            /**
             * \brief Write \p model out.
             * \throws std::runtime_error when the model is Maps and neither a registered
             *         name nor a linearization is given, or TimeVarying with no steps.
             */
            static ModelFile Export(const Model &model, const Options &options);
            /// The same with the default options.
            static ModelFile Export(const Model &model);

            /**
             * \brief Build the model the file describes.
             * \throws std::runtime_error for another format or a later version, a
             *         malformed document, or a reference to a model that is not
             *         registered here (the message names the plugin to load).
             */
            std::unique_ptr<Model> Import() const;

            /// `model.json`.
            const Json &Document() const { return fDocument; }
            /// "complete", "reference" or "linearization".
            std::string Kind() const;
            /// The `spec` member: what the JSON API takes.
            const Json &Spec() const { return fDocument["spec"]; }
            bool HasTables() const { return !fTables.Entries().empty(); }

            /// The text of `model.json`.
            std::string Text() const { return fDocument.Dump(2) + "\n"; }
            /// The archive as bytes.
            std::string Bytes() const;

            /**
             * \brief Write the file: an archive, or plain JSON when \p path ends in `.json`.
             * \throws std::runtime_error for plain JSON when the file has tables.
             */
            void Save(const std::string &path) const;

            /// Read an archive or a plain JSON document (told apart by content, not by name).
            static ModelFile Load(const std::string &path);
            /// The same from bytes already read.
            static ModelFile FromBytes(const std::string &bytes);
    };
}

#endif // KALMAN_IO_MODELFILE_H
