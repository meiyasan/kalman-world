/**
 * *********************************************
 *
 * \file Metadata.h
 * \brief Self-description shared by every framework role
 * \author Marco Meyer \<marco.meyer@omu.ac.jp\>
 *
 * *********************************************
 *
 * \class Kalman::Abstract::Metadata
 * \brief What a model, engine or filter says about itself.
 *
 * \details The C++ counterpart of the `NAME` / `DESCRIPTION` / `VERSION` class
 * attributes that `modules/framework/abstract/metadata.py` gives every role. They
 * are what a catalogue, a log line or an error message quotes, so they belong to
 * the type rather than to the instance — Kalman::Model additionally carries an
 * instance Name() — whatever the application called that particular system —
 * which is a different thing from its Title() (`"Kalman Model"`).
 *
 * *********************************************
 */

#ifndef KALMAN_ABSTRACT_METADATA_H
#define KALMAN_ABSTRACT_METADATA_H

namespace Kalman::Abstract
{
    class Metadata
    {
        public:
            virtual ~Metadata() = default;

            /// Human-readable name of the *kind* of thing this is.
            virtual const char *Title() const = 0;

            /// One sentence on what it does.
            virtual const char *Description() const = 0;

            /// Implementation version, bumped when behaviour changes.
            virtual const char *Version() const { return "1.0"; }
    };
}

#endif // KALMAN_ABSTRACT_METADATA_H
