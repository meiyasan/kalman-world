#! /bin/bash

# Configure the CMake build tree. Invoked by `make configure`, and therefore by
# `make docs` / `make pages`, which configure on demand:
#
#     ./bootstrap.sh [BUILDTYPE]        # Release (default), Debug, Doc, ...
#
# Same shape as the sibling projects using this template, with one difference:
# it stops after configuring. There is no CMake library to compile here — the
# Python package is installed with pip (`make setup`) and the C fast path is
# built by CFFI (`make accel`) — so the Makefile drives `docs` / `pages` /
# `install` explicitly rather than having bootstrap build everything.

[[ $_ != $0 ]] && REALPATH=`dirname $(readlink -f ${BASH_SOURCE[0]})` || REALPATH=`dirname $(readlink -f $0)`

if [[ -f "/.dockerenv" ]]; then
    BUILD_DIR="$REALPATH/var/container/.build"
    INSTALL_DIR="$REALPATH/var/container/.install"
else
    BUILD_DIR="$REALPATH/var/build"
    INSTALL_DIR="$REALPATH/var/install"
fi

# The theme (Doxyfile.in, layout, CSS/JS) and the shared CMake modules are
# submodules; nothing configures without them.
for sub in share/doxygen cmake/Modules; do
    if [ ! "$(ls -A $REALPATH/$sub 2>/dev/null)" ]; then
        echo "==> fetching submodule $sub"
        git -C "$REALPATH" submodule update --init --recursive -- "$sub"
    fi
done

if ! command -v cmake >/dev/null 2>&1; then
    echo "cmake is required to configure the documentation build" >&2
    echo "    apt-get install cmake   |   brew install cmake" >&2
    exit 1
fi

if [[ ! -z "$1" || ! -d "$BUILD_DIR" || ! -f "$BUILD_DIR/Makefile" ]]; then

        mkdir -p $BUILD_DIR
        cd $BUILD_DIR

        BUILD_TYPE=$(echo "$1" | tr '[:upper:]' '[:lower:]')
        [ "$BUILD_TYPE" == "debug" ] && CMAKE_BUILDTYPE_FLAGS="-DCMAKE_BUILD_TYPE=Debug"
        [ "$BUILD_TYPE" == "release" ] && CMAKE_BUILDTYPE_FLAGS="-DCMAKE_BUILD_TYPE=Release"
        [ "$BUILD_TYPE" == "doc" ] && CMAKE_OPTIONAL_FLAGS="-DBUILD_DOCUMENTATION='ON'"

        cmake $REALPATH -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR -DMOTD='"$MOTD"' $CMAKE_OPTIONAL_FLAGS $CMAKE_BUILDTYPE_FLAGS
        [ $? -eq 1 ] && exit 1
fi

cd - > /dev/null 2> /dev/null
