#!/usr/bin/env bash
# Builds the native pieces into the active Python environment:
#   DBoW2 + dpretrieval  loop retrieval (retrieval.method: dbow, the default)
#   ALIKED get_patches   the covisibility gate (backend.long_context.min_covis)
set -euo pipefail
cd "$(dirname "$0")"
TP=$PWD
JOBS=${JOBS:-$(nproc)}

# CMake 4 dropped compatibility with DBoW2's cmake_minimum_required.
cmake -S DBoW2 -B DBoW2/build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="$TP/DBoW2/install" -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build DBoW2/build -j "$JOBS"
cmake --build DBoW2/build --target install

# rpath to the DBoW2 install, so no LD_LIBRARY_PATH is needed at run time.
CMAKE_ARGS="-DDBoW2_DIR=$TP/DBoW2/install/lib/cmake/DBoW2 -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DCMAKE_CXX_STANDARD=14 \
-DCMAKE_BUILD_RPATH=$TP/DBoW2/install/lib -DCMAKE_INSTALL_RPATH=$TP/DBoW2/install/lib" \
    pip install --no-build-isolation ./DPRetrieval

(cd ALIKED/custom_ops && python setup.py build_ext --inplace)
