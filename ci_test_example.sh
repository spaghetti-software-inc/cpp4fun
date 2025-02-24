#!/bin/bash

set -e
set -x

BASEDIR=$(dirname "$0")
pushd "$BASEDIR"

rm -rf build
conan install . --output-folder=build --build=missing 

cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release
cmake --build . -j20

cp -r ../data ./

#./tests/tests
./src/pde/pde
./src/complex/complex
./src/starfield/starfield
./src/curand3d/curand3d
./src/cpp4fun/cpp4fun
./src/curand4fun/curand4fun

#./src/cpp4fun/imguiDemo
#./src/particles/particles
#./src/cudaGL/cudaGL


