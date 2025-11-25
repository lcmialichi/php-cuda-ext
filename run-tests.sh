#!/bin/bash
set -e

EXT_NAME="cuda"
BUILD_DIR="./${EXT_NAME}_build"

if [ ! -d "$BUILD_DIR" ]; then
   echo "Compile the extension before running tests"
   exit 1
fi

cd "$BUILD_DIR"
make test