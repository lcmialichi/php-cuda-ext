#!/bin/bash
set -e

EXT_NAME="cuda"
BUILD_DIR="./${EXT_NAME}_build"

if [ ! -d "$BUILD_DIR" ]; then
   echo "Compile the extension before running tests"
   exit 1
fi

cd "$BUILD_DIR"
POOL_TEST_BINARY=$(mktemp)
trap 'rm -f "$POOL_TEST_BINARY"' EXIT
${CC:-cc} -D_GNU_SOURCE -pthread $(php-config --includes) \
   -I"${CUDA_HOME:-/usr/local/cuda}/include" -Isrc/cuda \
   tests/memory_pool_test.c -o "$POOL_TEST_BINARY"
"$POOL_TEST_BINARY"
make test TEST_PHP_ARGS="-q"