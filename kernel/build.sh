#!/usr/bin/env bash
# Build the fused int4 GEMV for whatever CPU this machine has.
set -euo pipefail
cd "$(dirname "$0")"
ARCH=$(uname -m)
OUT=libq4gemv.so
FLAGS="-O3 -shared -fPIC -pthread"
if [[ "$ARCH" == "arm64" ]]; then
  FLAGS="$FLAGS -mcpu=native"
else
  FLAGS="$FLAGS -mavx2 -mfma -mf16c"
fi
${CC:-clang} $FLAGS -o "$OUT" q4gemv.c
echo "built $(pwd)/$OUT for $ARCH"
