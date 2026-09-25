#!/usr/bin/env bash
# Build the FFT solver driver used by src/piml/mesh_convergence.py and corpus.py.
#
# It compiles against the header-only Moulinec-Suquet solver inside the MatViz3D
# clone, so that clone must exist first. See docs/INSTALL.md.
#
# Usage: tools/build_mesh_refine.sh [output_path]
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
MATVIZ3D="${PIML_MATVIZ3D:-$ROOT/MatViz3D}"
OUT="${1:-${PIML_MESH_REFINE:-$ROOT/build/mesh_refine}}"

[ -f "$MATVIZ3D/fft_homog.hpp" ] || {
    echo "MatViz3D not found at $MATVIZ3D; see docs/INSTALL.md" >&2; exit 1; }

mkdir -p "$(dirname "$OUT")"
g++ -O3 -std=c++17 -march=native -fopenmp -I"$MATVIZ3D" "$HERE/mesh_refine.cpp" -o "$OUT"
echo "built $OUT"
