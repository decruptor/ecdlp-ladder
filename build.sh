#!/bin/zsh
# Build rho80.cu: CPU reference builds (any C++ compiler) and CUDA builds.
#   ./build.sh ref [rung ...]     -> rho_ref<rung>  (no GPU needed)
#   ./build.sh cuda [rung] [arch] -> rho80          (needs nvcc)
set -e
cd "$(dirname "$0")"
CC=${CC:-clang++}

build_ref() {
  r=$1
  echo "build rho_ref$r (CPU_REF)"
  $CC -x c++ -O3 -DCPU_REF -DRUNG=$r -o rho_ref$r rho80.cu
}

build_cuda() {
  r=${1:-80}; arch=${2:-sm_89}
  echo "build rho80 (RUNG=$r, $arch)"
  nvcc -O3 -DRUNG=$r -arch=$arch -o rho80 rho80.cu
}

case "${1:-ref}" in
  ref)
    shift
    rungs=(${@:-48 80})
    for r in $rungs; do build_ref $r; done
    ;;
  cuda)
    shift
    build_cuda "$@"
    ;;
  all)
    for r in 32 48 64 80; do build_ref $r; done
    build_cuda 80 sm_89
    ;;
  *) echo "usage: ./build.sh [ref|cuda|all]"; exit 1 ;;
esac
