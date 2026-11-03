#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
apt-get update -qq
apt-get install -y --no-install-recommends libdlib-dev libopenblas-dev libboost-filesystem-dev libboost-system-dev libboost-regex-dev libtbb-dev ffmpeg git ca-certificates
cd /root/autodl-tmp
if [ ! -d OpenFace/.git ]; then
  git clone --depth 1 https://github.com/TadasBaltrusaitis/OpenFace.git
fi
cd OpenFace
git rev-parse HEAD > /root/autodl-tmp/mosei-q1/mosei_q1/openface-commit.txt
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target FeatureExtraction -j 12
# Model acquisition is handled explicitly after inspecting upstream download URLs.
printf 'OPENFACE_BUILD_COMPLETE\n'
