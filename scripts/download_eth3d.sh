#!/usr/bin/env bash
# ETH3D SLAM, the 55 training sequences: RGB (`_mono`) plus the registered depth
# (`_rgbd`, which adds only depth/ and associated.txt). 26 GB, or 19 GB with --no-depth.
#
#   bash scripts/download_eth3d.sh [DEST] [--no-depth] [SEQUENCE ...]
#   python run.py --dataset eth3d --data_path DEST [--sensor rgbd]
source "$(dirname "$0")/common.sh"
DEST="${1:-$ROOT/data/eth3d}"; shift || true
KINDS=(mono rgbd)
if [ "${1:-}" = --no-depth ]; then KINDS=(mono); shift; fi
SEQS=("$@")
if [ ${#SEQS[@]} -eq 0 ]; then
    SEQS=($("$PYTHON" -c "import sys; sys.path.insert(0, '$ROOT')
from amb3r_slam.datasets.eth3d import ETH3D_SCENES; print(*ETH3D_SCENES)"))
fi
BASE=https://www.eth3d.net/data/slam/datasets

for s in "${SEQS[@]}"; do
    for k in "${KINDS[@]}"; do
        probe=$([ "$k" = mono ] && echo rgb.txt || echo associated.txt)
        if [ -f "$DEST/$s/$probe" ]; then continue; fi
        say "$s ($k)"
        fetch "$BASE/${s}_$k.zip" "$DEST/${s}_$k.zip"
        unpack "$DEST/${s}_$k.zip" "$DEST"
    done
done
say "ETH3D ready under $DEST"
