#!/usr/bin/env bash
# TUM RGB-D, the nine freiburg1 sequences (RGB, depth, ground truth; ~5 GB).
#
#   bash scripts/download_tum.sh [DEST] [SCENE ...]      # SCENE: 360 desk desk2 ...
#   python run.py --dataset tum --data_path DEST [--sensor rgbd]
#
# The loader runs every sequence under DEST, so keep only the evaluated ones there.
source "$(dirname "$0")/common.sh"
DEST="${1:-$ROOT/data/tum}"; shift || true
SCENES=(360 desk desk2 floor plant room rpy teddy xyz)
[ $# -gt 0 ] && SCENES=("$@")
BASE=https://cvg.cit.tum.de/rgbd/dataset/freiburg1

for s in "${SCENES[@]}"; do
    name=rgbd_dataset_freiburg1_$s
    if [ -f "$DEST/$name/groundtruth.txt" ]; then say "$name present"; continue; fi
    say "$name"
    fetch "$BASE/$name.tgz" "$DEST/$name.tgz"
    unpack "$DEST/$name.tgz" "$DEST"
done
say "TUM ready under $DEST"
