#!/usr/bin/env bash
# Bonn RGB-D Dynamic, the eight sequences of the dynamic-scene table (~3 GB).
#
#   bash scripts/download_bonn.sh [DEST] [SEQUENCE ...]  # SEQUENCE: balloon crowd2 ...
#   python run.py --dataset bonn --data_path DEST
#
# The loader runs every sequence under DEST, so keep only the evaluated ones there.
source "$(dirname "$0")/common.sh"
DEST="${1:-$ROOT/data/bonn}"; shift || true
SEQS=(balloon balloon2 crowd crowd2 moving_nonobstructing_box moving_nonobstructing_box2
      person_tracking person_tracking2)
[ $# -gt 0 ] && SEQS=("$@")
BASE=https://www.ipb.uni-bonn.de/html/projects/rgbd_dynamic2019

for s in "${SEQS[@]}"; do
    name=rgbd_bonn_$s
    if [ -f "$DEST/$name/groundtruth.txt" ]; then say "$name present"; continue; fi
    say "$name"
    fetch "$BASE/$name.zip" "$DEST/$name.zip"
    unpack "$DEST/$name.zip" "$DEST"
done
say "Bonn ready under $DEST"
