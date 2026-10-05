#!/usr/bin/env bash
# VBR (Vision Benchmark in Rome), the eight training sequences.
#
#   bash scripts/download_vbr.sh [DEST] [--images-only] [SEQUENCE ...]
#   python run.py --dataset vbr --data_path DEST [--sensor lidar]
#
# Images and camera ground truth for seven sequences come from LoGeR's preprocessed
# release (Junyi42/vbr_processed, streamed and filtered: the 120 GB archive is never
# stored; set VBR_TARBALL to use a local copy). The raw recordings -- needed for LiDAR runs,
# for VBR's own metrics and for ciampino_train0's images -- come from the VBR devkit
# (`pip install vbr-devkit`, which provides `vbr`); set VBR=/path/to/vbr if it lives in
# another environment. Each sequence's bags (100 GB or more) are converted and deleted
# before the next is fetched, so peak disk is one sequence plus the prepared data.
source "$(dirname "$0")/common.sh"
DEST="${1:-$ROOT/data/vbr}"; shift || true
RAW_TOO=1
if [ "${1:-}" = --images-only ]; then RAW_TOO=0; shift; fi
SEQS=(campus_train0 campus_train1 ciampino_train0 ciampino_train1 colosseo_train0
      diag_train0 pincio_train0 spagna_train0)
[ $# -gt 0 ] && SEQS=("$@")
LOGER=https://huggingface.co/datasets/Junyi42/vbr_processed/resolve/main/vbr_processed.tar.gz
VBR="${VBR:-vbr}"
mkdir -p "$DEST"

need_images=0
for s in "${SEQS[@]}"; do
    [ "$s" = ciampino_train0 ] || [ -d "$DEST/$s/rgb" ] || need_images=1
done
if [ $need_images = 1 ]; then
    say "LoGeR images, intrinsics and camera ground truth"
    tmp="$DEST/.loger"; mkdir -p "$tmp"
    members=('vbr/*_processed_aligned/rgb/*' 'vbr/*_processed_aligned/intrinsics.txt'
             'vbr/*_processed_aligned/camera_pose.txt')
    if [ -n "${VBR_TARBALL:-}" ]; then
        tar -xzf "$VBR_TARBALL" -C "$tmp" --wildcards "${members[@]}"
    else
        curl -sfL "$LOGER" | tar -xz -C "$tmp" --wildcards "${members[@]}"
    fi
    "$PYTHON" "$ROOT/scripts/prepare_vbr.py" loger "$tmp/vbr" "$DEST"
    rm -rf "$tmp"
fi

[ $RAW_TOO = 1 ] || { say "VBR images ready under $DEST"; exit 0; }
command -v "$VBR" >/dev/null || { echo "'$VBR' not found: pip install vbr-devkit" >&2; exit 1; }
for s in "${SEQS[@]}"; do
    if [ -f "$DEST/$s/pairs.json" ]; then say "$s raw data present"; continue; fi
    raw="$DEST/.raw"; place=${s%%_*}
    say "$s: download"
    "$VBR" download "$s" "$raw"
    say "$s: convert"
    "$VBR" convert kitti "$raw/vbr_slam/$place/$s" "$raw/${s}_kitti"
    rm -f "$raw/vbr_slam/$place/$s"/*.bag
    "$PYTHON" "$ROOT/scripts/prepare_vbr.py" raw "$raw/${s}_kitti" \
        "$raw/vbr_slam/$place/$s" "$s" "$DEST"
    rm -rf "$raw/${s}_kitti" "$raw/vbr_slam/$place/$s"
done
rm -rf "$DEST/.raw"
say "VBR ready under $DEST"
