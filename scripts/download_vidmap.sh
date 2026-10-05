#!/usr/bin/env bash
# LaMAR and CroCoDL, prepared by VidMap's own preparer so that the test sequences and
# the W-AUC protocol are exactly VidMap's.
#
#   bash scripts/download_vidmap.sh lamar   [DEST] [--scenes CAB ...]          # 45 GB, 24 GB kept
#   bash scripts/download_vidmap.sh crocodl [DEST] [--scenes ios-ARCHE_B3 ...] # 15 GB, 14 GB kept
#   python run.py --dataset lamar|crocodl --data_path DEST
#
# The preparer needs pycolmap >= 3.12, numpy, pyyaml, pillow, requests and tqdm (no
# VidMap build); CroCoDL also needs Python >= 3.10. Set PYTHON to use another interpreter.
source "$(dirname "$0")/common.sh"
DATASET="${1:?lamar or crocodl}"
DEST="${2:-$ROOT/data/$DATASET}"
shift; shift || true
VIDMAP_REPO=https://github.com/cvg/vidmap.git
VIDMAP_COMMIT=556312aaaf71ffe0a588b1a1343e6b0dac5cb1a3
SRC="$DEST/.vidmap"

if [ ! -d "$SRC/.git" ]; then
    git clone --quiet --filter=blob:none "$VIDMAP_REPO" "$SRC"
fi
git -C "$SRC" checkout --quiet "$VIDMAP_COMMIT"

KEY=$(echo "$DATASET" | tr a-z A-Z)
export "VIDMAP_${KEY}_DATA_DIR=$DEST/datasets"
export "VIDMAP_${KEY}_CACHE_DIR=$DEST/.cache"
export "VIDMAP_${KEY}_EXP_DIR=$DEST/.experiments"
case "$DATASET" in
    lamar) export "VIDMAP_LAMAR_TESTSETS_DIR=$DEST/testsets" ;;
    # the CroCoDL preparer writes test sets to <parent of this>/crocodl
    crocodl) export "VIDMAP_CROCODL_TESTSETS_DIR=$DEST/.testsets/crocodl" ;;
    *) echo "unknown dataset $DATASET (lamar or crocodl)" >&2; exit 1 ;;
esac

say "preparing $DATASET under $DEST"
(cd "$SRC" && PYTHONPATH="$SRC" "$PYTHON" -m vidmap.datasets.prepare \
     --dataset "$DATASET" --delete-downloads "$@")

if [ "$DATASET" = crocodl ]; then
    mkdir -p "$DEST/testsets"
    for d in "$DEST/.testsets/crocodl"/ios-*; do
        rm -rf "$DEST/testsets/$(basename "$d")"
        mv "$d" "$DEST/testsets/"
    done
    rm -rf "$DEST/.testsets" "$DEST/datasets/capture"
fi
rm -rf "$DEST/.cache" "$DEST/.experiments"
say "$DATASET ready under $DEST"
