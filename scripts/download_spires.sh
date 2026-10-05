#!/usr/bin/env bash
# Oxford Spires, the eleven evaluated sequences: the raw camera capture (`images.zip`,
# read in place, never extracted), the TLS-registered ground truth and the camera
# calibration (~83 GB). Installing aria2c speeds this up considerably.
#
#   bash scripts/download_spires.sh [DEST] [SEQUENCE ...]
#   python run.py --dataset spires --data_path DEST
source "$(dirname "$0")/common.sh"
DEST="${1:-$ROOT/data/spires}"; shift || true
SEQS=(2024-03-12-keble-college-02 2024-03-12-keble-college-03 2024-03-12-keble-college-04
      2024-03-12-keble-college-05 2024-03-13-observatory-quarter-01
      2024-03-13-observatory-quarter-02 2024-03-14-blenheim-palace-05
      2024-03-18-christ-church-02 2024-03-18-christ-church-03 2024-03-20-christ-church-05
      2024-05-20-bodleian-library-02)
[ $# -gt 0 ] && SEQS=("$@")
BASE=https://huggingface.co/datasets/ori-drs/oxford_spires_dataset/resolve/main

for f in cam0.yaml cam-lidar-imu.yaml; do
    [ -f "$DEST/calibration/$f" ] || fetch "$BASE/calibration/$f" "$DEST/calibration/$f"
done
for s in "${SEQS[@]}"; do
    d="$DEST/sequences/$s"
    [ -f "$d/processed/trajectory/gt-tum.txt" ] || \
        fetch "$BASE/sequences/$s/processed/trajectory/gt-tum.txt" \
              "$d/processed/trajectory/gt-tum.txt"
    if "$PYTHON" -c "import zipfile; zipfile.ZipFile('$d/raw/images.zip')" 2>/dev/null; then
        say "$s present"; continue
    fi
    say "$s"
    fetch "$BASE/sequences/$s/raw/images.zip" "$d/raw/images.zip"
done
say "Oxford Spires ready under $DEST"
