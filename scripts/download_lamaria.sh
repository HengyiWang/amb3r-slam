#!/usr/bin/env bash
# LaMAria, the 21 training sequences with pseudo ground truth: ASL images, pinhole
# calibration and pseudo-dense ground truth (~170 GB).
#
#   bash scripts/download_lamaria.sh [DEST] [SEQUENCE ...]
#   python run.py --dataset lamaria --data_path DEST
source "$(dirname "$0")/common.sh"
DEST="${1:-$ROOT/data/lamaria}"; shift || true
SEQS=(R_01_easy R_02_easy R_03_easy R_04_medium R_05_medium R_06_medium R_07_medium
      R_08_hard R_09_hard R_10_hard R_11_5cp R_12_10cp R_13_15cp
      sequence_1_19 sequence_1_20 sequence_2_11 sequence_2_12 sequence_3_17 sequence_3_18
      sequence_4_10 sequence_4_11)
[ $# -gt 0 ] && SEQS=("$@")
BASE=https://cvg-data.inf.ethz.ch/lamaria

for s in "${SEQS[@]}"; do
    d="$DEST/training/$s"
    if [ -d "$d/asl_folder/$s/aria/cam0/data" ] && [ -s "$d/ground_truth/pGT/$s.txt" ]; then
        say "$s present"; continue
    fi
    say "$s"
    fetch "$BASE/pinhole_calibrations/training/$s.json" "$d/pinhole_calibrations/$s.json"
    fetch "$BASE/ground_truth/pseudo_dense/$s.txt" "$d/ground_truth/pGT/$s.txt"
    fetch "$BASE/asl_folder/training/$s.zip" "$d/asl_folder/$s.zip"
    unpack "$d/asl_folder/$s.zip" "$d/asl_folder"
done
say "LaMAria ready under $DEST"
