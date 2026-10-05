#!/usr/bin/env bash
# EuRoC MAV, the eleven sequences (ASL format, 11 GB).
#
#   bash scripts/download_euroc.sh [DEST] [SEQUENCE ...]      # SEQUENCE: MH_01_easy ...
#   python run.py --dataset euroc --data_path DEST
#
# ETH's Research Collection (doi 10.3929/ethz-b-000690084) publishes three grouped
# archives, each holding every sequence twice: as the ASL zip and as a ROS bag. Only the
# ASL zips are read out of them (scripts/remote_zip.py), then unpacked to <SEQ>/mav0.
source "$(dirname "$0")/common.sh"
DEST="${1:-$ROOT/data/euroc}"; shift || true
SEQS=(MH_01_easy MH_02_easy MH_03_medium MH_04_difficult MH_05_difficult
      V1_01_easy V1_02_medium V1_03_difficult V2_01_easy V2_02_medium V2_03_difficult)
[ $# -gt 0 ] && SEQS=("$@")
API=https://www.research-collection.ethz.ch/server/api/core/bitstreams
declare -A GROUP=([MH]=7b2419c1-62b5-4714-b7f8-485e5fe3e5fe
                  [V1]=02ecda9a-298f-498b-970c-b7c44334d880
                  [V2]=ea12bc01-3677-4b4c-853d-87c7870b8c44)

for s in "${SEQS[@]}"; do
    if [ -d "$DEST/$s/mav0/cam0/data" ]; then say "$s present"; continue; fi
    say "$s"
    "$PYTHON" "$ROOT/scripts/remote_zip.py" "$API/${GROUP[${s%%_*}]}/content" "$DEST/.zips" \
        --match "/$s\.zip\$" --buffer_mb 256
    unpack "$(find "$DEST/.zips" -name "$s.zip" | head -1)" "$DEST/$s"
    rm -rf "$DEST/$s/__MACOSX"
done
rm -rf "$DEST/.zips"
say "EuRoC ready under $DEST"
