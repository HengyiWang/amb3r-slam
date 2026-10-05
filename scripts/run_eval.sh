#!/usr/bin/env bash
# Runs every benchmark evaluation of the paper, one run per table row.
#
#   bash scripts/run_eval.sh                 # everything
#   bash scripts/run_eval.sh tum kitti       # selected datasets
#
# Each dataset is read from its own variable, e.g. TUM=/path/to/tum KITTI=/path/to/kitti,
# which defaults to $DATA/<dataset> (DATA defaults to data/). A dataset whose directory does
# not exist is skipped. Each run writes to outputs/<dataset>_<sensor>_<model>, ending with
# a summary table.
#
# Every run uses the defaults: uncalibrated, encoder weights in bf16.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
DATA="${DATA:-$ROOT/data}"
OMEGA="--model_name omega"
TUM="${TUM:-$DATA/tum}"
BONN="${BONN:-$DATA/bonn}"
ETH3D="${ETH3D:-$DATA/eth3d}"
EUROC="${EUROC:-$DATA/euroc}"
KITTI="${KITTI:-$DATA/kitti}"
VBR="${VBR:-$DATA/vbr}"
SPIRES="${SPIRES:-$DATA/spires}"
LAMAR="${LAMAR:-$DATA/lamar}"
CROCODL="${CROCODL:-$DATA/crocodl}"
LAMARIA="${LAMARIA:-$DATA/lamaria}"
say () { echo "[$(date +%H:%M:%S)] $*"; }

# A failing sequence is recorded in its run's results and the run goes on; a native library
# can also crash at interpreter exit after everything is written, so a non-zero exit status
# does not stop the remaining runs.
run () { say "run.py $*"; "$PYTHON" "$ROOT/run.py" "$@" || say "run.py exited with status $?"; }

tum () {
    run --dataset tum --data_path "$TUM"
    run --dataset tum --data_path "$TUM" --sensor rgbd
}
bonn () {
    run --dataset bonn --data_path "$BONN"
    run --dataset bonn --data_path "$BONN" $OMEGA
}
eth3d () {
    run --dataset eth3d --data_path "$ETH3D"
    run --dataset eth3d --data_path "$ETH3D" $OMEGA
    run --dataset eth3d --data_path "$ETH3D" --sensor rgbd
}
euroc () {
    run --dataset euroc --data_path "$EUROC"
    run --dataset euroc --data_path "$EUROC" $OMEGA
}
kitti () {
    run --dataset kitti --data_path "$KITTI"
    run --dataset kitti --data_path "$KITTI" --sensor stereo
    run --dataset kitti --data_path "$KITTI" --sensor lidar
}
vbr () {
    run --dataset vbr --data_path "$VBR"
    run --dataset vbr --data_path "$VBR" $OMEGA
    run --dataset vbr --data_path "$VBR" --sensor lidar
}
spires () {
    run --dataset spires --data_path "$SPIRES"
    run --dataset spires --data_path "$SPIRES" $OMEGA
}
lamar () {
    run --dataset lamar --data_path "$LAMAR"
    run --dataset lamar --data_path "$LAMAR" $OMEGA
}
crocodl () {
    run --dataset crocodl --data_path "$CROCODL"
    run --dataset crocodl --data_path "$CROCODL" $OMEGA
}
lamaria () {                     
    run --dataset lamaria --data_path "$LAMARIA"
    run --dataset lamaria --data_path "$LAMARIA" $OMEGA
}

ALL=(tum bonn eth3d euroc kitti vbr spires lamar crocodl)
SEL=("${ALL[@]}")
[ $# -gt 0 ] && SEL=("$@")
for d in "${SEL[@]}"; do
    declare -F "$d" >/dev/null || { echo "unknown dataset: $d" >&2; exit 1; }
    path="${d^^}"
    if [ ! -d "${!path}" ]; then say "skip $d: no data at ${!path} (set $path=...)"; continue; fi
    "$d"
done
say "done; summaries in $ROOT/outputs/*/summary.txt"
