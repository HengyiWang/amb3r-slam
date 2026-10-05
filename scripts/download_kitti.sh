#!/usr/bin/env bash
# KITTI odometry, sequences 00-10 (the ones with ground truth): stereo colour images,
# calibration and poses (37 GB); --lidar adds the Velodyne scans (46 GB).
#
#   bash scripts/download_kitti.sh [DEST] [--lidar] [SEQUENCE ...]     # SEQUENCE: 00 ... 10
#   python run.py --dataset kitti --data_path DEST [--sensor stereo|lidar]
#
# Only those sequences are transferred, read out of KITTI's archives by byte range
# (scripts/remote_zip.py) rather than downloading the 69 GB and 85 GB zips.
source "$(dirname "$0")/common.sh"
DEST="${1:-$ROOT/data/kitti}"; shift || true
BASE=https://s3.eu-central-1.amazonaws.com/avg-kitti
LIDAR=0
if [ "${1:-}" = --lidar ]; then LIDAR=1; shift; fi
SEQ='(0[0-9]|10)'
[ $# -gt 0 ] && SEQ="($(IFS='|'; echo "$*"))"
get () { "$PYTHON" "$ROOT/scripts/remote_zip.py" "$BASE/$1" "$DEST" --match "$2" --strip dataset/; }

say "calibration (with the velodyne-to-camera Tr) and poses"
get data_odometry_calib.zip "sequences/$SEQ/calib\.txt"
get data_odometry_poses.zip "poses/$SEQ\.txt"
say "stereo colour images"
get data_odometry_color.zip "sequences/$SEQ/(image_[23]/|times\.txt)"
if [ $LIDAR = 1 ]; then
    say "Velodyne scans"
    get data_odometry_velodyne.zip "sequences/$SEQ/velodyne/"
fi
say "KITTI ready under $DEST"
