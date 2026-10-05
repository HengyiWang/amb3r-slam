"""Arrange VBR into the layout `--dataset vbr` reads.

    <DEST>/<seq>/rgb/%06d.png          left camera frames
                 intrinsics.txt        3x3 K of the left camera
                 <seq>_gt.txt          camera poses, one row per image, TUM order
                 ouster_points/        sweeps and their timestamps          (LiDAR, VBR metrics)
                 pairs.json            image -> sweep                       (LiDAR, VBR metrics)
                 vbr_calib.yaml        the sequence's calibration           (LiDAR, VBR metrics)
                 gt_official.txt       VBR's per-sweep LiDAR ground truth   (VBR metrics)

`loger`: the images, intrinsics and camera ground truth from LoGeR's preprocessed release
(`vbr/<seq>_processed_aligned/`), whose frames are named by their index in the full
left-camera stream.

`raw`: everything else, from a whole-sequence `vbr convert kitti` export and the
calibration folder `vbr download` writes. An image is paired with the sweep whose timestamp
coincides with its own (the rig is hardware synchronised). A sequence without LoGeR images
gets them from the export instead: the left frames that coincide with a sweep, in time order,
with the intrinsics from the calibration and camera ground truth derived from the official
per-sweep poses as T_world_cam = T_world_lidar inv(T_cam_lidar).

    python scripts/prepare_vbr.py loger <extracted>/vbr <DEST>
    python scripts/prepare_vbr.py raw <export> <calib_dir> <seq> <DEST>
"""

import argparse
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from amb3r_slam.datasets.vbr import read_timestamps, vbr_calib  # noqa: E402

SYNC = 1e-6          # seconds; coincident timestamps on the synchronised rig


def loger(src, dest):
    """`<src>/<seq>_processed_aligned/` -> `<dest>/<seq>/`."""
    for d in sorted(os.listdir(src)):
        if not d.endswith('_processed_aligned'):
            continue
        seq = d[:-len('_processed_aligned')]
        out = os.path.join(dest, seq)
        os.makedirs(out, exist_ok=True)
        for name, target in (('rgb', 'rgb'), ('intrinsics.txt', 'intrinsics.txt'),
                             ('camera_pose.txt', f'{seq}_gt.txt')):
            p = os.path.join(src, d, name)
            if os.path.exists(p) and not os.path.exists(os.path.join(out, target)):
                shutil.move(p, os.path.join(out, target))
        print(f'  {seq}: {len(os.listdir(os.path.join(out, "rgb")))} images')


def coincident(t_query, t_sweep):
    """Index of the nearest sweep for each query time, and whether it coincides."""
    j = np.clip(np.searchsorted(t_sweep, t_query), 1, len(t_sweep) - 1)
    j = np.where(np.abs(t_sweep[j] - t_query) < np.abs(t_sweep[j - 1] - t_query), j, j - 1)
    return j, np.abs(t_sweep[j] - t_query) < SYNC


def pairs_for(names, t_frame, t_sweep):
    j, ok = coincident(t_frame, t_sweep)
    return [{'left': n, 'sweep': int(j[i]), 'dt': float(t_sweep[j[i]] - t_frame[i])}
            for i, n in enumerate(names) if ok[i]]


def camera_gt(names, pairs, t_sweep, official, T_cam_lidar):
    """Rows `i tx ty tz qx qy qz qw` for the images with a paired, posed sweep."""
    from scipy.spatial.transform import Rotation
    g = np.loadtxt(official)
    P = np.tile(np.eye(4), (len(g), 1, 1))
    P[:, :3, :3] = Rotation.from_quat(g[:, 4:8]).as_matrix()
    P[:, :3, 3] = g[:, 1:4]
    by_name = {p['left']: p['sweep'] for p in pairs}
    T_lidar_cam = np.linalg.inv(T_cam_lidar)
    rows = []
    for i, n in enumerate(names):
        s = by_name.get(n)
        if s is None:
            continue
        k = int(np.argmin(np.abs(g[:, 0] - t_sweep[s])))
        if abs(g[k, 0] - t_sweep[s]) > 0.05:
            continue
        T = P[k] @ T_lidar_cam
        rows.append((i, *T[:3, 3], *Rotation.from_matrix(T[:3, :3]).as_quat()))
    return rows


def raw(export, calib_dir, seq, dest):
    out = os.path.join(dest, seq)
    os.makedirs(out, exist_ok=True)
    calib = os.path.join(calib_dir, 'vbr_calib.yaml')
    official = os.path.join(calib_dir, f'{seq}_gt.txt')
    shutil.copy(calib, os.path.join(out, 'vbr_calib.yaml'))
    if os.path.exists(official):
        shutil.copy(official, os.path.join(out, 'gt_official.txt'))

    sweeps = os.path.join(out, 'ouster_points')
    if not os.path.isdir(sweeps):
        shutil.move(os.path.join(export, 'ouster_points'), sweeps)
    t_sweep = read_timestamps(os.path.join(sweeps, 'timestamps.txt'))
    t_left = read_timestamps(os.path.join(export, 'camera_left', 'timestamps.txt'))

    rgb = os.path.join(out, 'rgb')
    if os.path.isdir(rgb):
        names = sorted(os.listdir(rgb))
        pairs = pairs_for(names, t_left[[int(os.path.splitext(n)[0]) for n in names]], t_sweep)
    else:
        files = sorted(os.listdir(os.path.join(export, 'camera_left', 'data')))
        assert len(files) == len(t_left), 'left images and timestamps disagree'
        _, ok = coincident(t_left, t_sweep)
        os.makedirs(rgb)
        names = []
        for f in np.asarray(files)[ok]:
            names.append('%06d.png' % len(names))
            shutil.move(os.path.join(export, 'camera_left', 'data', f),
                        os.path.join(rgb, names[-1]))
        pairs = pairs_for(names, t_left[ok], t_sweep)
        K = vbr_calib(calib)[0]
        np.savetxt(os.path.join(out, 'intrinsics.txt'), K)
        if os.path.exists(official):
            rows = camera_gt(names, pairs, t_sweep, official, vbr_calib(calib)[3])
            with open(os.path.join(out, f'{seq}_gt.txt'), 'w') as f:
                f.write('# timestamp tx ty tz qx qy qz qw\n')
                for r in rows:
                    f.write('%d %.17g %.17g %.17g %.17g %.17g %.17g %.17g\n' % r)
    with open(os.path.join(out, 'pairs.json'), 'w') as f:
        json.dump({'pairs': pairs}, f)
    print(f'  {seq}: {len(names)} images, {len(t_sweep)} sweeps, {len(pairs)} paired')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    sub = ap.add_subparsers(dest='step', required=True)
    a = sub.add_parser('loger')
    a.add_argument('src')
    a.add_argument('dest')
    b = sub.add_parser('raw')
    for k in ('export', 'calib_dir', 'seq', 'dest'):
        b.add_argument(k)
    args = ap.parse_args()
    if args.step == 'loger':
        loger(args.src, args.dest)
    else:
        raw(args.export, args.calib_dir, args.seq, args.dest)


if __name__ == '__main__':
    main()
