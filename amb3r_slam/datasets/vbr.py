"""VBR (Vision Benchmark in Rome)."""

import json
import os
import numpy as np
import torch
from amb3r_slam.datasets.common import (Dataset, Sequence, _LazyBatch, _LazyFrames,
                                        register)

VBR_SEQS = ('campus_train0', 'campus_train1', 'ciampino_train0', 'ciampino_train1',
            'colosseo_train0', 'diag_train0', 'pincio_train0', 'spagna_train0')


def vbr_resolution(width=518, src=(1280, 720), patch=14):
    """VBR is 1280x720; keep the aspect and snap to the patch grid, as KITTI does."""
    W = (width // patch) * patch
    H = int(round(width * src[1] / src[0]))
    return W, (H // patch) * patch


def load_vbr(root, resolution, scenes, stride=1, max_frames=None):
    import cv2
    W, H = resolution
    for seq in scenes:
        d = os.path.join(root, seq)
        img_dir = os.path.join(d, 'rgb')
        gt_file = os.path.join(d, f'{seq}_gt.txt')
        if not os.path.isdir(img_dir):
            print(f'[skip] {img_dir} missing')
            continue
        names = sorted(os.listdir(img_dir))
        idx = list(range(0, len(names), stride))
        if max_frames:
            idx = idx[:max_frames]
        files = [names[i] for i in idx]

        poses = None
        if os.path.exists(gt_file):
            raw = np.loadtxt(gt_file)
            raw = raw[idx] if len(raw) >= len(names) else raw
            if max_frames:
                raw = raw[:max_frames]
            n = min(len(raw), len(files))
            raw, files, idx = raw[:n], files[:n], idx[:n]
            poses = np.tile(np.eye(4), (n, 1, 1))
            poses[:, :3, 3] = raw[:, 1:4]
            from amb3r_slam.tools.geometry import quat_to_rmat
            poses[:, :3, :3] = quat_to_rmat(
                torch.from_numpy(raw[:, 4:8]).float()).numpy()

        probe = cv2.imread(os.path.join(img_dir, files[0]))
        h0, w0 = probe.shape[:2]
        K = None
        kf = os.path.join(d, 'intrinsics.txt')
        if os.path.exists(kf):
            K = np.loadtxt(kf).reshape(3, 3).astype(np.float32)
            K[0] *= W / w0
            K[1] *= H / h0

        imgs = np.empty((len(files), 3, H, W), dtype=np.uint8)
        for i, f in enumerate(files):
            im = cv2.imread(os.path.join(img_dir, f))[:, :, ::-1]
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            imgs[i] = im.transpose(2, 0, 1)
        yield f'vbr_{seq}', _LazyBatch(_LazyFrames(imgs)), poses, K, idx

from amb3r_slam.lidar.scans import ScanSource  # noqa: E402


def _iso_seconds(s):
    """Seconds, from either VBR's ISO stamp or a bare float."""
    s = s.strip()
    if 'T' not in s:
        return float(s)
    t = s.split('T')[-1]
    h, m, sec = t.split(':')
    return int(h) * 3600 + int(m) * 60 + float(sec)


def read_timestamps(path):
    with open(path) as fh:
        return np.array([_iso_seconds(ln) for ln in fh if ln.strip()])


def vbr_calib(calib_file, cam='cam_l'):
    """``(K, D, (W, H), T_cam_lidar)`` from the sequence's `vbr_calib.yaml`."""
    import yaml
    c = yaml.safe_load(open(calib_file))
    fx, fy, cx, cy = c[cam]['intrinsics']
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)
    D = np.asarray(c[cam]['distortion_coeffs'], np.float64)
    W, H = c[cam]['resolution']
    T_b_cam = np.asarray(c[cam]['T_b'], np.float64)
    T_b_lidar = np.asarray(c['lidar']['T_b'], np.float64)
    return K, D, (int(W), int(H)), np.linalg.inv(T_b_cam) @ T_b_lidar


def vbr_frame_to_sweep(pairs_json, names):
    """Map each loaded image (by its filename) to a sweep index, via `pairs.json`."""
    d = json.load(open(pairs_json))
    by_name = {p['left']: p['sweep'] for p in d['pairs']}
    return [by_name.get(os.path.basename(n)) for n in names]


class VbrScans(ScanSource):
    """Ouster sweeps from a `vbr convert kitti` export, cached as Open3D clouds."""

    kiss_min_range = 1.0
    kiss_max_range = 100.0

    def __init__(self, root, voxel=0.5, min_range=1.0, max_range=80.0, cap=512):
        self.dir = os.path.join(root, 'ouster_points', 'data')
        self.voxel, self.min_range, self.max_range = voxel, min_range, max_range
        self.names = sorted(f for f in os.listdir(self.dir) if f.endswith('.bin'))
        self._c, self._cap, self._t, self._p = {}, int(cap), {}, {}
        ts = os.path.join(root, 'ouster_points', 'timestamps.txt')
        self.ts = read_timestamps(ts) if os.path.isfile(ts) else None

    def __len__(self):
        return len(self.names)

    def kiss_points(self, i):
        """Raw (N, 3) valid returns, LiDAR frame, before any downsampling."""
        i = int(i)
        if i in self._p:
            return self._p[i]
        if i < 0 or i >= len(self.names):
            return np.zeros((0, 3))
        p = np.fromfile(os.path.join(self.dir, self.names[i]), np.float32).reshape(-1, 4)
        xyz = p[:, :3].astype(np.float64)
        out = xyz[np.linalg.norm(xyz, axis=1) > 1e-6]
        if len(self._p) >= self._cap:
            self._p.pop(next(iter(self._p)))
        self._p[i] = out
        return out

    def cloud(self, i):
        from amb3r_slam.lidar.registration import _o3d
        i = int(i)
        if i in self._c:
            return self._c[i]
        o3d = _o3d()
        if i < 0 or i >= len(self.names):
            self._c[i] = o3d.geometry.PointCloud()
            return self._c[i]
        xyz = self.kiss_points(i)
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(r > self.min_range) & (r < self.max_range)]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        if self.voxel > 0 and len(xyz):
            pc = pc.voxel_down_sample(self.voxel)
        if len(pc.points):
            pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        if len(self._c) >= self._cap:
            self._c.pop(next(iter(self._c)))
        self._c[i] = pc
        return pc


@register
class Vbr(Dataset):
    name = 'vbr'
    fps = 10
    scenes = VBR_SEQS
    metrics = ('ate', 'vbr')

    def resolution(self):
        return vbr_resolution(self.width, patch=self.patch)

    def sequences(self, resolution):
        for name, images, poses, K, idx in load_vbr(
                self.root, resolution, self.args.scenes or list(self.scenes),
                self.stride, self.args.max_frames):
            yield Sequence(name, images, poses, K, idx)

    def attach(self, seq, backend, cfg, modality, resolution):
        if modality != 'lidar':
            return super().attach(seq, backend, cfg, modality, resolution)
        lc = cfg.backend.lidar
        seq_v = seq.name[len('vbr_'):]
        sdir = os.path.join(self.root, seq_v)
        if not os.path.isdir(os.path.join(sdir, 'ouster_points')):
            raise FileNotFoundError(f'{seq_v}: no ouster_points; fetch the raw recordings with '
                                    f'scripts/download_vbr.sh (without --images-only)')
        calib = os.path.join(sdir, 'vbr_calib.yaml')
        scans = VbrScans(sdir, voxel=float(lc.voxel), min_range=1.0,
                         max_range=float(lc.max_range))
        names_all = sorted(os.listdir(os.path.join(sdir, 'rgb')))
        sel = list(seq.frames) if seq.frames is not None else range(len(names_all))
        f2s = vbr_frame_to_sweep(os.path.join(sdir, 'pairs.json'),
                                 [names_all[i] for i in sel])
        backend.lidar_reg = {'scans': scans, 'T_cv': vbr_calib(calib)[3],
                             'frame_to_scan': f2s}
        print(f'  [lidar] {seq_v}: {len(scans)} sweeps, '
              f'{sum(1 for x in f2s if x is not None)}/{len(f2s)} frames mapped', flush=True)
