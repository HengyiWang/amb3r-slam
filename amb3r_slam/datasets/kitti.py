"""KITTI odometry."""

import os
import numpy as np
import torch
from amb3r_slam.metric.depth import StereoDepth
from amb3r_slam.datasets.common import Dataset, Sequence, register


def kitti_resolution(width=518, src=(1241, 376), patch=14):
    """Aspect-preserving size for KITTI, rounded to the model's patch grid."""
    h = int(round(width * src[1] / src[0] / patch)) * patch
    return width, max(patch * 2, h)


def load_kitti(root, resolution, scenes, stride=1, max_frames=None):
    """KITTI odometry: image_2 + poses/<seq>.txt (left camera, world<-cam)."""
    import cv2
    W, H = resolution
    for seq in scenes:
        img_dir = os.path.join(root, 'sequences', seq, 'image_2')
        pose_file = os.path.join(root, 'poses', f'{seq}.txt')
        if not os.path.isdir(img_dir):
            print(f"[skip] {img_dir} missing")
            continue
        names = sorted(os.listdir(img_dir))
        idx = list(range(0, len(names), stride))
        if max_frames:
            idx = idx[:max_frames]
        files = [names[i] for i in idx]

        poses = None
        if os.path.exists(pose_file):
            raw = np.loadtxt(pose_file).reshape(-1, 3, 4)[::stride]
            if max_frames:
                raw = raw[:max_frames]
            poses = np.tile(np.eye(4), (len(raw), 1, 1))
            poses[:, :3, :4] = raw
            poses = poses[:len(files)]

        imgs = np.empty((len(files), 3, H, W), dtype=np.float32)
        for i, f in enumerate(files):
            im = cv2.imread(os.path.join(img_dir, f))[:, :, ::-1]
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            imgs[i] = im.transpose(2, 0, 1) / 255.0 * 2.0 - 1.0
        yield f'kitti_{seq}', torch.from_numpy(imgs)[None], poses, None, idx

from amb3r_slam.lidar.scans import ScanSource  # noqa: E402


def kitti_velo_to_cam2(root, seq):
    """(4, 4) rigid transform taking velodyne points into the cam2 frame."""
    calib = os.path.join(root, 'sequences', seq, 'calib.txt')
    M = {}
    with open(calib) as f:
        for line in f:
            if ':' not in line:
                continue
            k, v = line.split(':', 1)
            M[k.strip()] = np.array([float(x) for x in v.split()], np.float64)
    P2 = M['P2'].reshape(3, 4)
    Tr = np.eye(4)
    Tr[:3, :4] = M['Tr'].reshape(3, 4)
    K = P2[:, :3]
    T_c0_c2 = np.eye(4)
    T_c0_c2[:3, 3] = np.linalg.solve(K, P2[:, 3])
    return T_c0_c2 @ Tr


class KittiScans(ScanSource):
    """Velodyne scans, voxel-downsampled and cached as Open3D clouds."""

    kiss_min_range = 5.0
    kiss_max_range = 100.0

    @staticmethod
    def correct_kitti(xyz, deg=0.205, min_z=-5.0):
        """KITTI's HDL-64 intrinsic calibration error, as CT-ICP and KISS-ICP both fix it."""
        xyz = xyz[xyz[:, 2] > min_z]
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        r = np.hypot(x, y)
        ok = r > 1e-9
        th = np.deg2rad(deg)
        c, sn = np.cos(th), np.sin(th)
        out = xyz * c
        out[ok, 0] += sn * (-x[ok] * z[ok]) / r[ok]
        out[ok, 1] += sn * (-y[ok] * z[ok]) / r[ok]
        out[ok, 2] += sn * r[ok]
        out[~ok] = xyz[~ok]
        return out

    def __init__(self, root, seq, voxel=0.5, min_range=3.0, max_range=80.0, cap=512,
                 kitti_correct=False):
        self.dir = os.path.join(root, 'sequences', seq, 'velodyne')
        self.voxel, self.min_range, self.max_range = voxel, min_range, max_range
        self.names = sorted(f for f in os.listdir(self.dir) if f.endswith('.bin'))
        self._c, self._cap = {}, int(cap)
        self._t = {}
        self.kitti_correct = bool(kitti_correct)

    def __len__(self):
        return len(self.names)

    def kiss_points(self, i):
        """(N, 3) returns for odometry, velodyne frame, with KISS-ICP's KITTI correction."""
        from kiss_icp.pybind import kiss_icp_pybind
        p = np.fromfile(os.path.join(self.dir, self.names[int(i)]),
                        np.float32).reshape(-1, 4)
        xyz = p[:, :3].astype(np.float64)
        return np.asarray(kiss_icp_pybind._correct_kitti_scan(
            kiss_icp_pybind._Vector3dVector(xyz)))

    def cloud(self, i):
        i = int(i)
        if i in self._c:
            return self._c[i]
        from amb3r_slam.lidar.registration import _o3d
        o3d = _o3d()
        p = np.fromfile(os.path.join(self.dir, self.names[i]), np.float32).reshape(-1, 4)
        xyz = p[:, :3].astype(np.float64)
        if self.kitti_correct:
            xyz = self.correct_kitti(xyz)
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(r > self.min_range) & (r < self.max_range)]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
        pc = pc.voxel_down_sample(self.voxel)
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
        if len(self._c) >= self._cap:
            self._c.pop(next(iter(self._c)))
        self._c[i] = pc
        return pc


def kitti_stereo_source(root, seq, resolution, stride=1, max_frames=None):
    """StereoDepth for a KITTI odometry sequence, with fx and baseline from calib."""
    P = {}
    with open(os.path.join(root, 'sequences', seq, 'calib.txt')) as fh:
        for line in fh:
            k, _, v = line.partition(':')
            if k in ('P0', 'P1', 'P2', 'P3'):
                P[k] = np.fromstring(v, sep=' ').reshape(3, 4)
    fx = float(P['P2'][0, 0])
    baseline = float((P['P2'][0, 3] - P['P3'][0, 3]) / fx)
    ld = os.path.join(root, 'sequences', seq, 'image_2')
    rd = os.path.join(root, 'sequences', seq, 'image_3')
    lf = sorted(os.listdir(ld))[::stride]
    rf = sorted(os.listdir(rd))[::stride]
    if max_frames:
        lf, rf = lf[:max_frames], rf[:max_frames]
    n = min(len(lf), len(rf))
    return StereoDepth([os.path.join(ld, x) for x in lf[:n]],
                       [os.path.join(rd, x) for x in rf[:n]],
                       fx, baseline, resolution), baseline


@register
class Kitti(Dataset):
    """KITTI odometry, image_2: monocular, stereo with image_3, or LiDAR with the Velodyne."""
    name = 'kitti'
    fps = 10
    scenes = tuple(f'{i:02d}' for i in range(11))     # the sequences with ground truth
    plot_mode = 'xz'

    def resolution(self):
        return kitti_resolution(self.width, patch=self.patch)

    def sequences(self, resolution):
        for name, images, poses, K, idx in load_kitti(
                self.root, resolution, self.args.scenes or list(self.scenes),
                self.stride, self.args.max_frames):
            yield Sequence(name, images, poses, K, idx)

    def attach(self, seq, backend, cfg, modality, resolution):
        seq_id = seq.name[len('kitti_'):]
        if modality == 'stereo':
            src, base = kitti_stereo_source(self.root, seq_id, resolution,
                                            self.stride, self.args.max_frames)
            backend.depth_source = src
            print(f"  [stereo] sgbm, baseline {base:.4f} m, {len(src.left_files)} pairs",
                  flush=True)
        elif modality == 'lidar':
            lc = cfg.backend.lidar
            backend.lidar_reg = {
                'scans': KittiScans(self.root, seq_id, voxel=float(lc.voxel),
                                    min_range=float(lc.min_range),
                                    max_range=float(lc.max_range)),
                'T_cv': kitti_velo_to_cam2(self.root, seq_id),
                'frame_to_scan': list(seq.frames) if seq.frames is not None else None,
            }
            print(f"  [lidar] {seq_id}: {len(backend.lidar_reg['scans'])} scans", flush=True)
        else:
            super().attach(seq, backend, cfg, modality, resolution)
