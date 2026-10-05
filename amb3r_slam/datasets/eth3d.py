"""ETH3D-SLAM."""

import os
import numpy as np
from amb3r_slam.datasets.common import (Dataset, Sequence, _LazyBatch, _LazyFrames,
                                        register)

ETH3D_SCENES = (
    'cables_1', 'cables_2', 'cables_3',
    'camera_shake_1', 'camera_shake_2', 'camera_shake_3',
    'ceiling_1', 'ceiling_2', 'desk_3',
    'desk_changing_1', 'einstein_1', 'einstein_2',
    'einstein_flashlight', 'einstein_global_light_changes_1', 'einstein_global_light_changes_2',
    'einstein_global_light_changes_3', 'kidnap_1', 'sfm_lab_room_1',
    'sfm_lab_room_2', 'large_loop_1', 'mannequin_1',
    'mannequin_3', 'mannequin_4', 'mannequin_5',
    'mannequin_7', 'mannequin_face_1', 'mannequin_face_2',
    'mannequin_face_3', 'mannequin_head', 'motion_1',
    'planar_2', 'planar_3', 'plant_1',
    'plant_2', 'plant_3', 'plant_4',
    'plant_5', 'plant_scene_1', 'plant_scene_2',
    'plant_scene_3', 'reflective_1', 'repetitive',
    'sfm_bench', 'sfm_garden', 'sfm_house_loop',
    'sofa_1', 'sofa_2', 'sofa_3',
    'sofa_4', 'sofa_shake', 'table_3',
    'table_4', 'table_7', 'vicon_light_1',
    'vicon_light_2',
)


def eth3d_resolution(width=518, src=(739, 458), patch=14):
    """ETH3D SLAM's RGB camera is 739x458 after their rectification."""
    W = (width // patch) * patch
    H = int(round(W * src[1] / src[0]))
    return W, max(patch * 2, (H // patch) * patch)


def load_eth3d(root, resolution, scenes=None, stride=1, max_frames=None):
    """ETH3D SLAM training sequences, monocular RGB."""
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    from scipy.spatial.transform import Rotation
    W, H = resolution
    for seq in (scenes or ETH3D_SCENES):
        d = os.path.join(root, seq)
        if not os.path.isfile(os.path.join(d, 'rgb.txt')):
            print(f'[skip] {seq}: no rgb.txt')
            continue
        rows = [ln.split() for ln in open(os.path.join(d, 'rgb.txt'))
                if ln.strip() and not ln.startswith('#')]
        ts_img = np.array([float(r[0]) for r in rows])
        files = [r[1] for r in rows]

        g = np.loadtxt(os.path.join(d, 'groundtruth.txt'))
        ts_gt = g[:, 0]
        j = np.searchsorted(ts_gt, ts_img).clip(1, len(ts_gt) - 1)
        pick = np.where(np.abs(ts_gt[j - 1] - ts_img) < np.abs(ts_gt[j] - ts_img), j - 1, j)
        ok = np.abs(ts_gt[pick] - ts_img) < 0.010
        idx = np.nonzero(ok)[0][::stride]
        if max_frames:
            idx = idx[:max_frames]
        if len(idx) < 16:
            print(f'[skip] {seq}: only {len(idx)} frames with ground truth')
            continue

        gg = g[pick[idx]]
        poses = np.tile(np.eye(4), (len(idx), 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(gg[:, 4:8]).as_matrix()
        poses[:, :3, 3] = gg[:, 1:4]

        fx, fy, cx, cy = np.loadtxt(os.path.join(d, 'calibration.txt'))[:4]
        im0 = cv2.imread(os.path.join(d, files[idx[0]]))
        h0, w0 = im0.shape[:2]
        K = np.array([[fx * W / w0, 0, cx * W / w0],
                      [0, fy * H / h0, cy * H / h0], [0, 0, 1]], np.float32)

        imgs = np.empty((len(idx), 3, H, W), dtype=np.uint8)

        def _one(t):
            i, k = t
            im = cv2.imread(os.path.join(d, files[k]))[:, :, ::-1]
            imgs[i] = cv2.resize(im, (W, H),
                                 interpolation=cv2.INTER_AREA).transpose(2, 0, 1)

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as pool:
            list(pool.map(_one, enumerate(idx)))
        yield f'eth3d_{seq}', _LazyBatch(_LazyFrames(imgs)), poses, K, list(idx)


ETH3D_DEPTH_SCALE = 5000.0     # raw uint16 per metre (TUM convention)


def load_eth3d_depth(seq_dir, resolution, frames=None):
    """(N, H, W) float32 metric depth for an ETH3D sequence, 0 where invalid."""
    import cv2

    W, H = resolution
    pairs = {}
    with open(os.path.join(seq_dir, 'associated.txt')) as fh:
        for ln in fh:
            t = ln.split()
            if len(t) >= 4:
                pairs[t[1]] = t[3]
    rgb_rows = [ln.split() for ln in open(os.path.join(seq_dir, 'rgb.txt'))
                if ln.strip() and not ln.startswith('#')]
    names = [r[1] for r in rgb_rows]
    sel = list(frames) if frames is not None else list(range(len(names)))
    out = np.zeros((len(sel), H, W), np.float32)
    for i, k in enumerate(sel):
        dp = pairs.get(names[k])
        if dp is None:
            continue
        im = cv2.imread(os.path.join(seq_dir, dp), cv2.IMREAD_UNCHANGED)
        if im is None:
            continue
        d = im.astype(np.float32) / ETH3D_DEPTH_SCALE
        out[i] = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
    return out


@register
class Eth3d(Dataset):
    """ETH3D-SLAM training sequences: monocular, or RGB-D with the registered depth."""
    name = 'eth3d'

    def resolution(self):
        return eth3d_resolution(self.width, patch=self.patch)

    def sequences(self, resolution):
        for name, images, poses, K, idx in load_eth3d(
                self.root, resolution, self.args.scenes, self.stride,
                self.args.max_frames):
            yield Sequence(name, images, poses, K, idx)

    def attach(self, seq, backend, cfg, modality, resolution):
        if modality != 'rgbd':
            return super().attach(seq, backend, cfg, modality, resolution)
        import torch
        from amb3r_slam.metric.depth import ArrayDepth
        mdc = cfg.backend.metric_depth
        dep = torch.from_numpy(load_eth3d_depth(
            os.path.join(self.root, seq.name[len('eth3d_'):]), resolution, frames=seq.frames))
        backend.depth_source = ArrayDepth(dep, max_depth=float(mdc.max_depth),
                                          min_depth=float(mdc.min_depth))
        print(f"  [rgbd] depth {tuple(dep.shape)}, "
              f"{float((dep > 0).float().mean()):.2f} valid", flush=True)
