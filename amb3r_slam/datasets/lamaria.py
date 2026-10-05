"""LaMAria."""

import json
import os
import numpy as np
from amb3r_slam.datasets.common import (Dataset, Sequence, _LazyBatch, _LazyFrames,
                                        register)


def lamaria_resolution(width=518, src=(758, 572), patch=14):
    h = int(round(width * src[1] / src[0] / patch)) * patch
    return width, max(patch * 2, h)


def load_lamaria(root, resolution, scenes=None, stride=1, max_frames=None, cam='cam0'):
    """LaMAria (Aria glasses, city scale) in ASL layout, against the pseudo-GT. The Aria
    camera is mounted sideways, so frames are rotated upright; intrinsics and ground truth
    follow the rotation."""
    import cv2
    from scipy.spatial.transform import Rotation

    W, H = resolution
    troot = os.path.join(root, 'training')
    names = scenes or sorted(d for d in os.listdir(troot)
                             if os.path.isdir(os.path.join(troot, d)))
    for seq in names:
        base = os.path.join(troot, seq)
        adir = os.path.join(base, 'asl_folder', seq, 'aria', cam)
        gt_file = os.path.join(base, 'ground_truth', 'pGT', f'{seq}.txt')
        cal_file = os.path.join(base, 'pinhole_calibrations', f'{seq}.json')
        if not (os.path.isdir(adir) and os.path.isfile(gt_file)
                and os.path.getsize(gt_file) > 0):
            print(f"[skip] {seq}: missing images or pseudo-GT")
            continue
        with open(cal_file) as fh:
            cal = json.load(fh)
        fx, fy, cx, cy = cal[cam]['params']
        W0 = int(cal[cam]['resolution']['width'])
        H0 = int(cal[cam]['resolution']['height'])
        T_bs = np.eye(4)
        T_bs[:3, :3] = Rotation.from_quat(cal[cam]['T_b_s']['qvec']).as_matrix()
        T_bs[:3, 3] = cal[cam]['T_b_s']['tvec']

        listing = np.loadtxt(os.path.join(adir, 'data.csv'), delimiter=',',
                             dtype=str)
        ts_img = listing[:, 0].astype(np.int64)
        files = listing[:, 1]

        gt = np.loadtxt(gt_file)
        ts_gt = gt[:, 0].astype(np.int64)
        j = np.searchsorted(ts_gt, ts_img).clip(1, len(ts_gt) - 1)
        pick = np.where(np.abs(ts_gt[j - 1] - ts_img) < np.abs(ts_gt[j] - ts_img),
                        j - 1, j)
        ok = np.abs(ts_gt[pick] - ts_img) < 25_000_000
        idx = np.nonzero(ok)[0][::stride]
        if max_frames:
            idx = idx[:max_frames]
        if len(idx) < 32:
            print(f"[skip] {seq}: only {len(idx)} frames with ground truth")
            continue

        g = gt[pick[idx]]
        poses = np.tile(np.eye(4), (len(idx), 1, 1))
        poses[:, :3, :3] = Rotation.from_quat(g[:, 4:8]).as_matrix()
        poses[:, :3, 3] = g[:, 1:4]
        poses = poses @ T_bs[None]

        W0, H0 = H0, W0
        fx, fy = fy, fx
        cx, cy = (H0 - 1) - cy, cx
        Rz = np.eye(4)
        Rz[:3, :3] = Rotation.from_rotvec([0, 0, -np.pi / 2]).as_matrix()
        poses = poses @ Rz[None]

        imgs = np.empty((len(idx), 3, H, W), dtype=np.uint8)
        for i, k in enumerate(idx):
            im = cv2.imread(os.path.join(adir, 'data', files[k]))[:, :, ::-1]
            im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
            im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
            imgs[i] = im.transpose(2, 0, 1)
        Kr = np.array([[fx * W / W0, 0, cx * W / W0],
                       [0, fy * H / H0, cy * H / H0], [0, 0, 1]], dtype=np.float64)
        yield (f'lamaria_{seq}', _LazyBatch(_LazyFrames(imgs)), poses, Kr,
               idx, base)


@register
class Lamaria(Dataset):
    """LaMAria (Aria glasses), cam0 against the pseudo ground truth."""
    name = 'lamaria'

    def resolution(self):
        W, H = lamaria_resolution(self.width, patch=self.patch)
        return H, W                        # upright: portrait

    def sequences(self, resolution):
        for name, images, poses, K, idx, _base in load_lamaria(
                self.root, resolution, self.args.scenes, self.stride,
                self.args.max_frames):
            yield Sequence(name, images, poses, K, idx)
