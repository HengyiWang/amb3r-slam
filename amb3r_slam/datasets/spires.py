"""Oxford Spires."""

import os
import zipfile

import numpy as np
from amb3r_slam.datasets.common import (Dataset, Sequence, _LazyBatch, _LazyFrames,
                                        register)

SPIRES_CAM_ID = 0
SPIRES_SEQ_SCENES = ('2024-03-13-observatory-quarter-01', '2024-03-18-christ-church-02',
                     '2024-03-14-blenheim-palace-05', '2024-03-12-keble-college-04',
                     '2024-03-12-keble-college-02', '2024-03-12-keble-college-03',
                     '2024-03-12-keble-college-05', '2024-03-13-observatory-quarter-02',
                     '2024-03-18-christ-church-03', '2024-03-20-christ-church-05',
                     '2024-05-20-bodleian-library-02')


def spires_calib(root, cam_id=SPIRES_CAM_ID):
    """(K, D, (w0, h0), T_cam_lidar) for one Alphasense camera."""
    import yaml
    cdir = os.path.join(root, 'calibration')
    with open(os.path.join(cdir, f'cam{cam_id}.yaml')) as f:
        c = yaml.safe_load(f)
    K = np.array(c['camera_matrix']['data'], np.float64).reshape(3, 3)
    D = np.array(c['distortion_coefficients']['data'], np.float64)
    assert c['distortion_model'] == 'equidistant', c['distortion_model']
    with open(os.path.join(cdir, 'cam-lidar-imu.yaml')) as f:
        T = np.array(yaml.safe_load(f)[f'cam{cam_id}']['T_cam_lidar'],
                     np.float64).reshape(4, 4)
    return K, D, (c['image_width'], c['image_height']), T


def spires_resolution(width=518, src=(1440, 1080), patch=14):
    """Oxford Spires Alphasense cam0 is 1440x1080; keep the aspect, snap to the grid."""
    W = (width // patch) * patch
    H = int(round(width * src[1] / src[0]))
    return W, (H // patch) * patch


def spires_rectifier(root, resolution, cam_id=SPIRES_CAM_ID):
    """(remap_x, remap_y, K_at_target_resolution) for one Alphasense camera."""
    import cv2
    W, H = resolution
    K0, D, (w0, h0), _ = spires_calib(root, cam_id)
    Kr = K0.copy()
    Kr[0, 2], Kr[1, 2] = w0 / 2.0, h0 / 2.0
    m1, m2 = cv2.fisheye.initUndistortRectifyMap(K0, D, np.eye(3), Kr, (w0, h0),
                                                 cv2.CV_16SC2)
    Kt = Kr.copy()
    Kt[0] *= W / w0
    Kt[1] *= H / h0
    return m1, m2, Kt.astype(np.float32)


def spires_T_base_cam(root, cam_id=SPIRES_CAM_ID):
    """`T_base_cam`, so that `C2W = T_WB @ T_base_cam` for a `gt-tum.txt` pose."""
    _, _, _, T_cam_lidar = spires_calib(root, cam_id)
    T_base_lidar = np.eye(4)
    T_base_lidar[0, 0] = T_base_lidar[1, 1] = -1.0
    T_base_lidar[2, 3] = 0.124
    return np.linalg.inv(T_cam_lidar @ np.linalg.inv(T_base_lidar))


def load_spires_seq(root, resolution, scenes, stride=1, max_frames=None,
                    cam_id=SPIRES_CAM_ID):
    """Oxford Spires `sequences/`: the actual capture, at 20 Hz."""
    from concurrent.futures import ThreadPoolExecutor

    import cv2
    from scipy.spatial.transform import Rotation, Slerp
    W, H = resolution
    m1, m2, Kt = spires_rectifier(root, resolution, cam_id)
    T_base_cam = spires_T_base_cam(root, cam_id)

    for seq in scenes:
        sdir = os.path.join(root, 'sequences', seq)
        zpath = os.path.join(sdir, 'raw', 'images.zip')
        gpath = os.path.join(sdir, 'processed', 'trajectory', 'gt-tum.txt')
        if not os.path.isfile(zpath):
            print(f'[skip] {seq}: no {zpath}')
            continue
        have_gt = os.path.isfile(gpath)
        if not have_gt:
            print(f'[no-gt] {seq}: no {gpath}; yielding poses=None '
                  f'(qualitative only, cannot be scored)')
        zf = zipfile.ZipFile(zpath)
        pre = f'cam{cam_id}/'
        names = sorted((e for e in zf.namelist()
                        if e.startswith(pre) and e.lower().endswith('.jpg')),
                       key=lambda e: float(os.path.basename(e)[:-4]))
        ts = np.array([float(os.path.basename(e)[:-4]) for e in names])

        if have_gt:
            g = np.loadtxt(gpath)
            g = g[np.argsort(g[:, 0])]
            inside = (ts >= g[0, 0]) & (ts <= g[-1, 0])
            if inside.sum() < 10:
                print(f'[skip] {seq}: only {int(inside.sum())} images inside the pose span')
                continue
            names = [n for n, k in zip(names, inside) if k]
            ts = ts[inside]

        idx = list(range(0, len(names), stride))
        if max_frames:
            idx = idx[:max_frames]
        names = [names[i] for i in idx]
        ts = ts[idx]

        if have_gt:
            slerp = Slerp(g[:, 0], Rotation.from_quat(g[:, 4:8]))
            poses = np.tile(np.eye(4), (len(ts), 1, 1))
            for j in range(3):
                poses[:, j, 3] = np.interp(ts, g[:, 0], g[:, 1 + j])
            poses[:, :3, :3] = slerp(ts).as_matrix()
            poses = poses @ T_base_cam
        else:
            poses = None

        imgs = np.empty((len(names), 3, H, W), dtype=np.uint8)

        def _one(args):
            j, buf = args
            im = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
            im = cv2.remap(im[:, :, ::-1], m1, m2, cv2.INTER_LINEAR)
            imgs[j] = cv2.resize(im, (W, H),
                                 interpolation=cv2.INTER_AREA).transpose(2, 0, 1)

        nw = min(8, (os.cpu_count() or 4))
        with ThreadPoolExecutor(max_workers=nw) as pool:
            for s0 in range(0, len(names), 256):
                batch = [(j, zf.read(names[j]))
                         for j in range(s0, min(s0 + 256, len(names)))]
                list(pool.map(_one, batch))
        yield f'spires_{seq}', _LazyBatch(_LazyFrames(imgs)), poses, Kt, idx


@register
class Spires(Dataset):
    """Oxford Spires, Alphasense cam0: the 20 Hz capture with TLS-registered ground truth."""
    name = 'spires'
    fps = 20
    scenes = SPIRES_SEQ_SCENES

    def resolution(self):
        return spires_resolution(self.width, patch=self.patch)

    def sequences(self, resolution):
        for name, images, poses, K, idx in load_spires_seq(
                self.root, resolution, self.args.scenes or list(self.scenes),
                self.stride, self.args.max_frames):
            yield Sequence(name, images, poses, K, idx)
