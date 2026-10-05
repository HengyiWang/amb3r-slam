"""TUM RGB-D."""

import os

import cv2
import numpy as np
import PIL.Image
import torch
from scipy.spatial.transform import Rotation

from amb3r_slam.datasets.common import Dataset, Sequence, register

# Nominal intrinsics for every sequence (fr3's focal length, principal point at the
# image centre); they only place the crop.
_FX, _FY = 535.4, 539.2


def _read_list(path, skiprows=0):
    return np.loadtxt(path, delimiter=' ', dtype=str, skiprows=skiprows)


def _associate(t_img, t_dep, t_pose, max_dt=0.08):
    """(image, depth, pose) indices whose timestamps agree within `max_dt`; depth None
    when the sequence has none."""
    out = []
    for i, t in enumerate(t_img):
        k = int(np.argmin(np.abs(t_pose - t)))
        if abs(t_pose[k] - t) >= max_dt:
            continue
        j = None
        if t_dep is not None:
            j = int(np.argmin(np.abs(t_dep - t)))
            if abs(t_dep[j] - t) >= max_dt:
                continue
        out.append((i, j, k))
    return out


def _scene(path):
    """Associated frames of one sequence: image list, depth list (or None), c2w poses."""
    pose_file = ('groundtruth.txt' if os.path.isfile(os.path.join(path, 'groundtruth.txt'))
                 else 'pose.txt')
    imgs = _read_list(os.path.join(path, 'rgb.txt'))
    deps = (_read_list(os.path.join(path, 'depth.txt'))
            if os.path.exists(os.path.join(path, 'depth.txt')) else None)
    pose = _read_list(os.path.join(path, pose_file), skiprows=1)
    assoc = _associate(imgs[:, 0].astype(np.float64),
                       None if deps is None else deps[:, 0].astype(np.float64),
                       pose[:, 0].astype(np.float64))
    vec = pose[:, 1:].astype(np.float64)
    c2w = np.tile(np.eye(4), (len(assoc), 1, 1))
    for n, (_, _, k) in enumerate(assoc):
        c2w[n, :3, :3] = Rotation.from_quat(vec[k, 3:]).as_matrix()
        c2w[n, :3, 3] = vec[k, :3]
    return imgs, deps, assoc, c2w


def _crop_camera(K, size_in, size_out, scaling=1.0):
    """Intrinsics after scaling by `scaling` and centre-cropping to `size_out`."""
    margins = np.asarray(size_in) * scaling - size_out
    assert np.all(margins >= 0.0)
    K = K.copy()
    K[:2, 2] += 0.5
    K[:2, :] *= scaling
    K[:2, 2] -= 0.5 * margins
    K[:2, 2] -= 0.5
    return K


def _crop(image, depth, K, box):
    l, t, r, b = box
    K = K.copy()
    K[0, 2] -= l
    K[1, 2] -= t
    return image.crop(box), None if depth is None else depth[t:b, l:r], K


def _crop_resize(rgb, depth, K, resolution):
    """Crop symmetrically about the principal point, Lanczos-downscale to cover
    `resolution` (W, H), then centre-crop to it; depth follows with nearest sampling."""
    image = PIL.Image.fromarray(rgb)
    W, H = image.size
    cx, cy = K[:2, 2].round().astype(int)
    mx, my = min(cx, W - cx), min(cy, H - cy)
    assert mx > W / 5 and my > H / 5, 'principal point too close to the border'
    image, depth, K = _crop(image, depth, K, (cx - mx, cy - my, cx + mx, cy + my))

    W, H = image.size
    if H > 1.1 * W:
        resolution = resolution[::-1]
    size_in, target = np.array(image.size), np.array(resolution)
    scale = max(target / image.size) + 1e-8
    size = np.floor(size_in * scale).astype(int)
    image = image.resize(tuple(size), resample=PIL.Image.LANCZOS)
    if depth is not None:
        depth = cv2.resize(depth, size, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    K = _crop_camera(K, size_in, size, scaling=scale)

    K2 = _crop_camera(K, image.size, resolution)
    l, t = np.int32(np.round(K[:2, 2] - K2[:2, 2]))
    image, depth, _ = _crop(image, depth, K, (l, t, l + resolution[0], t + resolution[1]))
    return np.array(image), depth


def _intrinsics(H, W):
    return np.array([[_FX, 0, W // 2], [0, _FY, H // 2], [0, 0, 1]], dtype=np.float32)


def load_tum(root, resolution, stride=2, scenes=None, max_frames=None):
    names = sorted(n for n in os.listdir(root) if os.path.isdir(os.path.join(root, n))
                   and (not scenes or any(t in n for t in scenes)))
    for name in names:
        path = os.path.join(root, name)
        imgs, _, assoc, c2w = _scene(path)
        sel = list(range(0, len(assoc), stride))[:max_frames or None]
        frames = []
        for n in sel:
            rgb = cv2.cvtColor(cv2.imread(os.path.join(path, imgs[assoc[n][0], 1])),
                               cv2.COLOR_BGR2RGB)
            img, _ = _crop_resize(rgb, None, _intrinsics(*rgb.shape[:2]), resolution)
            frames.append(torch.from_numpy(img).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0)
        yield name, torch.stack(frames)[None], c2w[sel].astype(np.float32)


def load_tum_depth(root, scene, resolution, stride=2, max_frames=None, scale=5000.0):
    """TUM RGB-D depth for the frames `load_tum` selects, with the same crop."""
    path = os.path.join(root, scene)
    imgs, deps, assoc, _ = _scene(path)
    out = []
    for n in list(range(0, len(assoc), stride))[:max_frames or None]:
        i, j, _ = assoc[n]
        H, W = cv2.imread(os.path.join(path, imgs[i, 1])).shape[:2]
        dep = cv2.imread(os.path.join(path, deps[j, 1]), cv2.IMREAD_UNCHANGED)
        _, d = _crop_resize(np.zeros((H, W, 3), np.uint8), dep.astype(np.float32) / scale,
                            _intrinsics(H, W), resolution)
        out.append(np.asarray(d, dtype=np.float32))
    return torch.from_numpy(np.stack(out))


@register
class Tum(Dataset):
    """TUM RGB-D fr1: monocular, or RGB-D with the Kinect depth."""
    name = 'tum'
    fps = 30
    stride = 2

    def sequences(self, resolution):
        for name, images, poses in load_tum(self.root, resolution, self.stride,
                                            self.args.scenes, self.args.max_frames):
            yield Sequence(name, images, poses)

    def attach(self, seq, backend, cfg, modality, resolution):
        if modality != 'rgbd':
            return super().attach(seq, backend, cfg, modality, resolution)
        from amb3r_slam.metric.depth import ArrayDepth
        mdc = cfg.backend.metric_depth
        dep = load_tum_depth(self.root, seq.name, resolution, self.stride,
                             self.args.max_frames)
        backend.depth_source = ArrayDepth(dep, max_depth=float(mdc.max_depth),
                                          min_depth=float(mdc.min_depth))
        print(f"  [rgbd] depth {tuple(dep.shape)}, "
              f"{float((dep > 0).float().mean()):.2f} valid", flush=True)
