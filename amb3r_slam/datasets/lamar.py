"""LaMAR, as VidMap's benchmark prepares it: phone captures scored by W-AUC."""

import os

import numpy as np

from amb3r_slam.datasets.common import Dataset, Sequence, _LazyBatch, _LazyFrames, register


def _posed(im):
    """A LaMAR rec stores unposed images as the identity, and `has_pose` still says True."""
    T = im.cam_from_world()
    t = np.asarray(T.translation)
    R = np.asarray(T.rotation.matrix())
    return not (np.allclose(t, 0.0) and np.allclose(R, np.eye(3)))


def c2w(im):
    T = im.cam_from_world()
    R = np.asarray(T.rotation.matrix())
    t = np.asarray(T.translation)
    M = np.eye(4)
    M[:3, :3] = R.T
    M[:3, 3] = -R.T @ t
    return M


def posed_sequence(rec, ids, img_root):
    """(image paths, gt index, gt c2w) for the images ``ids`` of one reconstruction, in
    name order, or None when fewer than four are posed."""
    ims = [rec.images[i] for i in ids]
    order = np.argsort([im.name for im in ims])
    ims = [ims[i] for i in order]
    gt_idx = [i for i, im in enumerate(ims) if _posed(im)]
    if len(gt_idx) < 4:
        return None
    paths = [os.path.join(img_root, im.name) for im in ims]
    return paths, np.asarray(gt_idx), np.stack([c2w(ims[i]) for i in gt_idx])


def target_size(src_wh, long_side=504, patch=14):
    """DA3's own input geometry: isotropic scale of the longest side, then patch-align."""
    w, h = src_wh
    s = long_side / max(w, h)
    W, H = int(round(w * s)), int(round(h * s))
    return (W // patch) * patch, (H // patch) * patch


def load_images(paths, resolution=None, long_side=504, patch=14):
    """Isotropic resize to `long_side`, then centre-crop to the patch grid."""
    import cv2
    probe = cv2.imread(paths[0])
    if probe is None:
        raise RuntimeError(f'unreadable {paths[0]}')
    h0, w0 = probe.shape[:2]
    s = long_side / max(w0, h0)
    rw, rh = int(round(w0 * s)), int(round(h0 * s))
    W, H = resolution if resolution else target_size((w0, h0), long_side, patch)
    x0, y0 = (rw - W) // 2, (rh - H) // 2
    u8 = np.empty((len(paths), 3, H, W), dtype=np.uint8)
    for i, p in enumerate(paths):
        im = cv2.imread(p)
        if im is None:
            raise RuntimeError(f'unreadable {p}')
        im = cv2.resize(im[:, :, ::-1], (rw, rh), interpolation=cv2.INTER_AREA)
        u8[i] = im[y0:y0 + H, x0:x0 + W].transpose(2, 0, 1)
    return _LazyBatch(_LazyFrames(u8))


@register
class Lamar(Dataset):
    """LaMAR CAB / HGE / LIN test sequences, uncalibrated, W-AUC as VidMap reports it."""
    name = 'lamar'
    scenes = ('CAB', 'HGE', 'LIN')
    metrics = ('wauc',)

    def resolution(self):
        return None                        # per sequence, from its own image size

    def long_side(self):
        """The longest image side the model is run at."""
        return 512 if self.patch == 16 else 504

    def testset(self, scenes):
        """(scene, name, paths, gt_index, gt_c2w) for every test sequence."""
        import pycolmap
        import yaml
        for scene in scenes:
            rec = pycolmap.Reconstruction(os.path.join(self.root, 'datasets', scene, 'rec'))
            sel = yaml.safe_load(open(os.path.join(self.root, 'testsets', scene,
                                                   'sample.yaml')))
            img_root = os.path.join(self.root, 'datasets', scene, 'images')
            for name in sorted(sel):
                got = posed_sequence(rec, sel[name], img_root)
                if got is None:
                    print(f'[skip] {scene}/{name}: fewer than 4 posed frames')
                    continue
                yield (scene, name) + got

    def sequences(self, resolution):
        for scene, name, paths, gt_idx, gt in self.testset(self.args.scenes
                                                           or list(self.scenes)):
            if self.wanted(f'{scene}/{name}'):
                images = load_images(paths, resolution, self.long_side(), self.patch)
                yield Sequence(f'{scene}/{name}', images, gt, gt_index=gt_idx, scene=scene)
