"""How much do two views actually see of the same place?"""

import os

import numpy as np
import torch


_ALIKED = {}


def _aliked(cfg, device):
    """Build the detector once per process; it is ~1 M parameters."""
    name = str(cfg.get('aliked_model', 'aliked-n16'))
    n_limit = int(cfg['max_corners'])
    key = (name, device, n_limit, float(cfg['aliked_score']))
    if key in _ALIKED:
        return _ALIKED[key]

    from amb3r_slam import thirdparty
    root = thirdparty('ALIKED')
    cwd = os.getcwd()
    try:
        os.chdir(root)
        from nets.aliked import ALIKED
        det = ALIKED(model_name=name, device=device, top_k=-1,
                     scores_th=float(cfg['aliked_score']), n_limit=n_limit)
    finally:
        os.chdir(cwd)
    det = det.to(device).eval()
    _ALIKED[key] = det
    return det


def _match(d0, d1, ratio):
    if d0.shape[0] < 2 or d1.shape[0] < 2:
        return np.zeros((0, 2), np.int64)
    sim = d0 @ d1.T
    top, idx = sim.topk(2, dim=1)
    back = sim.argmax(dim=0)
    mutual = back[idx[:, 0]] == torch.arange(d0.shape[0], device=d0.device)
    passes = (2 - 2 * top[:, 0]) < (ratio ** 2) * (2 - 2 * top[:, 1]).clamp_min(1e-8)
    keep = mutual & passes
    a = torch.nonzero(keep, as_tuple=False).reshape(-1)
    return torch.stack([a, idx[a, 0]], dim=1).cpu().numpy()


class AlikedCovis:
    """Pairwise covisibility from ALIKED keypoints, cached per frame."""

    def __init__(self, cfg=None, device='cuda'):
        cfg = dict(cfg or {})
        cfg.setdefault('aliked_model', 'aliked-n16')
        self.cfg = cfg
        self.device = device
        self.ratio = float(cfg['ratio'])
        self.ransac_px = float(cfg['ransac_px'])
        self.min_matches = int(cfg.get('min_matches', 8))
        self.det = _aliked(cfg, device)
        self._feat = {}
        self._pair = {}

    def evict_before(self, f):
        """Drop cached features and pairs of frames before ``f``; windows only move forward,
        so they are never asked for again."""
        self._feat = {k: v for k, v in self._feat.items() if k >= f}
        self._pair = {k: v for k, v in self._pair.items() if k[0] >= f}

    @torch.no_grad()
    def features(self, images, f):
        """Keypoints in pixels and L2-normalised descriptors for frame ``f``."""
        if f in self._feat:
            return self._feat[f]
        im = images[f]
        x = ((im.float() + 1.0) * 0.5).clamp(0, 1).unsqueeze(0).to(self.device)
        p = self.det(x)
        k = p['keypoints'][0]
        h, w = x.shape[-2:]
        wh = torch.tensor([w - 1, h - 1], device=k.device, dtype=k.dtype)
        kps = (wh * (k + 1) / 2).cpu().numpy().astype(np.float32)
        self._feat[f] = (kps, p['descriptors'][0])
        return self._feat[f]


    def pair(self, images, f, g):
        """``{'covis', 'raw', 'inliers', 'matches', 'n_kpts'}`` for one frame pair."""
        if f == g:
            k, _ = self.features(images, f)
            n = len(k)
            return {'covis': 1.0, 'raw': 1.0, 'inliers': n, 'matches': n, 'n_kpts': n}
        key = (min(f, g), max(f, g))
        if key in self._pair:
            return self._pair[key]

        import cv2
        k0, d0 = self.features(images, key[0])
        k1, d1 = self.features(images, key[1])
        n_min = min(len(k0), len(k1))
        out = {'covis': 0.0, 'raw': 0.0, 'inliers': 0, 'matches': 0, 'n_kpts': int(n_min)}
        if n_min < 2:
            self._pair[key] = out
            return out
        m = _match(d0, d1, self.ratio)
        out['matches'] = int(len(m))
        out['raw'] = float(len(m) / max(n_min, 1))
        if len(m) >= self.min_matches:
            p0 = k0[m[:, 0]].astype(np.float64)
            p1 = k1[m[:, 1]].astype(np.float64)
            _, mask = cv2.findFundamentalMat(p0, p1, cv2.FM_RANSAC,
                                             self.ransac_px, 0.999)
            if mask is not None:
                keep = mask.reshape(-1).astype(bool)
                inl = int(keep.sum())
                if inl >= self.min_matches:
                    out['inliers'] = inl
                    out['covis'] = float(inl / max(n_min, 1))
        self._pair[key] = out
        return out

    def span_pair(self, images, frames_a, frames_b, strong_at=0.05):
        """Covisibility between two *sets* of frames, e.g. two graph nodes' spans."""
        fa = [f for f in frames_a if f not in set(frames_b)]
        fb = [f for f in frames_b if f not in set(frames_a)]
        if not fa or not fb:
            return {'max': 0.0, 'mean': 0.0, 'n_pairs': 0, 'n_strong': 0,
                    'frac_strong': 0.0, 'raw_max': 0.0, 'best': None}
        vals, raws, best, bv = [], [], None, -1.0
        for f in fa:
            for g in fb:
                r = self.pair(images, f, g)
                vals.append(r['covis'])
                raws.append(r['raw'])
                if r['covis'] > bv:
                    bv, best = r['covis'], (int(f), int(g))
        v = np.asarray(vals)
        n_strong = int((v >= strong_at).sum())
        return {'max': float(v.max()), 'mean': float(v.mean()),
                'n_pairs': int(v.size), 'n_strong': n_strong,
                'frac_strong': float(n_strong / max(v.size, 1)),
                'raw_max': float(max(raws)), 'best': best}

