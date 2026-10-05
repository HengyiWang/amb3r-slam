import numpy as np
import torch


class DepthSource:
    """Base: metric depth for global frame indices, at the model's input resolution."""

    def frame_depth(self, frames):
        """-> (len(frames), H, W) float32, 0 where there is no measurement."""
        raise NotImplementedError


class ArrayDepth(DepthSource):
    """A precomputed (T, H, W) stack, indexed by global frame id."""

    def __init__(self, depth, max_depth=0.0, min_depth=0.0):
        self.depth = depth if isinstance(depth, torch.Tensor) else torch.as_tensor(depth)
        self.max_depth = float(max_depth)
        self.min_depth = float(min_depth)

    def frame_depth(self, frames):
        d = self.depth[torch.as_tensor(list(frames))].clone().float()
        if self.min_depth > 0:
            d[d < self.min_depth] = 0.0
        if self.max_depth > 0:
            d[d > self.max_depth] = 0.0
        return d


@torch.no_grad()
def submap_metric_scale(pts, poses, sensor_d, cfg, conf=None):
    """One robust scale taking a submap's units into metres."""
    F = pts.shape[0]
    R = poses[:, :3, :3]
    c = poses[:, :3, 3]
    z = torch.einsum('fij,fhwj->fhwi', R.transpose(1, 2),
                     pts - c[:, None, None, :])[..., 2]
    s_d = sensor_d.to(z.device)
    ok = (s_d > 1e-6) & (z > 1e-6) & torch.isfinite(z) & torch.isfinite(s_d)
    conf_rel = float(cfg.conf_rel)
    if conf is not None and conf_rel > 0:
        c = conf.to(z.device).float()
        med = c[c > 0].median() if bool((c > 0).any()) else None
        if med is not None and float(med) > 0:
            ok = ok & (c > conf_rel * med)
    min_px = int(cfg.min_pixels)
    per, counts, inl = [], [], []
    inlier_tol = float(cfg.inlier_tol)
    for f in range(F):
        m = ok[f]
        n = int(m.sum())
        if n < min_px:
            continue
        r = torch.log(s_d[f][m] / z[f][m])
        r_all = r
        zz = s_d[f][m]
        mode = str(cfg.depth_weight)
        if mode == 'inv_sq':
            w = 1.0 / zz.clamp_min(1e-6) ** 2
        elif mode == 'inv':
            w = 1.0 / zz.clamp_min(1e-6)
        else:
            w = torch.ones_like(zz)
        trim = float(cfg.trim)
        if 0.0 < trim < 0.5:
            lo = torch.quantile(r, trim)
            hi = torch.quantile(r, 1.0 - trim)
            keep_r = (r >= lo) & (r <= hi)
            r, w = r[keep_r], w[keep_r]
        if r.numel() < 64:
            continue
        fit_f = float((w * r).sum() / w.sum().clamp_min(1e-12))
        per.append(fit_f)
        counts.append(n)
        inl.append(float(((r_all - fit_f).abs() < inlier_tol).float().mean()))
    if len(per) < max(1, int(cfg.min_frames)):
        return None, {'reason': f'only {len(per)} usable frames'}
    lp = np.asarray(per)
    s = float(np.exp(np.median(lp)))
    spread = float(np.std(lp))
    if not np.isfinite(s) or s <= 0:
        return None, {'reason': f'bad scale {s}'}
    max_spread = float(cfg.max_spread)
    if spread > max_spread:
        return None, {'reason': f'spread {spread:.3f} > {max_spread}',
                      'spread': spread}
    inlier_frac = float(np.median(inl)) if inl else float('nan')
    min_inl = float(cfg.min_inlier_frac)
    if min_inl > 0 and np.isfinite(inlier_frac) and inlier_frac < min_inl:
        return None, {'reason': f'inlier {inlier_frac:.2f} < {min_inl}',
                      'inlier_frac': inlier_frac}
    info = {'scale': s, 'spread': spread, 'frames': len(per),
            'inlier_frac': inlier_frac, 'pixels': int(np.median(counts))}
    return s, info


class StereoDepth(DepthSource):
    """Metric depth from a rectified stereo pair, computed on demand. ``max_depth`` 0 keeps
    the far field, which the 1/z^2 weighting already fades out."""

    def __init__(self, left_files, right_files, fx, baseline, resolution,
                 max_depth=0.0, min_depth=3.0, num_disp=128, block=5, cache=512):
        self.left_files = list(left_files)
        self.right_files = list(right_files)
        self.fx_full = float(fx)
        self.baseline = float(baseline)
        self.resolution = resolution
        self.max_depth = float(max_depth)
        self.min_depth = float(min_depth)
        self.num_disp = int(num_disp)
        self.block = int(block)
        self._cache = {}
        self._cache_cap = int(cache)
        self._matcher = None

    def _sgbm(self):
        import cv2
        if self._matcher is None:
            b = self.block
            self._matcher = cv2.StereoSGBM_create(
                minDisparity=0, numDisparities=self.num_disp, blockSize=b,
                P1=8 * 3 * b * b, P2=32 * 3 * b * b, disp12MaxDiff=1,
                uniquenessRatio=10, speckleWindowSize=100, speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        return self._matcher

    def _depth_of(self, f):
        import cv2
        if f in self._cache:
            return self._cache[f]
        L = cv2.imread(self.left_files[f], cv2.IMREAD_GRAYSCALE)
        R = cv2.imread(self.right_files[f], cv2.IMREAD_GRAYSCALE)
        if L is None or R is None:
            return None
        disp = self._sgbm().compute(L, R).astype(np.float32) / 16.0
        W, H = self.resolution
        d = np.zeros_like(disp)
        ok = disp > 0.5
        d[ok] = self.fx_full * self.baseline / disp[ok]
        d[~ok] = 0.0
        if self.max_depth > 0:
            d[d > self.max_depth] = 0.0
        if self.min_depth > 0:
            d[d < self.min_depth] = 0.0
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_NEAREST)
        if len(self._cache) >= self._cache_cap:
            self._cache.pop(next(iter(self._cache)))
        self._cache[f] = d
        return d

    def frame_depth(self, frames):
        W, H = self.resolution
        out = np.zeros((len(frames), H, W), dtype=np.float32)
        for i, f in enumerate(frames):
            if 0 <= f < len(self.left_files):
                d = self._depth_of(int(f))
                if d is not None:
                    out[i] = d
        return torch.from_numpy(out)
