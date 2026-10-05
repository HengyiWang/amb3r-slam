"""Robust Sim(3) alignment between two pointmaps."""

import numpy as np
import torch

from amb3r_slam.tools.geometry import average_transforms, rmat_to_quat


def weighted_umeyama(src, dst, w, with_scale=True):
    """Closed-form weighted Sim(3) mapping ``src`` onto ``dst``."""
    tot = w.sum()
    if not torch.isfinite(tot) or tot < 1e-12:
        raise ValueError("degenerate weights")
    wn = (w / tot).to(torch.float64)
    src, dst = src.to(torch.float64), dst.to(torch.float64)

    mu_s = (wn[:, None] * src).sum(0)
    mu_d = (wn[:, None] * dst).sum(0)
    sc, dc = src - mu_s, dst - mu_d

    if with_scale:
        spread_s = torch.sqrt((wn * (sc ** 2).sum(-1)).sum()).clamp_min(1e-12)
        spread_d = torch.sqrt((wn * (dc ** 2).sum(-1)).sum())
        s = (spread_d / spread_s).clamp(1e-6, 1e6)
    else:
        s = torch.ones((), dtype=torch.float64, device=src.device)

    rw = wn.sqrt()[:, None]
    H = (s * sc * rw).T @ (dc * rw)
    U, _, Vt = torch.linalg.svd(H)
    R = Vt.T @ U.T
    if torch.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1] *= -1
        R = Vt.T @ U.T

    t = mu_d - s * (R @ mu_s)
    return s, R, t


def _lmeds_init(src, dst, w, tries=32, sample=6, generator=None):
    """Least-median-of-squares seed for IRLS."""
    n = src.shape[0]
    best, best_med = None, float('inf')
    for _ in range(tries):
        idx = torch.randint(0, n, (sample,), device=src.device, generator=generator)
        try:
            cand = weighted_umeyama(src[idx], dst[idx], w[idx].clamp_min(1e-6), True)
        except Exception:
            continue
        s, R, t = cand
        if not (torch.isfinite(s) and torch.isfinite(R).all() and torch.isfinite(t).all()):
            continue
        med = float((dst.to(torch.float64) - (s * (src.to(torch.float64) @ R.T) + t))
                    .norm(dim=-1).median())
        if med < best_med:
            best, best_med = cand, med
    return best


def robust_sim3(src, dst, weights, delta_rel=0.1, max_iters=10, tol=1e-9,
                with_scale=True, lmeds_tries=32):
    s, R, t = weighted_umeyama(src, dst, weights, with_scale)
    if lmeds_tries > 0:
        plain = float((dst.to(torch.float64) - (s * (src.to(torch.float64) @ R.T) + t))
                      .norm(dim=-1).median())
        seed = _lmeds_init(src, dst, weights, tries=lmeds_tries)
        if seed is not None:
            s2, R2, t2 = seed
            med2 = float((dst.to(torch.float64) - (s2 * (src.to(torch.float64) @ R2.T) + t2))
                         .norm(dim=-1).median())
            if med2 < plain:
                s, R, t = s2, R2, t2
    src64, dst64 = src.to(torch.float64), dst.to(torch.float64)
    w0 = weights.to(torch.float64)
    prev = float('inf')
    delta = None

    for _ in range(max_iters):
        res = (dst64 - (s * (src64 @ R.T) + t)).norm(dim=-1)
        d_now = (delta_rel * res.median()).clamp_min(1e-12)
        delta = d_now if delta is None else torch.minimum(delta, d_now)
        hw = torch.where(res <= delta, torch.ones_like(res), delta / res.clamp_min(1e-12))
        w = w0 * hw
        if w.sum() < 1e-12:
            break
        s_new, R_new, t_new = weighted_umeyama(src, dst, w, with_scale)

        cost = float((torch.where(res <= delta, 0.5 * res ** 2,
                                  delta * (res - 0.5 * delta)) * w0).sum())
        moved = float((s_new - s).abs() + (t_new - t).norm())
        s, R, t = s_new, R_new, t_new
        if moved < tol or abs(prev - cost) < tol * max(prev, 1e-12):
            prev = cost
            break
        prev = cost

    res = (dst64 - (s * (src64 @ R.T) + t)).norm(dim=-1)
    scale_ref = dst64.norm(dim=-1).median().clamp_min(1e-9)
    return s, R, t, {'residual_rel': float(res.median() / scale_ref)}


def to_sim3(s, R, t):
    """(s, R, t) -> the (8,) ``[t, q_xyzw, s]`` layout used everywhere else."""
    return torch.cat([t.float(), rmat_to_quat(R.float()), s.float().reshape(1)])


def cloud_overlap(pts_a, conf_a, pts_b, conf_b, voxel_rel=0.02, conf_rel=0.5):
    """Fraction of one point cloud that falls where the other one has geometry."""
    def keep(p, c):
        thr = conf_rel * float(torch.median(c))
        m = (c > thr) & torch.isfinite(p).all(-1)
        return p[m]

    a = keep(pts_a.reshape(-1, 3), conf_a.reshape(-1))
    b = keep(pts_b.reshape(-1, 3), conf_b.reshape(-1))
    if a.shape[0] < 100 or b.shape[0] < 100:
        return 0.0

    scale = torch.median(torch.cat([a.norm(dim=-1), b.norm(dim=-1)])).clamp_min(1e-6)
    voxel = (voxel_rel * scale).clamp_min(1e-9)

    def keys(p):
        q = torch.floor(p / voxel).to(torch.int64)
        return q[:, 0] * 73_856_093 + q[:, 1] * 19_349_663 + q[:, 2] * 83_492_791

    ka, kb = keys(a), keys(b)
    ua = torch.unique(ka)
    ub = torch.unique(kb)
    frac_b_in_a = float(torch.isin(kb, ua).float().mean())
    frac_a_in_b = float(torch.isin(ka, ub).float().mean())
    return min(frac_a_in_b, frac_b_in_a)


def align_overlap(pts_a, conf_a, pts_b, conf_b, cfg):
    """Sim(3) taking submap *b*'s coordinates into submap *a*'s, from their overlap."""
    pts_a = pts_a.reshape(-1, 3)
    pts_b = pts_b.reshape(-1, 3)
    ca, cb = conf_a.reshape(-1), conf_b.reshape(-1)

    thr = float(cfg['conf_rel']) * float(torch.median(torch.cat([ca, cb])))
    keep = (ca > thr) & (cb > thr) & torch.isfinite(pts_a).all(-1) & torch.isfinite(pts_b).all(-1)
    n = int(keep.sum())
    if n < int(cfg['min_points']):
        return None, {'reason': 'too few confident points'}

    w = torch.sqrt(ca[keep] * cb[keep])
    try:
        s, R, t, info = robust_sim3(pts_b[keep], pts_a[keep], w,
                                    delta_rel=float(cfg['huber_rel']),
                                    max_iters=int(cfg['max_iters']))
    except Exception as e:
        return None, {'reason': f'{type(e).__name__}: {e}'}

    g = to_sim3(s, R, t)
    if not torch.isfinite(g).all():
        return None, {'reason': 'non-finite transform'}
    return g, info


def align_overlap_poses(pts_a, conf_a, poses_a, pts_b, conf_b, poses_b, cfg):
    """AMB3R coordinate alignment"""
    F, N, _ = pts_a.shape
    if F < 1:
        return None, {'reason': 'no shared frames'}

    ref = F // 2
    Ra, ta = poses_a[ref, :3, :3], poses_a[ref, :3, 3]
    Rb, tb = poses_b[ref, :3, :3], poses_b[ref, :3, 3]
    qa = (pts_a.reshape(-1, 3) - ta) @ Ra
    qb = (pts_b.reshape(-1, 3) - tb) @ Rb

    ca, cb = conf_a.reshape(-1), conf_b.reshape(-1)
    thr = float(cfg['conf_rel']) * float(torch.median(torch.cat([ca, cb])))
    keep = ((ca > thr) & (cb > thr)
            & torch.isfinite(qa).all(-1) & torch.isfinite(qb).all(-1))
    n = int(keep.sum())
    if n < int(cfg['min_points']):
        return None, {'reason': 'too few confident points'}

    s = _ratio_scale(pts_a, poses_a, pts_b, poses_b, keep.view(F, -1),
                     trim=float(cfg.get('ratio_trim', 0.1)))
    if s is None:
        return None, {'reason': 'no depth ratios'}
    if bool(cfg.get('fix_scale', False)):
        s = 1.0

    pb_s = poses_b.clone()
    pb_s[..., :3, 3] *= s
    rel = poses_a @ torch.inverse(pb_s)
    w = (conf_a * conf_b).mean(dim=1)
    T = average_transforms(rel, w)
    R, t = T[:3, :3], T[:3, 3]
    g = to_sim3(torch.as_tensor(s, device=R.device, dtype=R.dtype), R, t)
    if not torch.isfinite(g).all():
        return None, {'reason': 'non-finite transform'}

    with torch.no_grad():
        pred = s * (pts_b.reshape(-1, 3)[keep] @ R.T) + t
        tgt = pts_a.reshape(-1, 3)[keep]
        res = (pred - tgt).norm(dim=-1)
        scale_ref = float(tgt.std(dim=0).norm().clamp_min(1e-8))
    return g, {'residual_rel': float(res.median()) / scale_ref}


def _ratio_scale(pts_a, poses_a, pts_b, poses_b, keep_f, trim=0.1):
    """Relative scale from per-pixel depth ratios on the shared frames, or None."""
    da = (pts_a - poses_a[:, None, :3, 3]).norm(dim=-1)
    db = (pts_b - poses_b[:, None, :3, 3]).norm(dim=-1)
    ok = keep_f & (da > 1e-6) & (db > 1e-6) & torch.isfinite(da) & torch.isfinite(db)
    if int(ok.sum()) < 64:
        return None
    r = torch.log(da[ok] / db[ok])
    if 0.0 < trim < 0.5:
        lo = torch.quantile(r, trim)
        hi = torch.quantile(r, 1.0 - trim)
        r = r[(r >= lo) & (r <= hi)]
        if r.numel() < 32:
            return None
    s = float(torch.exp(r.mean()))
    return s if np.isfinite(s) and s > 0 else None
