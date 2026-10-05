"""Camera front-end: a pose for every frame as it arrives, from a few-view forward pass."""

import numpy as np
import torch

from amb3r_slam.model import SLAMModel
from amb3r_slam.tools.geometry import (average_transforms, sim3_from_matrix, sim3_inv,
                                      sim3_mul, sim3_to_se3_matrix)


def _umeyama_from_cameras(P_local, P_world, weights=None, s_prior=None, min_sep=1e-6):
    """Sim(3) taking a forward's local frame into the world, from cameras known in both."""
    F = P_local.shape[0]
    if F < 2:
        return None
    cw, cl = P_world[:, :3, 3], P_local[:, :3, 3]
    iu = torch.triu_indices(F, F, offset=1, device=cw.device)
    dw = (cw[iu[0]] - cw[iu[1]]).norm(dim=-1)
    dl = (cl[iu[0]] - cl[iu[1]]).norm(dim=-1)
    ok = (dl > min_sep) & (dw > min_sep)
    s = float(torch.median(dw[ok] / dl[ok])) if int(ok.sum()) else float('nan')
    if not np.isfinite(s) or s <= 0:
        if s_prior is None or not np.isfinite(s_prior) or s_prior <= 0:
            return None
        s = float(s_prior)

    Pl = P_local.clone()
    Pl[:, :3, 3] *= s
    rel = P_world @ torch.inverse(Pl)
    w = torch.ones(F, device=P_local.device) if weights is None else weights
    T = average_transforms(rel, w)
    g = sim3_from_matrix(T.double(), scale=1.0).float()
    g[7] = s
    return g


class FrontEnd:
    """Tracks frames against a keyframe and promotes keyframes by measured overlap."""

    def __init__(self, model, cfg, calib_K=None):
        self.net = SLAMModel(model, cfg.device, cfg.backend.pixel_target, patch=14)
        self.cfg = cfg
        self.device = cfg.device
        self.calib_K = calib_K
        self._last_K = None
        self.reset()

    def reset(self):
        self.kf_idx = None
        self.kf_pose = None
        self.recent = []
        self.poses = {}
        self.scale = None
        self.last_conf = 0.0


    @torch.no_grad()
    def _forward(self, images, ref):
        poses, pts, conf, self._last_K, _ = self.net(images, ref, self.calib_K)
        pts, conf = self.net.subsample(pts, conf)
        return poses, pts, conf

    def reanchor(self, frames, P):
        """Replace the poses of ``frames`` with the backend's, so tracking restarts from them."""
        for i, fr in enumerate(frames):
            self.poses[fr] = P[i]
        if self.kf_idx in self.poses:
            self.kf_pose = self.poses[self.kf_idx]
        self.recent = [(i, self.poses[i]) for i, _ in self.recent if i in self.poses]

    def transform(self, D):
        """Apply a Sim(3) correction ``D`` of the world to every stored pose."""
        if not self.poses:
            return
        keys = list(self.poses)
        P = torch.stack([self.poses[k] for k in keys]).to(self.device)
        P = sim3_to_se3_matrix(sim3_mul(D[None].expand(len(keys), 8),
                                        sim3_from_matrix(P.double()).float()))
        self.reanchor(keys, P)
        if self.scale is not None:
            self.scale = float(self.scale) * float(D[7])

    @torch.no_grad()
    def track(self, imgs, f):
        """Track frame ``f``; returns its world pose (4, 4) or None before the anchor."""
        fcfg = self.cfg.frontend
        n_ctx = max(0, int(fcfg.context))

        if self.kf_idx is None:
            self.kf_idx = f
            self.kf_pose = torch.eye(4, device=self.device)
            self.poses[f] = self.kf_pose
            return self.kf_pose

        ctx = [i for i, _ in self.recent[-n_ctx:]]
        views = sorted(dict.fromkeys([self.kf_idx] + ctx + [f]))
        poses_l, pts, conf = self._forward(imgs[views], ref=fcfg.ref)

        known = [(k, v) for k, v in enumerate(views) if v in self.poses]
        j = views.index(f)
        k = views.index(self.kf_idx)
        if len(known) >= 2:
            rows = torch.tensor([k0 for k0, _ in known], device=poses_l.device)
            P_world = torch.stack([self.poses[v] for _, v in known]).to(poses_l.device)
            P_local = poses_l[rows].float()
            g = _umeyama_from_cameras(P_local, P_world.float(), s_prior=self.scale)
        else:
            k0 = known[0][0]
            g0 = sim3_from_matrix(self.poses[views[k0]].double(), scale=1.0).float()
            g = sim3_mul(g0, sim3_inv(sim3_from_matrix(poses_l[k0].double(),
                                                       scale=1.0).float()))
        if g is None or not torch.isfinite(g).all():
            return self.poses.get(f)

        self.scale = float(g[7])
        gl = sim3_from_matrix(poses_l[j].double(), scale=1.0).float()
        pose = sim3_to_se3_matrix(sim3_mul(g, gl))
        if not torch.isfinite(pose).all():
            return self.poses.get(f)
        self.poses[f] = pose

        ov = self._overlap(pts, poses_l, k, j, imgs.shape[-2:])
        self.last_conf = float(conf[j].float().mean())
        if ov < float(fcfg.min_overlap) or \
                (f - self.kf_idx) >= int(fcfg.max_gap):
            self.kf_idx, self.kf_pose = f, pose
        self.recent.append((f, pose))
        self.recent = self.recent[-max(1, n_ctx):]
        return pose

    def _overlap(self, pts, poses_l, k, j, hw):
        """Fraction of the keyframe's points that still land inside the new frame."""
        K = self._last_K
        if K is None:
            return 1.0
        H, W = hw
        R, t = poses_l[j][:3, :3], poses_l[j][:3, 3]
        X = (pts[k].reshape(-1, 3) - t) @ R
        z = X[:, 2]
        Kj = K[j] if K.dim() == 3 else K
        uv = X @ Kj.T
        u, v = uv[:, 0] / z.clamp_min(1e-6), uv[:, 1] / z.clamp_min(1e-6)
        ok = (z > 1e-6) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        return float(ok.float().mean())

