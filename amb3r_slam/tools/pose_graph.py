"""Sim(3) pose-graph optimisation."""

import numpy as np
import torch
from scipy.sparse import coo_matrix, identity as sp_identity
from scipy.sparse.linalg import spsolve

from amb3r_slam.tools.geometry import sim3_inv, sim3_mul

_EPS = 1e-7


def _hat(w):
    """(..., 3) -> (..., 3, 3) skew-symmetric."""
    O = torch.zeros(*w.shape[:-1], 3, 3, dtype=w.dtype, device=w.device)
    O[..., 0, 1], O[..., 0, 2] = -w[..., 2], w[..., 1]
    O[..., 1, 0], O[..., 1, 2] = w[..., 2], -w[..., 0]
    O[..., 2, 0], O[..., 2, 1] = -w[..., 1], w[..., 0]
    return O


def _sim3_W(phi, sigma):
    """The translation coupling matrix W of Sim(3) exp, matching lietorch/Sophus."""
    theta_sq = (phi ** 2).sum(-1)
    theta = theta_sq.sqrt()
    scale = torch.exp(sigma)

    small_s = sigma.abs() < _EPS
    small_t = theta < _EPS

    C0 = torch.ones_like(sigma)
    A0 = torch.where(small_t, torch.full_like(theta, 0.5),
                     (1 - torch.cos(theta)) / theta_sq.clamp_min(_EPS))
    B0 = torch.where(small_t, torch.full_like(theta, 1.0 / 6.0),
                     (theta - torch.sin(theta)) / (theta_sq * theta).clamp_min(_EPS))

    sig = torch.where(small_s, torch.full_like(sigma, _EPS), sigma)
    C1 = (scale - 1) / sig
    sig_sq = sig ** 2
    A1_small = ((sig - 1) * scale + 1) / sig_sq
    B1_small = (scale * 0.5 * sig_sq + scale - 1 - sig * scale) / (sig_sq * sig)
    a = scale * torch.sin(theta)
    b = scale * torch.cos(theta)
    c = theta_sq + sig_sq
    A1_big = (a * sig + (1 - b) * theta) / (theta.clamp_min(_EPS) * c)
    B1_big = (C1 - ((b - 1) * sig + a * theta) / c) / theta_sq.clamp_min(_EPS)
    A1 = torch.where(small_t, A1_small, A1_big)
    B1 = torch.where(small_t, B1_small, B1_big)

    A = torch.where(small_s, A0, A1)
    B = torch.where(small_s, B0, B1)
    C = torch.where(small_s, C0, C1)

    Phi = _hat(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand(*phi.shape[:-1], 3, 3)
    return C[..., None, None] * I + A[..., None, None] * Phi + B[..., None, None] * (Phi @ Phi)


def sim3_exp(xi):
    """(..., 7) [nu(3), phi(3), sigma] -> (..., 8) Sim3 data."""
    nu, phi, sigma = xi[..., :3], xi[..., 3:6], xi[..., 6]
    theta = (phi ** 2).sum(-1).sqrt()
    half = 0.5 * theta
    k = torch.where(theta < _EPS,
                    0.5 - theta ** 2 / 48.0,
                    torch.sin(half) / theta.clamp_min(_EPS))
    q = torch.cat([phi * k[..., None], torch.cos(half)[..., None]], dim=-1)
    W = _sim3_W(phi, sigma)
    t = torch.einsum('...ij,...j->...i', W, nu)
    return torch.cat([t, q, torch.exp(sigma)[..., None]], dim=-1)


def sim3_log(g):
    """(..., 8) Sim3 data -> (..., 7) [nu(3), phi(3), sigma]."""
    t, q, s = g[..., :3], g[..., 3:7], g[..., 7]
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[..., 3:4] < 0, -q, q)
    vec, w = q[..., :3], q[..., 3].clamp(-1.0, 1.0)
    n = vec.norm(dim=-1)
    theta = 2.0 * torch.atan2(n, w)
    factor = torch.where(n < _EPS, 2.0 / w.clamp_min(_EPS), theta / n.clamp_min(_EPS))
    phi = vec * factor[..., None]
    sigma = torch.log(s.clamp_min(1e-12))
    W = _sim3_W(phi, sigma)
    nu = torch.linalg.solve(W, t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([nu, phi, sigma[..., None]], dim=-1)


class Sim3PoseGraph:
    """Levenberg-Marquardt over Sim(3) nodes with relative-pose edges."""

    def __init__(self, poses, device='cpu', rot_weight=1.0, scale_weight=1.0,
                 huber=None, per_edge_trans=False, per_edge_floor=0.25):
        """``poses``: (N, 8) Sim3 world<-camera."""
        self.T = poses.to(device, torch.float64).clone()
        self.device = device
        self.ii, self.jj, self.Z, self.w = [], [], [], []
        self.rot_weight = float(rot_weight)
        self.scale_weight = float(scale_weight)
        self.huber = huber
        self.per_edge_trans = bool(per_edge_trans)
        self.per_edge_floor = float(per_edge_floor)
        self.sw = []
        self.rw = []
        self.cauchy = []

    def _dim_weights(self):
        """(E, 7) whitening, one row per edge."""
        Z = torch.stack(self.Z)
        tn = Z[:, :3].norm(dim=-1)
        med = float(tn.median())
        if self.per_edge_trans:
            floor = max(self.per_edge_floor * med, 1e-9)
            ts = tn.clamp_min(floor).to(torch.float64)
        else:
            ts = torch.full_like(tn, max(med, 1e-9)).to(torch.float64)
        dw = torch.empty(len(self.ii), 7, dtype=torch.float64, device=self.device)
        dw[:, :3] = (1.0 / ts)[:, None]
        dw[:, 3:6] = self.rot_weight
        dw[:, 6] = self.scale_weight
        return dw

    def add_edge(self, i, j, Z_ij, weight=1.0, scale_w=1.0, cauchy=False, rot_w=1.0):
        """Add a relative-pose constraint."""
        self.ii.append(int(i))
        self.jj.append(int(j))
        self.Z.append(torch.as_tensor(Z_ij, dtype=torch.float64, device=self.device))
        self.w.append(float(weight))
        self.sw.append(float(scale_w))
        self.rw.append(float(rot_w))
        self.cauchy.append(bool(cauchy))

    def _residuals(self, T):
        ii = torch.as_tensor(self.ii, device=self.device)
        jj = torch.as_tensor(self.jj, device=self.device)
        Z = torch.stack(self.Z)
        rel = sim3_mul(sim3_inv(T[ii]), T[jj])
        return sim3_log(sim3_mul(sim3_inv(Z), rel))

    def optimize(self, iters=30, fix=(0,), lm_init=1e-4, tol=1e-9):
        n = self.T.shape[0]
        E = len(self.ii)
        if E == 0 or n < 2:
            return self.T
        ii = np.asarray(self.ii)
        jj = np.asarray(self.jj)
        wts = torch.as_tensor(self.w, device=self.device, dtype=torch.float64)[:, None]

        free = np.ones(n, dtype=bool)
        for f in fix:
            free[f] = False
        col = -np.ones(n, dtype=np.int64)
        col[free] = np.arange(free.sum())
        n_free = int(free.sum())
        if n_free == 0:
            return self.T

        dw = self._dim_weights().clone()
        if len(self.sw) == E:
            sw_e = torch.as_tensor(self.sw, device=self.device, dtype=torch.float64)
            dw[:, 6] = dw[:, 6] * sw_e
        if len(self.rw) == E:
            rw_e = torch.as_tensor(self.rw, device=self.device, dtype=torch.float64)
            dw[:, 3:6] = dw[:, 3:6] * rw_e[:, None]
        wts_eff = wts
        if bool(getattr(self, 'weight_in_robust', False)) and self.huber is not None:
            dw = dw * torch.sqrt(wts.clamp_min(0.0))
            wts_eff = torch.ones_like(wts)
        def wht(resid):
            """Diagonal block weights."""
            return resid * dw

        is_cauchy = (torch.as_tensor(self.cauchy, device=self.device)
                     if len(self.cauchy) == E
                     else torch.zeros(E, dtype=torch.bool, device=self.device))

        def robust(rs):
            """Per-edge weight: base weight times a robust factor on the scaled norm."""
            if self.huber is None:
                return wts_eff
            nrm = rs.norm(dim=-1, keepdim=True)
            k = float(self.huber)
            hub = torch.where(nrm <= k, torch.ones_like(nrm), k / nrm.clamp_min(1e-12))
            cau = 1.0 / (1.0 + (nrm / k) ** 2)
            return wts_eff * torch.where(is_cauchy[:, None], cau, hub)

        lam = lm_init
        r = wht(self._residuals(self.T))
        cost = float((robust(r) * r ** 2).sum())

        for it in range(iters):
            J_i = torch.zeros(E, 7, 7, dtype=torch.float64, device=self.device)
            J_j = torch.zeros(E, 7, 7, dtype=torch.float64, device=self.device)
            eps = 1e-6
            basis = torch.eye(7, dtype=torch.float64, device=self.device) * eps
            ii_t = torch.as_tensor(ii, device=self.device)
            jj_t = torch.as_tensor(jj, device=self.device)
            Zs = torch.stack(self.Z)
            Zi = sim3_inv(Zs)
            for d in range(7):
                pert = sim3_exp(basis[d].expand(E, 7))
                rel_i = sim3_mul(sim3_inv(sim3_mul(self.T[ii_t], pert)), self.T[jj_t])
                J_i[:, :, d] = (wht(sim3_log(sim3_mul(Zi, rel_i))) - r) / eps
                rel_j = sim3_mul(sim3_inv(self.T[ii_t]), sim3_mul(self.T[jj_t], pert))
                J_j[:, :, d] = (wht(sim3_log(sim3_mul(Zi, rel_j))) - r) / eps

            ew = robust(r)
            sw = ew[:, :, None] ** 0.5
            Jw_i, Jw_j, rw = J_i * sw, J_j * sw, r * ew ** 0.5

            rows, cols, vals = [], [], []
            b = np.zeros(n_free * 7)
            Ji_np, Jj_np, r_np = Jw_i.cpu().numpy(), Jw_j.cpu().numpy(), rw.cpu().numpy()
            for (idx, Jn) in ((ii, Ji_np), (jj, Jj_np)):
                keep = free[idx]
                if not keep.any():
                    continue
                base = col[idx[keep]] * 7
                Jk = Jn[keep]
                rk = r_np[keep]
                e = Jk.shape[0]
                rr = (base[:, None] + np.arange(7)[None, :])
                rows.append(np.repeat(rr, 7, axis=1).ravel())
                cols.append(np.tile(rr, (1, 7)).ravel())
                vals.append(np.einsum('eki,ekj->eij', Jk, Jk).reshape(e, 49).ravel())
                np.add.at(b, rr.ravel(), -np.einsum('eki,ek->ei', Jk, rk).ravel())
            both = free[ii] & free[jj]
            if both.any():
                bi = col[ii[both]] * 7
                bj = col[jj[both]] * 7
                Ji_b, Jj_b = Ji_np[both], Jj_np[both]
                e = Ji_b.shape[0]
                ri = (bi[:, None] + np.arange(7)[None, :])
                rj = (bj[:, None] + np.arange(7)[None, :])
                Hij = np.einsum('eki,ekj->eij', Ji_b, Jj_b)
                rows.append(np.repeat(ri, 7, axis=1).ravel())
                cols.append(np.tile(rj, (1, 7)).ravel())
                vals.append(Hij.reshape(e, 49).ravel())
                rows.append(np.repeat(rj, 7, axis=1).ravel())
                cols.append(np.tile(ri, (1, 7)).ravel())
                vals.append(np.transpose(Hij, (0, 2, 1)).reshape(e, 49).ravel())


            H = coo_matrix((np.concatenate(vals),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n_free * 7, n_free * 7)).tocsr()

            accepted = False
            for _ in range(8):
                try:
                    dx = spsolve((H + lam * sp_identity(n_free * 7, format='csr')).tocsc(), b)
                except Exception:
                    dx = None
                if dx is None or not np.all(np.isfinite(dx)):
                    lam *= 10
                    continue
                step = torch.zeros(n, 7, dtype=torch.float64, device=self.device)
                step[torch.as_tensor(np.nonzero(free)[0])] = torch.from_numpy(
                    dx.reshape(n_free, 7)).to(self.device)
                T_new = sim3_mul(self.T, sim3_exp(step))
                r_new = wht(self._residuals(T_new))
                cost_new = float((robust(r_new) * r_new ** 2).sum())
                if np.isfinite(cost_new) and cost_new < cost:
                    self.T, r, cost = T_new, r_new, cost_new
                    lam = max(lam * 0.3, 1e-10)
                    accepted = True
                    break
                lam *= 10
            if not accepted or cost < tol:
                break

        return self.T
