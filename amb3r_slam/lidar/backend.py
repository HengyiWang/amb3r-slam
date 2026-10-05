from types import SimpleNamespace

import numpy as np
import torch

from amb3r_slam import ROOT
from amb3r_slam.backend import BackendResult, HierarchicalBackend
from amb3r_slam.lidar.registration import se3_inv
from amb3r_slam.tools.align import cloud_overlap
from amb3r_slam.tools.geometry import sim3_from_matrix, sim3_inv, sim3_mul, sim3_to_se3_matrix
from amb3r_slam.tools.loop import build_detector
from amb3r_slam.tools.pose_graph import Sim3PoseGraph

def _window_scale(P_win, P_met, idx):
    """Metric scale of a monocular window, from displacements both it and the LiDAR observe."""
    r = []
    for a in range(len(idx)):
        for b in range(a + 1, len(idx)):
            dv = np.linalg.norm(P_win[idx[a], :3, 3] - P_win[idx[b], :3, 3])
            dm = np.linalg.norm(P_met[idx[a], :3, 3] - P_met[idx[b], :3, 3])
            if dv > 1e-6 and dm > 1e-3:
                r.append(dm / dv)
    return float(np.median(r)) if len(r) >= 3 else None

def _open3d_device(device):
    kind, _, idx = str(device).partition(':')
    return f"{'CUDA' if kind == 'cuda' else 'CPU'}:{idx or 0}"


class LidarBackend(HierarchicalBackend):
    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.lidar_reg = None

    def reset(self, calib_K=None):
        super().reset(calib_K)
        self.lidar_reg = None

    def begin(self, imgs, fe, verbose=False):
        """Start a sequence: imgs (T, 3, H, W) in [-1, 1], odometry from ``fe``."""
        if self.lidar_reg is None:
            raise RuntimeError('LidarBackend needs LiDAR scans, and none are attached')
        lcfg = self.cfg.backend.loop
        n = int(imgs.shape[0])
        self.stats = {'n_loops': 0}
        self._ls = st = SimpleNamespace()
        st.imgs, st.n, st.fe, st.verbose = imgs, n, fe, verbose
        st.T_cv = np.asarray(self.lidar_reg['T_cv'], np.float64)
        st.step = int(self.cfg.backend.lidar.keyframe_every)
        st.half = max(3, int(self.cfg.backend.loop_half_window))
        st.k_anchor = int(lcfg.anchor_search)
        st.map_win = int(lcfg.lidar_map_win)
        st.Pc = np.tile(np.eye(4), (n, 1, 1))
        st.fe_poses = np.tile(np.eye(4), (n, 1, 1))
        st.nf, st.S0, st.S, st.odo = [], [], [], []
        st.accepted, st.pending = [], []
        st.corrected = False
        st.det = build_detector(self.cfg.retrieval, ROOT, self.device)
        st.stride = int(self.cfg.retrieval.probe_every)
        st.min_gap = max(1, int(self.cfg.retrieval.min_frame_gap) // st.stride)
        st.n_probe = 0

    def _add_node(self, f):
        st = self._ls
        g0 = sim3_from_matrix(torch.from_numpy(st.Pc[f]).float(), scale=torch.tensor(1.0))
        if st.nf:
            G = se3_inv(st.Pc[st.nf[-1]]) @ st.Pc[f]
            st.odo.append(sim3_from_matrix(torch.from_numpy(G).float(),
                                           scale=torch.tensor(1.0)).double())
        if st.corrected:
            st.S.append(sim3_mul(st.S[-1].float(), sim3_mul(sim3_inv(st.S0[-1].float()), g0)))
        else:
            st.S.append(g0)
        st.nf.append(f)
        st.S0.append(g0)

    def _ready(self, fj, f):
        """Everything the verification of a loop at ``fj`` reads has arrived by ``f``."""
        st, n = self._ls, self._ls.n
        nxt = (fj // st.step + 1) * st.step
        if nxt >= n - 1 or f < max(nxt, fj + (st.k_anchor + 1) * st.half):
            return False
        s_need = st.fe.scan_of(min(n - 1, fj + st.k_anchor * st.half))
        s_now = st.fe.scan_of(f)
        return s_need is None or s_now is None or \
            s_now >= min(len(st.fe.scans) - 1, s_need + st.map_win)

    def _verify(self, fj, fi):
        st = self._ls
        return self._lidar_loop(st.imgs, fj, fi, st.n, st.nf, st.Pc, st.half, st.k_anchor,
                                st.fe.scans, st.fe.odometry, st.T_cv, st.fe.T_vc, st.fe.f2s,
                                st.accepted)

    def add_frame(self, f, pose, conf=None):
        """Frame ``f`` with its odometry pose: keyframe every `keyframe_every` frames,
        retrieval per probe, and the loops whose windows have now arrived."""
        st = self._ls
        st.Pc[f] = pose
        if f % st.step == 0:
            self._add_node(f)
        j = len(st.nf) - 1
        st.fe_poses[f] = (sim3_to_se3_matrix(st.S[j].double()).numpy()
                          @ se3_inv(st.Pc[st.nf[j]]) @ st.Pc[f])
        if st.det is not None and f % st.stride == 0:
            hit = st.det.detect(st.imgs[f], st.min_gap)
            if hit is not None and hit < st.n_probe:
                st.pending.append((f, hit * st.stride, st.n_probe, hit))
                st.det.propose(st.n_probe, hit)
            st.n_probe += 1
        online_pgo = str(self.cfg.backend.loop.pgo_mode) == 'online'
        while st.pending and self._ready(st.pending[0][0], f):
            fj, fi, k, hit = st.pending.pop(0)
            ok = self._verify(fj, fi)
            st.det.resolve(k, hit, ok)
            if ok and online_pgo:
                st.S[:] = list(self._lidar_pgo(st.S, st.odo, st.accepted, len(st.nf)).float())
                st.corrected = True

    def finish(self):
        """The final keyframe, the remaining loops, the final pose graph, and a pose per
        frame riding the odometry from its keyframe."""
        st, stats, n = self._ls, self.stats, self._ls.n
        if st.nf[-1] != n - 1:
            self._add_node(n - 1)
        for fj, fi, _k, _hit in st.pending:
            self._verify(fj, fi)
        st.det = None
        stats['n_loops'] = len(st.accepted)
        if st.verbose:
            print(f'  [lidar] {len(st.nf)} keyframes, {len(st.accepted)} loops', flush=True)
        online_pgo = str(self.cfg.backend.loop.pgo_mode) == 'online'
        Sf = self._lidar_pgo(st.S if online_pgo else st.S0, st.odo, st.accepted,
                             len(st.nf)).float()
        Sm = sim3_to_se3_matrix(Sf.double()).numpy()
        out = np.tile(np.eye(4), (n, 1, 1))
        j = 0
        for f in range(n):
            while j + 1 < len(st.nf) and st.nf[j + 1] <= f:
                j += 1
            out[f] = Sm[j] @ (se3_inv(st.Pc[st.nf[j]]) @ st.Pc[f])
        return BackendResult(torch.from_numpy(out).float(), st.nf, stats,
                             frontend_poses=torch.from_numpy(st.fe_poses).float())

    def _lidar_pgo(self, init, odo, accepted, n_nodes):
        """Sim(3) graph over the keyframes: odometry edges, then the loop edges."""
        lcfg = self.cfg.backend.loop
        lw = self.cfg.backend.lidar
        pgo = Sim3PoseGraph(torch.stack([s.double() for s in init[:n_nodes]]),
                            huber=float(lcfg.pgo_huber),
                            per_edge_trans=bool(lcfg.per_edge_trans),
                            per_edge_floor=float(lcfg.per_edge_floor))
        pgo.weight_in_robust = bool(lcfg.weight_in_robust)
        for a, g in enumerate(odo[:n_nodes - 1]):
            pgo.add_edge(a, a + 1, g,
                         weight=float(lw.weight),
                         scale_w=float(lw.scale_w),
                         rot_w=float(lw.rot_w),
                         cauchy=bool(lw.cauchy))
        rs = [r for (_, _, _, r) in accepted if r > 0]
        med = float(np.median(rs)) if len(rs) >= 4 else None
        as_odo = bool(lcfg.lidar_edge_weights)
        for (ka, kb, G, rmse) in accepted:
            w = float(lw.weight) if as_odo else float(lcfg.weight)
            sw = float(lw.scale_w) if as_odo else 1.0
            rw = float(lw.rot_w) if as_odo else 1.0
            if med and rmse > 0 and bool(lcfg.lidar_weight_by_rmse):
                w *= float(np.clip((med / rmse) ** 2, 1.0 / 16, 16.0))
            pgo.add_edge(ka, kb, sim3_from_matrix(torch.from_numpy(G).float(),
                                                  scale=torch.tensor(1.0)).double(),
                         weight=w, scale_w=sw, rot_w=rw,
                         cauchy=bool(lcfg.cauchy))
        return pgo.optimize(iters=int(lcfg.pgo_iters))

    def _lidar_loop(self, imgs, fj, fi, n, nf, Pc, half, k_anchor, scans, odometry,
                    T_cv, T_vc, f2s, accepted):
        """Verify one retrieval candidate; an accepted one is appended to `accepted`."""
        lcfg = self.cfg.backend.loop
        wa = [f for f in range(fi - half, fi + half) if 0 <= f < n]
        wb = [f for f in range(fj - half, fj + half) if 0 <= f < n]
        if len(wa) < 3 or len(wb) < 3:
            return self._reject('window too short')

        def _ov(w_a, w_b):
            p, pt, cf = self._forward(imgs[w_a + w_b], ref=self._pair_ref(),
                                      frames=w_a + w_b)
            n_a = len(w_a)
            o = cloud_overlap(pt[:n_a].float(), cf[:n_a].float(),
                              pt[n_a:].float(), cf[n_a:].float(),
                              voxel_rel=float(lcfg.voxel_rel), conf_rel=float(lcfg.conf_rel))
            return o, p, pt, cf

        min_ov = float(lcfg.min_overlap)
        ov, poses_l, pts_l, conf_l = _ov(wa, wb)
        if ov < min_ov and k_anchor > 0:
            best = (ov, wa, wb, fi, fj, poses_l, pts_l, conf_l)
            offs = ([(s * d, 0) for d in range(1, k_anchor + 1) for s in (-1, 1)]
                    + [(0, s * d) for d in range(1, k_anchor + 1) for s in (-1, 1)])
            for da, db in offs:
                a_fi, a_fj = fi + da * half, fj + db * half
                if not (0 <= a_fi < n and 0 <= a_fj < n):
                    continue
                w_a = [f for f in range(a_fi - half, a_fi + half) if 0 <= f < n] if da else wa
                w_b = [f for f in range(a_fj - half, a_fj + half) if 0 <= f < n] if db else wb
                if len(w_a) < max(3, int(0.6 * len(wa))) or \
                        len(w_b) < max(3, int(0.6 * len(wb))):
                    continue
                if a_fi not in w_a or a_fj not in w_b:
                    continue
                o2, p2, pt2, cf2 = _ov(w_a, w_b)
                if o2 > best[0]:
                    best = (o2, w_a, w_b, a_fi, a_fj, p2, pt2, cf2)
                else:
                    del p2, pt2, cf2
                if best[0] >= min_ov:
                    break
            ov, wa, wb, fi, fj, poses_l, pts_l, conf_l = best
        na = len(wa)
        if ov < min_ov:
            del poses_l, pts_l, conf_l
            return self._reject('windows do not overlap')

        Pw = poses_l.double().cpu().numpy()
        del poses_l, pts_l, conf_l
        rows = wa + wb
        Pm = np.stack([Pc[f] for f in rows])
        s_a = _window_scale(Pw, Pm, list(range(na)))
        s_b = _window_scale(Pw, Pm, list(range(na, len(rows))))
        ss = [x for x in (s_a, s_b) if x is not None and np.isfinite(x) and x > 0]
        if not ss:
            return self._reject('window scale unobservable')
        s = float(np.median(ss))
        ia, jb = wa.index(fi), na + wb.index(fj)
        Ta, Tb = Pw[ia].copy(), Pw[jb].copy()
        Ta[:3, 3] *= s
        Tb[:3, 3] *= s
        G_cam = se3_inv(Ta) @ Tb

        ka = int(np.argmin([abs(fi - x) for x in nf]))
        kb = int(np.argmin([abs(fj - x) for x in nf]))
        if ka == kb:
            return self._reject('same node')
        G = (se3_inv(Pc[nf[ka]]) @ Pc[fi]) @ G_cam @ (se3_inv(Pc[fj]) @ Pc[nf[kb]])

        rmse = 0.0
        if lcfg.lidar_refine:
            G2, rmse, why, fit = self._refine(scans, odometry, fi, fj, G, T_cv, T_vc, f2s,
                                              lcfg, Pc, nf, ka, kb)
            vf = float(lcfg.lidar_verify_fitness)
            if vf > 0 and fit < vf:
                return self._reject('registration will not verify'
                                    + (f' [{why}]' if G2 is None and why else ''))
            if G2 is not None:
                G = G2
        accepted.append((ka, kb, G, rmse))
        return True

    def _refine(self, scans, odometry, fi, fj, G, T_cv, T_vc, f2s, lcfg, Pc, nf, ka, kb):
        """Register the two places, seeded by the window's metric estimate."""
        from amb3r_slam.lidar.registration import register_places
        sa = f2s[fi] if f2s is not None else fi
        sb = f2s[fj] if f2s is not None else fj
        if sa is None or sb is None or max(int(sa), int(sb)) >= len(scans):
            return None, 0.0, 'frames outside the scan range', 0.0
        G_f = (se3_inv(se3_inv(Pc[nf[ka]]) @ Pc[fi]) @ G
               @ se3_inv(se3_inv(Pc[fj]) @ Pc[nf[kb]]))
        T_init = T_vc @ G_f @ T_cv
        T, fit, info = register_places(
            scans, int(sa), int(sb), T_init, odometry.relative,
            win=int(lcfg.lidar_map_win),
            voxel=float(self.cfg.backend.lidar.voxel),
            max_corr=float(lcfg.lidar_max_corr),
            iters=int(lcfg.lidar_iters),
            backend='cuda' if str(self.cfg.device).startswith('cuda') else 'cpu',
            device=_open3d_device(self.cfg.device),
            radius=float(lcfg.lidar_radius))
        if T is None:
            return None, 0.0, str(info.get('reason', 'registration failed')), 0.0
        if fit < float(lcfg.lidar_min_fitness):
            return None, 0.0, 'registration refused', float(fit)
        bound = float(lcfg.lidar_max_correction)
        if bound > 0 and float(info.get('d_trans', 0.0)) > bound:
            return None, 0.0, 'correction exceeds bound', float(fit)
        G_f2 = T_cv @ T @ T_vc
        G2 = (se3_inv(Pc[nf[ka]]) @ Pc[fi]) @ G_f2 @ (se3_inv(Pc[fj]) @ Pc[nf[kb]])
        return G2, float(info.get('inlier_rmse', 0.0) or 0.0), None, float(fit)
