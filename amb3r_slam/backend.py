"""Hierarchical backend (paper section 3.2), monocular.

Dense local mapping, sparse long-context mapping, loop closure and a Sim(3) pose graph
over the submaps.
"""

import gc
import math

import numpy as np
import torch
from omegaconf import OmegaConf

from amb3r_slam import ROOT
from amb3r_slam.model import SLAMModel
from amb3r_slam.tools.align import align_overlap, align_overlap_poses, cloud_overlap
from amb3r_slam.tools.geometry import quat_to_rmat, sim3_from_matrix, sim3_interp, sim3_inv, sim3_mul, sim3_to_se3_matrix
from amb3r_slam.tools.loop import build_detector
from amb3r_slam.tools.pose_graph import Sim3PoseGraph


def sim3_identity(device):
    g = torch.zeros(8, device=device)
    g[6], g[7] = 1.0, 1.0
    return g


class Submap:

    __slots__ = ('start', 'end', 'poses', 'pts', 'conf', 'frames', '_pos',
                 'metric_scale', 'metric', 'dense')

    def __init__(self, start, end, poses, pts, conf, frames=None, metric_scale=None,
                 metric=False):
        self.metric = metric
        self.dense = None                     # full-resolution pass, held for map export
        self.start, self.end = start, end
        self.metric_scale = metric_scale
        self.poses = poses
        self.pts = pts
        self.conf = conf
        self.frames = list(range(start, end)) if frames is None else list(frames)
        self._pos = {f: i for i, f in enumerate(self.frames)}

    def __len__(self):
        return len(self.frames)

    def covers(self, f):
        return f in self._pos

    def shared(self, other):
        """Frames both submaps reconstructed, in order."""
        return [f for f in self.frames if f in other._pos]

    def local(self, frames):
        """Rows for the given global frame indices."""
        idx = torch.as_tensor([self._pos[f] for f in frames], device=self.pts.device)
        return self.pts[idx], self.conf[idx], self.poses[idx]


class BackendResult:
    """Per-frame poses, keyframe indices and statistics; ``frontend_poses`` is the online
    trajectory, ``submap_poses`` the (K, 8) world-from-submap Sim(3) of each submap."""

    def __init__(self, poses, keyframes, stats, submaps=(), frontend_poses=None,
                 submap_poses=None):
        self.poses = poses
        self.keyframes = [int(f) for f in keyframes]
        self.stats = stats
        self.submaps = list(submaps)
        self.frontend_poses = frontend_poses
        self.submap_poses = submap_poses


class _Stream:
    """Per-sequence backend state, filled as the stream arrives."""

    def __init__(self, fe):
        self.fe = fe
        self.submaps = []
        self.S_odo = []          # chain of local alignments; the drift gates read this
        self.S = []              # current estimate; differs from S_odo after online PGO
        self.adjacent = []       # g for submap k -> k + 1, or None
        self.span = []           # (ka, kb, g, info) between overlapping submaps
        self.long_context = []   # (ka, kb, Z, info) from long-context windows
        self.loops = []          # (ka, kb, Z)
        self.lc = None
        self.loop = None


class HierarchicalBackend:
    """Monocular: every submap lives in its own arbitrary units, joined by Sim(3) edges."""

    def __init__(self, model, cfg):
        self.net = SLAMModel(model, cfg.device, cfg.backend.pixel_target)
        self.recorder = None                  # tools.export.MapRecorder, for --save_map
        self.cfg = cfg
        self.device = cfg.device
        self.stats = {}
        self.calib_K = None

    def reset(self, calib_K=None):
        """Per-sequence inputs: intrinsics (3, 3) or None. Subclasses clear sensors."""
        self.calib_K = calib_K

    @torch.no_grad()
    def _forward(self, images, ref=None, frames=None, dense=False):
        """images (F, 3, H, W) -> (poses, subsampled pts, conf); ``frames`` are the rows'
        global indices, used to look up a metric sensor. With ``dense``, also the
        full-resolution (pts, conf, sky mask or None)."""
        poses, pts, conf, _K, self._last_scale = self.net(
            images, ref or self.cfg.backend.ref_view_strategy, self.calib_K)
        self._last_metric = False
        if frames is not None:
            pts, poses = self._to_metric(pts, poses, conf, frames)
        full = (pts, conf, getattr(self.net.model, 'last_sky', None)) if dense else None
        pts, conf = self.net.subsample(pts, conf)
        return (poses, pts, conf, full) if dense else (poses, pts, conf)

    def _to_metric(self, pts, poses, conf, frames):
        return pts, poses

    def _densify(self, kf, n, d):
        """Insert frames so no two keyframes are more than `d` apart."""
        if not kf:
            return kf
        out = [kf[0]]
        for a, b in zip(kf[:-1], kf[1:]):
            if b - a > d:
                k = int(np.ceil((b - a) / d)) - 1
                for i in range(1, k + 1):
                    f = a + int(round(i * (b - a) / (k + 1)))
                    if out[-1] < f < b:
                        out.append(f)
            out.append(b)
        if n - 1 - out[-1] > d:
            out.extend(range(out[-1] + d, n, d))
        return sorted(dict.fromkeys(out))

    def _group_keyframes(self, n, d, score):
        """The best-scoring frame of each group of `d`, grouped by absolute index."""
        out = []
        for a in range(0, n, d):
            b = min(a + d, n)
            if b > a:
                out.append(a + int(np.argmax(score[a:b])))
        return out

    @staticmethod
    def _submap_of(frame, submaps):
        """Index of the submap whose span holds ``frame`` closest to its middle."""
        best, best_d = None, None
        for k, sm in enumerate(submaps):
            a, b = sm.start, sm.end
            if a <= frame < b:
                d = abs(frame - 0.5 * (a + b - 1))
                if best_d is None or d < best_d:
                    best, best_d = k, d
        return best

    def _align_pair(self, a, b):
        """Sim(3) taking submap ``b``'s frame into submap ``a``'s, from shared frames."""
        shared = a.shared(b)
        if len(shared) < int(self.cfg.backend.align.min_shared_frames):
            return None, {'reason': f'only {len(shared)} shared frames'}
        pa, ca, qa = a.local(shared)
        pb, cb, qb = b.local(shared)
        return self._align(pa, ca, qa, pb, cb, qb, fix_scale=self._scale_known(a, b))

    def _align(self, pa, ca, qa, pb, cb, qb, fix_scale=False):
        """Sim(3) from b's frame into a's over the same frames: scale from depth ratios,
        rotation and translation from the shared cameras, Procrustes as fallback."""
        cfg = self.cfg.backend.align
        if fix_scale:
            cfg = OmegaConf.merge(cfg, {'fix_scale': True})
        pa, ca, pb, cb = pa.float(), ca.float(), pb.float(), cb.float()
        g, info = align_overlap_poses(pa, ca, qa.float(), pb, cb, qb.float(), cfg)
        if g is not None:
            return g, info
        return align_overlap(pa, ca, pb, cb, cfg)

    def _scale_known(self, a, b):
        """Whether an edge may fix s = 1; never for monocular submaps."""
        return False

    def _pair_ref(self):
        """Reference view for a pass over two disjoint windows (a loop window)."""
        return str(self.cfg.backend.pair_ref_view_strategy)

    @torch.no_grad()
    def _long_context_begin(self):
        cfg = self.cfg.backend.long_context
        lc = {'next': 0, 'seen': set(), 'cvm': None}
        if float(cfg.min_covis) > 0:
            from amb3r_slam.tools.covis import AlikedCovis
            lc['cvm'] = AlikedCovis(dict(cfg.covis or {}), self.device)
        return lc

    def _long_context_step(self, images, final=False):
        """Sparse long-context mapping: fire every window whose submaps have all arrived.

        A window spans `nodes_per_window` submaps and starts every `every_nodes`; at the
        end of the sequence the shorter tail windows fire too.
        """
        cfg = self.cfg.backend.long_context
        st, lc = self._st, self._st.lc
        k_nodes = int(cfg.nodes_per_window)
        every = max(1, int(cfg.every_nodes))
        min_gap = int(self.cfg.backend.align.max_pair_span) + 1
        n = len(st.submaps)
        while lc['next'] < n:
            a0 = lc['next']
            a1 = min(a0 + k_nodes, n)
            if a1 - a0 < k_nodes and not final:
                return
            if a1 - a0 < min_gap + 1:
                lc['next'] = n
                return
            st.long_context += self._long_context_window(images, list(range(a0, a1)))
            lc['next'] += every

    def _long_context_window(self, images, grp):
        """Edges between the far-apart submaps of one window.
        """
        cfg = self.cfg.backend.long_context
        submaps, lc = self._st.submaps, self._st.lc
        seen, cvm = lc['seen'], lc['cvm']
        rows = int(self.cfg.backend.views)
        min_shared = int(self.cfg.backend.align.min_shared_frames)
        min_gap = int(self.cfg.backend.align.max_pair_span) + 1
        thr = float(cfg.max_residual_rel)
        min_covis = float(cfg.min_covis)
        min_conf_ratio = float(cfg.min_conf_ratio)
        covis_stat = str(cfg.covis_stat)
        strong_at = float(cfg.covis_strong_at)

        pool = sorted({f for a in grp for f in submaps[a].frames})
        if len(pool) < 2:
            return []
        if len(pool) <= rows:
            sel = pool
        else:
            idx = np.linspace(0, len(pool) - 1, rows).round().astype(int)
            sel = [pool[i] for i in sorted(set(int(i) for i in idx))]
        if len(sel) < min_shared + 1:
            return []

        if cvm is not None:
            cvm.evict_before(sel[0])
        poses_c, pts_c, conf_c = self._forward(images[sel], frames=sel)
        pos = {f: i for i, f in enumerate(sel)}

        fits, conf_pair = {}, {}
        for a in grp:
            sm = submaps[a]
            shared = [f for f in sel if sm.covers(f)]
            if len(shared) < min_shared:
                continue
            rw = torch.as_tensor([pos[f] for f in shared], device=pts_c.device)
            pa, ca, qa = sm.local(shared)
            conf_pair[a] = (float(conf_c[rw].float().median()),
                            float(ca.float().median()))
            g, info = self._align(pa, ca, qa, pts_c[rw], conf_c[rw], poses_c[rw])
            if g is not None and torch.isfinite(g).all() and \
                    float(info.get('residual_rel', 1e9)) <= thr:
                fits[a] = g
        del poses_c, pts_c, conf_c

        if min_conf_ratio > 0 and conf_pair:
            rr = [c[0] / max(c[1], 1e-9) for c in conf_pair.values()]
            if float(np.median(rr)) < min_conf_ratio:
                return []

        edges = []
        ks = sorted(fits)
        for x, i in enumerate(ks):
            for j in ks[x + 1:]:
                if j - i < min_gap:
                    continue
                if (i, j) in seen:
                    continue
                Z = sim3_mul(fits[i], sim3_inv(fits[j]))
                if not torch.isfinite(Z).all():
                    continue
                seen.add((i, j))
                if cvm is not None and min_covis > 0:
                    fi = [f for f in sel if submaps[i].covers(f)]
                    fj = [f for f in sel if submaps[j].covers(f)]
                    cvg = cvm.span_pair(images, fi, fj, strong_at=strong_at)
                    if float(cvg[covis_stat]) < min_covis:
                        continue
                edges.append((i, j, Z, {'long_context': True}))
        return edges

    def _add_submap(self, sm, images):
        """Dense local mapping: link a new submap to the chain by its adjacent and span
        edges, then fire any long-context window it completes."""
        st = self._st
        k = len(st.submaps)
        if k == 0:
            st.S_odo.append(sim3_identity(self.device))
            st.S.append(sim3_identity(self.device))
        else:
            g, _info = self._align_pair(st.submaps[-1], sm)
            st.adjacent.append(g)
            g = sim3_identity(self.device) if g is None else g.to(self.device)
            st.S_odo.append(sim3_mul(st.S_odo[-1], g))
            st.S.append(sim3_mul(st.S[-1], g))
        st.submaps.append(sm)
        for d in range(2, int(self.cfg.backend.align.max_pair_span) + 1):
            a = k - d
            if a < 0 or st.submaps[a].end <= sm.start:
                break
            g, info = self._align_pair(st.submaps[a], sm)
            if g is not None:
                st.span.append((a, k, g, info))
        self._long_context_step(images)

    def _edge_weight(self, info):
        """Pose-graph weight of a span or long-context edge."""
        if info.get('long_context'):
            return float(self.cfg.backend.long_context.weight)
        return float(self.cfg.backend.align.extra_weight)

    def _loop_constraint(self, images, submaps, ka, kb, fi, fj):
        """Loop edge ``Z_ab`` from a joint forward over windows at ``fi`` (old) and ``fj``.

        Accepted if both ends overlap in the joint reconstruction and each fits its own
        submap; otherwise the windows are shifted (anchor search). Returns
        ``(ka, kb, Z_ab or None, info)``, or None.
        """
        half = int(self.cfg.backend.loop_half_window)
        if ka == kb:
            return None
        ca, cb = submaps[ka], submaps[kb]

        wa = [f for f in range(fi - half, fi + half) if ca.covers(f)]
        wb = [f for f in range(fj - half, fj + half) if cb.covers(f)]
        if len(wa) < 3 or len(wb) < 3:
            return None

        min_ov = float(self.cfg.backend.loop.min_overlap)
        thr_res = float(self.cfg.backend.loop.max_residual_rel)

        def _eval(w_a, w_b):
            """Reconstruct both windows together and apply the overlap and residual gates."""
            p, pt, cf = self._forward(images[w_a + w_b], ref=self._pair_ref(),
                                      frames=w_a + w_b)
            n_a = len(w_a)
            o = cloud_overlap(pt[:n_a].float(), cf[:n_a].float(),
                              pt[n_a:].float(), cf[n_a:].float(),
                              voxel_rel=float(self.cfg.backend.loop.voxel_rel),
                              conf_rel=float(self.cfg.backend.loop.conf_rel))
            if o < min_ov:
                del p, pt, cf
                return {'ok': False, 'ov': o,
                        'reason': f'windows do not overlap ({o:.2f})'}
            pa, cca, qa = ca.local(w_a)
            pb, ccb, qb = cb.local(w_b)
            ga, ja = self._align(pa, cca, qa, pt[:n_a], cf[:n_a], p[:n_a])
            gb, jb = self._align(pb, ccb, qb, pt[n_a:], cf[n_a:], p[n_a:])
            del p, pt, cf
            if ga is None or gb is None:
                return {'ok': False, 'ov': o, 'reason': 'align failed', 'hard': True}
            if ja.get('residual_rel', 1e9) > thr_res or jb.get('residual_rel', 1e9) > thr_res:
                return {'ok': False, 'ov': o, 'a': ja, 'b': jb,
                        'reason': 'residual too large'}
            Z = sim3_mul(ga, sim3_inv(gb))
            ja['overlap'] = o
            return {'ok': True, 'ov': o, 'a': ja, 'b': jb, 'Z': Z, 'shift': (0, 0)}

        res = _eval(wa, wb)
        k_anchor = int(self.cfg.backend.loop.anchor_search)
        if not res['ok'] and k_anchor > 0 and not res.get('hard'):
            best, step = res, max(1, half)
            offs = ([(s * d, 0) for d in range(1, k_anchor + 1) for s in (-1, 1)]
                    + [(0, s * d) for d in range(1, k_anchor + 1) for s in (-1, 1)])
            for da, db in offs:
                w_a = ([f for f in range(fi + da * step - half, fi + da * step + half)
                        if ca.covers(f)] if da else wa)
                w_b = ([f for f in range(fj + db * step - half, fj + db * step + half)
                        if cb.covers(f)] if db else wb)
                if len(w_a) < max(3, int(0.6 * len(wa))) or \
                        len(w_b) < max(3, int(0.6 * len(wb))):
                    continue
                r2 = _eval(w_a, w_b)
                if r2['ok'] and (not best['ok'] or r2['ov'] > best['ov']):
                    r2 = {**r2, 'shift': (da, db)}
                    best, wa, wb = r2, w_a, w_b
                elif not best['ok'] and r2['ov'] > best['ov']:
                    best = r2
            res = best
        if not res['ok']:
            if res.get('hard'):
                return None
            out = {'overlap': res['ov'], 'reason': res['reason']}
            if 'a' in res:
                out['a'], out['b'] = res['a'], res['b']
            return ka, kb, None, out
        return ka, kb, res['Z'], {'a': res['a'], 'b': res['b']}

    @torch.no_grad()
    def _optimise(self, init):
        """Sim(3) pose graph over the submaps from ``init``, with every edge so far."""
        st = self._st
        S_t = torch.stack(list(init))
        extra = sorted(st.span, key=lambda e: (e[0], e[1])) + st.long_context
        if not (st.loops or extra):
            return S_t
        lc = self.cfg.backend.loop
        pgo = Sim3PoseGraph(
            S_t, huber=float(lc.pgo_huber),
            per_edge_trans=bool(lc.per_edge_trans),
            per_edge_floor=float(lc.per_edge_floor))
        pgo.weight_in_robust = bool(lc.weight_in_robust)

        for k, g in enumerate(st.adjacent):
            if g is not None:
                pgo.add_edge(k, k + 1, g.double().cpu(), weight=1.0)
        for ka, kb, g, info in extra:
            pgo.add_edge(ka, kb, g.double().cpu(), weight=self._edge_weight(info))
        for e in st.loops:
            pgo.add_edge(e[0], e[1], e[2].double().cpu(), weight=float(lc.weight),
                         cauchy=bool(lc.cauchy))
        return pgo.optimize(iters=int(lc.pgo_iters)).float().to(self.device)

    def _correct_online(self):
        """Online PGO: optimise now, and carry the correction into the front-end."""
        st = self._st
        S_new = list(self._optimise(st.S))
        st.fe.transform(sim3_mul(S_new[-1], sim3_inv(st.S[-1])))
        st.S = S_new

    def _result(self, poses, submaps, frontend_poses=None, submap_poses=None):
        return BackendResult(poses, [(sm.start + sm.end) // 2 for sm in submaps],
                             self.stats, submaps, frontend_poses, submap_poses)

    def begin(self, imgs, fe, verbose=False):
        """Start a sequence: imgs (T, 3, H, W) in [-1, 1], tracked by ``fe``."""
        n = imgs.shape[0]
        self.stats = {'n_loops': 0}
        st = self._st = _Stream(fe)
        st.imgs, st.n, st.verbose = imgs, n, verbose
        st.online = torch.eye(4).repeat(n, 1, 1)
        st.conf_f = np.zeros(n)
        st.lc = self._long_context_begin()
        st.det = self._loop_begin()
        size = int(self.cfg.backend.size)
        warm = max(0, int(self.cfg.backend.warmup_min))
        st.due = warm if 0 < warm < size else size
        st.warming = st.due < size

    @torch.no_grad()
    def add_frame(self, f, pose, conf):
        """Frame ``f`` has been tracked: record it, query retrieval, and every `step` frames
        build a submap, re-anchor the front-end and verify the loops that became ready.

        Before the first full window a short submap is emitted from `warmup_min` frames
        and grown, each one replacing the last.
        """
        st = self._st
        if pose is not None:
            st.online[f] = pose.detach().cpu()
            st.conf_f[f] = conf
        if st.det is not None:
            self._loop_query(st.det, st.imgs, f)
        if f + 1 < st.due:
            return
        size, step = int(self.cfg.backend.size), int(self.cfg.backend.step)
        a, b = max(0, f + 1 - size), f + 1
        st.due = min(st.due + step, size) if st.warming else st.due + step
        own = self._submap_frames(st.conf_f, a, b)
        if len(own) < max(4, int(self.cfg.backend.align.min_shared_frames)):
            return
        sm = self._new_submap(a, b, own)
        if st.warming and st.submaps:
            self._record(sm, replaces=st.submaps[-1])
            st.submaps[-1] = sm
        else:
            self._add_submap(sm, st.imgs)
            self._record(sm)
        if b >= size:
            st.warming = False
        self._reanchor(sm)
        self._verify_ready()
        if st.verbose and len(st.submaps) % 20 == 1:
            print(f"  [stream] frame {f + 1}/{st.n}, submap {len(st.submaps)} over "
                  f"[{a}, {b}) with {len(own)} keyframes", flush=True)

    def _new_submap(self, a, b, own):
        if self.recorder is None:
            poses, pts, conf = self._forward(self._st.imgs[own], frames=own)
            full = None
        else:
            poses, pts, conf, full = self._forward(self._st.imgs[own], frames=own, dense=True)
        sm = Submap(a, b, poses, pts, conf, frames=own,
                    metric_scale=self._last_scale, metric=self._last_metric)
        sm.dense = full
        return sm

    def _record(self, sm, replaces=None):
        """Hand a committed submap's full-resolution pass to the map recorder."""
        if self.recorder is not None:
            if replaces is not None:
                self.recorder.drop(replaces)
            self.recorder.add(sm, self._st.imgs)
        sm.dense = None

    def _verify_ready(self, final=False):
        online_pgo = str(self.cfg.backend.loop.pgo_mode) == 'online'
        for _edge in self._loop_resolve(self._st.imgs, final):
            if online_pgo:
                self._correct_online()

    @torch.no_grad()
    def finish(self):
        """End of the sequence: the tail submap, the remaining windows and loops, then the
        final pose graph and a pose for every frame."""
        st = self._st
        n, size = st.n, int(self.cfg.backend.size)
        if st.submaps and st.submaps[-1].end < n:
            a, b = max(0, n - size), n
            own = self._submap_frames(st.conf_f, a, b)
            if len(own) >= max(4, int(self.cfg.backend.align.min_shared_frames)):
                sm = self._new_submap(a, b, own)
                self._add_submap(sm, st.imgs)
                self._record(sm)
                self._reanchor(sm)
        self._long_context_step(st.imgs, final=True)
        self._verify_ready(final=True)
        submaps = st.submaps
        st.det = None
        gc.collect()
        if st.verbose:
            print(f"  [backend] {len(submaps)} submaps, {len(st.loops)} loops", flush=True)
        if len(submaps) < 2:
            return self._result(st.online, submaps, st.online, list(st.S))
        S_t = self._optimise(st.S)
        return self._result(self._frame_poses(submaps, list(S_t), n), submaps, st.online,
                            list(S_t))

    def _submap_frames(self, conf_f, a, b):
        """Frames of [a, b) to reconstruct: the front-end's most confident frame per group
        of `frame_stride`, or the plain stride before any frame is scored."""
        d0 = max(1, int(self.cfg.backend.frame_stride))
        sc = np.asarray(conf_f, dtype=np.float64)
        if np.any(sc[a:b] > 0):
            kf = self._densify(self._group_keyframes(b, d0, sc), b, d0)
            return [f for f in kf if a <= f < b]
        return list(range(a + (-a) % d0, b, d0))

    def _reanchor(self, submap):
        """Hand a finished submap's poses to the front-end and the online trajectory."""
        st = self._st
        P = sim3_to_se3_matrix(sim3_mul(st.S[-1],
                                        sim3_from_matrix(submap.poses.double()).float()))
        st.fe.reanchor(submap.frames, P)
        for i, fr in enumerate(submap.frames):
            st.online[fr] = P[i].detach().cpu()

    def _frame_poses(self, submaps, S_t, n):
        """Per-frame poses: a weighted average over every submap containing the frame
        (down-weighted at submap edges), interpolated for unreconstructed frames."""
        acc_t = torch.zeros(n, 3, dtype=torch.float64)
        acc_q = torch.zeros(n, 4, dtype=torch.float64)
        acc_ls = torch.zeros(n, dtype=torch.float64)
        acc_w = torch.zeros(n, dtype=torch.float64)
        ref_q = torch.zeros(n, 4, dtype=torch.float64)
        edge = float(self.cfg.backend.edge_weight_floor)

        for k, sm in enumerate(submaps):
            fr = torch.as_tensor(sm.frames)
            nf = len(sm.frames)
            g_world = sim3_mul(S_t[k][None].expand(nf, 8),
                               sim3_from_matrix(sm.poses.double()).float()).double().cpu()
            mid = 0.5 * (sm.start + sm.end - 1)
            half_span = max((sm.end - sm.start - 1) / 2, 1)
            u = (fr.double() - mid).abs() / half_span
            w = (1.0 - (1.0 - edge) * u).clamp_min(1e-3)

            q = g_world[:, 3:7]
            first = acc_w[fr] == 0
            ref_q[fr] = torch.where(first[:, None], q, ref_q[fr])
            q = torch.where(((q * ref_q[fr]).sum(-1, keepdim=True) < 0), -q, q)

            acc_t.index_add_(0, fr, w[:, None] * g_world[:, :3])
            acc_q.index_add_(0, fr, w[:, None] * q)
            acc_ls.index_add_(0, fr, w * g_world[:, 7].clamp_min(1e-12).log())
            acc_w.index_add_(0, fr, w)

        ok = acc_w > 0
        if bool(ok.any()) and not bool(ok.all()):
            have = torch.nonzero(ok).squeeze(1)
            miss = torch.nonzero(~ok).squeeze(1)
            wsum0 = acc_w.clamp_min(1e-12)
            gq = acc_q / wsum0[:, None]
            g_have = torch.cat([acc_t[have] / wsum0[have, None],
                                gq[have] / gq[have].norm(dim=-1, keepdim=True).clamp_min(1e-12),
                                (acc_ls[have] / wsum0[have]).exp()[:, None]], dim=-1)
            pos = torch.searchsorted(have, miss).clamp(1, len(have) - 1)
            lo_i, hi_i = pos - 1, pos
            f_lo, f_hi = have[lo_i].double(), have[hi_i].double()
            t = ((miss.double() - f_lo) / (f_hi - f_lo).clamp_min(1)).clamp(0, 1)
            g_mid = sim3_interp(g_have[lo_i].float(), g_have[hi_i].float(), t.float())
            acc_t[miss] = g_mid[:, :3].double()
            acc_q[miss] = g_mid[:, 3:7].double()
            acc_ls[miss] = g_mid[:, 7].double().clamp_min(1e-12).log()
            acc_w[miss] = 1.0
            ok = acc_w > 0
        wsum = acc_w.clamp_min(1e-12)
        q = acc_q / wsum[:, None]
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        g = torch.cat([acc_t / wsum[:, None], q,
                       (acc_ls / wsum).exp()[:, None]], dim=-1).float()
        poses = sim3_to_se3_matrix(g)
        poses[~ok] = torch.eye(4)
        return poses

    def _loop_begin(self):
        """The retrieval index for this sequence, or None when loop closure is off."""
        self._st.loop = {'probe': 0, 'pending': [], 'seen': set()}
        return build_detector(self.cfg.retrieval, ROOT, self.device)

    def _loop_query(self, det, imgs, f):
        """Retrieval for frame ``f`` when it is a probe; a hit is queued for verification."""
        rcfg, L = self.cfg.retrieval, self._st.loop
        stride = int(rcfg.probe_every)
        if f % stride:
            return
        k = L['probe']
        hit = det.detect(imgs[f], max(1, int(rcfg.min_frame_gap) // stride))
        if hit is not None and hit < k:
            L['pending'].append((f, hit * stride, k, hit))
            det.propose(k, hit)
        L['probe'] = k + 1

    def _loop_resolve(self, imgs, final):
        """Verify, in order, the queued candidates whose submaps all exist. Yields each
        accepted edge.

        No later submap covers ``fj`` once the newest submap starts after it.
        """
        st = self._st
        pending = st.loop['pending']
        while pending:
            fj, fi, k, hit = pending[0]
            if not final and not (st.submaps and st.submaps[-1].start > fj):
                return
            pending.pop(0)
            edge = self._verify_loop(imgs, fj, fi)
            st.det.resolve(k, hit, edge is not None)
            if edge is not None:
                st.loops.append(edge)
                yield edge

    def _reject(self, reason):
        reasons = self.stats.setdefault('loop_reject_reasons', {})
        reasons[reason] = reasons.get(reason, 0) + 1

    def _verify_loop(self, imgs, fj, fi):
        """One retrieval candidate: `_loop_constraint`, then the drift gates.

        A verified closure is still rejected if its correction exceeds the chain's own
        plausible drift: chi = (correction / path) / (sigma_s sqrt(N)) > `max_drift_chi`,
        or rotation > `max_rot_drift_sqrt` sqrt(N) degrees, over N submaps between ends.
        Both are measured on the chain of local alignments, never on a corrected graph.
        Returns ``(ka, kb, Z)`` or None.
        """
        lcfg = self.cfg.backend.loop
        submaps, S = self._st.submaps, self._st.S_odo
        seen = self._st.loop['seen']
        max_chi = float(lcfg.max_drift_chi or 0.0)
        max_rot = float(lcfg.max_rot_drift_sqrt or 0.0)
        ka, kb = self._submap_of(fi, submaps), self._submap_of(fj, submaps)
        if ka is None or kb is None:
            return None
        if abs(ka - kb) < int(self.cfg.backend.align.max_pair_span) + 1:
            return self._reject('submaps too close')
        if (ka, kb) in seen:
            return self._reject('duplicate submap pair')
        seen.add((ka, kb))
        out = self._loop_constraint(imgs, submaps, ka, kb, fi, fj)
        if out is None:
            return self._reject('no usable windows')
        ka, kb, Z, info = out
        if Z is None:
            return self._reject(str(info.get('reason', '?')).split('(')[0].strip())

        cur = sim3_mul(sim3_inv(S[ka]), S[kb])
        dR = quat_to_rmat(cur[3:7]).T @ quat_to_rmat(Z.to(self.device)[3:7])
        drot = float(torch.rad2deg(torch.acos(((dR.diagonal().sum() - 1) / 2).clamp(-1, 1))))
        corr = float((cur[:3] - Z.to(self.device)[:3]).norm())
        a, b = sorted((int(ka), int(kb)))
        path = float(sum(float((S[i + 1][:3] - S[i][:3]).norm()) for i in range(a, b)))
        n = b - a
        log_s = np.log(np.clip([float(S[i][7]) for i in range(a, b + 1)], 1e-9, None))
        sig_s = float(np.median(np.abs(np.diff(log_s)))) if n >= 1 else 0.0
        if max_chi > 0 and path > 0 and sig_s > 0 and n >= 2 and \
                (corr / path) / (sig_s * math.sqrt(n)) > max_chi:
            return self._reject('drift')
        if max_rot > 0 and n >= 2 and drot > max_rot * math.sqrt(n):
            return self._reject('rotation drift')
        self.stats['n_loops'] += 1
        return ka, kb, Z
