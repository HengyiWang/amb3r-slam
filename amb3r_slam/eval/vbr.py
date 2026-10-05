"""ATE against VBR's official ground truth, in the frame and at the timestamps it scores."""
import os

import numpy as np


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def read_tum(path):
    """(N, 4, 4) from `timestamp tx ty tz qx qy qz qw`, comments skipped."""
    raw = np.loadtxt(path)
    T = np.tile(np.eye(4), (len(raw), 1, 1))
    T[:, :3, 3] = raw[:, 1:4]
    for i, r in enumerate(raw):
        T[i, :3, :3] = _quat_to_R(r[4:8])
    return T


def _ate(P, G, with_scale):
    """Umeyama-aligned translational RMSE, the usual ATE."""
    x, y = P[:, :3, 3], G[:, :3, 3]
    mx, my = x.mean(0), y.mean(0)
    Xc, Yc = x - mx, y - my
    U, D, Vt = np.linalg.svd(Xc.T @ Yc / len(x))
    W = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        W[2, 2] = -1
    R = Vt.T @ W @ U.T
    s = 1.0
    if with_scale:
        s = float((D * np.diag(W)).sum() / ((Xc ** 2).sum() / len(x)))
    t = my - s * R @ mx
    e = np.linalg.norm(s * (R @ x.T).T + t - y, axis=1)
    A = np.tile(np.eye(4), (len(P), 1, 1))
    A[:, :3, :3] = R @ P[:, :3, :3]
    A[:, :3, 3] = s * (R @ x.T).T + t
    dR = np.einsum('nij,njk->nik', np.transpose(G[:, :3, :3], (0, 2, 1)), A[:, :3, :3])
    ang = np.degrees(np.arccos(np.clip(
        (np.trace(dR, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
    return (float(np.sqrt((e ** 2).mean())), float(s),
            float(np.sqrt((ang ** 2).mean())))


_PERCENTAGES = (0.01, 0.02, 0.03, 0.05, 0.08, 0.13, 0.21, 0.34, 0.55)


def _rpe(P, G):
    """KITTI-style relative error: (t_err %, r_err deg/m), averaged over all segments."""
    d = np.concatenate([[0.0], np.cumsum(
        np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1))])
    total = float(d[-1])
    if total <= 0:
        return float('nan'), float('nan')
    t_err, r_err, cnt = 0.0, 0.0, 0
    for pc in _PERCENTAGES:
        L = pc * total
        j = 0
        for i in range(len(G)):
            if j < i:
                j = i
            while j < len(G) and d[j] - d[i] < L:
                j += 1
            if j >= len(G):
                break
            Eg = np.linalg.inv(G[i]) @ G[j]
            Ee = np.linalg.inv(P[i]) @ P[j]
            E = np.linalg.inv(Eg) @ Ee
            t_err += float(np.linalg.norm(E[:3, 3])) / L
            r_err += float(np.degrees(np.arccos(np.clip(
                (np.trace(E[:3, :3]) - 1.0) / 2.0, -1.0, 1.0)))) / L
            cnt += 1
    if not cnt:
        return float('nan'), float('nan')
    return 100.0 * t_err / cnt, r_err / cnt


def bench_ate(seq, pred_cam, gt_cam=None, root=None, stride=1, max_frames=0):
    """ATE against `gt_official.txt`. `pred_cam` is (N, 4, 4) world<-camera, per loaded frame."""
    from amb3r_slam.datasets.vbr import vbr_calib, vbr_frame_to_sweep

    d = os.path.join(root, seq)
    off = os.path.join(d, 'gt_official.txt')
    pairs = os.path.join(d, 'pairs.json')
    calib = os.path.join(d, 'vbr_calib.yaml')
    if not (os.path.exists(off) and os.path.exists(pairs) and os.path.exists(calib)):
        return None

    names = sorted(os.listdir(os.path.join(d, 'rgb')))
    idx = list(range(0, len(names), stride))
    if max_frames:
        idx = idx[:max_frames]
    names = [names[i] for i in idx][:len(pred_cam)]

    f2s = vbr_frame_to_sweep(pairs, names)
    G_off = read_tum(off)
    T_cam_lidar = vbr_calib(calib, cam='cam_l')[3]

    keep = [i for i, s in enumerate(f2s)
            if s is not None and 0 <= int(s) < len(G_off) and i < len(pred_cam)]
    if len(keep) < 10:
        return None
    sw = [int(f2s[i]) for i in keep]
    P = pred_cam[keep] @ T_cam_lidar
    G = G_off[sw]

    out = {'frames': len(keep), 'sweeps_total': len(G_off)}
    out['bench_ate_se3'], _, out['bench_are_se3'] = _ate(P, G, False)
    out['bench_ate'], out['bench_scale'], out['bench_are'] = _ate(P, G, True)
    out['bench_rpe_trans'], out['bench_rpe_rot'] = _rpe(P, G)
    if gt_cam is not None and len(gt_cam) >= len(pred_cam):
        out['check_gt_vs_official_se3'] = _ate(gt_cam[keep] @ T_cam_lidar, G, False)[0]
    return out



def evaluate(seq, poses, root, stride=1, max_frames=0):
    """VBR's own leaderboard metrics, in its frame and at its timestamps."""
    b = bench_ate(seq.name[len('vbr_'):], np.asarray(poses, dtype=np.float64),
                  None if seq.poses is None else np.asarray(seq.poses, dtype=np.float64),
                  root=root, stride=stride, max_frames=max_frames or 0) or {}
    return {k: b.get(k) for k in ('bench_ate_se3', 'bench_are_se3', 'bench_rpe_trans',
                                  'bench_rpe_rot', 'bench_ate')}
