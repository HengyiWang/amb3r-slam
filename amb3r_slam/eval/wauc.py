"""VidMap's windowed-ATE AUC (W-AUC), following its evaluation code."""

import numpy as np


def error_auc(errors, thresholds):
    """VidMap's AUC of the error CDF, normalised by the threshold (their `error_auc`)."""
    errors = np.sort(np.asarray(errors, dtype=np.float64))
    recall = (np.arange(len(errors)) + 1) / len(errors)
    errors = np.r_[0.0, errors]
    recall = np.r_[0.0, recall]
    out = []
    for t in thresholds:
        last = np.searchsorted(errors, t)
        r = np.r_[recall[:last], recall[last - 1]]
        e = np.r_[errors[:last], t]
        out.append(float(np.round(np.trapz(r, x=e) / t, 4)))
    return out


def align_umeyama_sim3(p_es, p_gt):
    """Sim(3) taking the estimate onto ground truth (their `align_umeyama_sim3`)."""
    mu_es, mu_gt = p_es.mean(0), p_gt.mean(0)
    es0, gt0 = p_es - mu_es, p_gt - mu_gt
    n = len(p_es)
    C = gt0.T @ es0 / n
    sigma2 = np.sum(es0 ** 2) / n
    U, D, Vt = np.linalg.svd(C)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / sigma2
    return s, R, mu_gt - s * R @ mu_es


def ransac_sim3(p_es, p_gt, max_error=1.0, iters=200, seed=0):
    """Robust Sim(3), standing in for COLMAP's `model_aligner --alignment_type custom`."""
    n = len(p_es)
    if n < 3:
        return align_umeyama_sim3(p_es, p_gt)
    rng = np.random.default_rng(seed)
    best, best_n = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 3, replace=False)
        try:
            s, R, t = align_umeyama_sim3(p_es[idx], p_gt[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(s) or s <= 0:
            continue
        e = np.linalg.norm(p_gt - (s * (R @ p_es.T).T + t), axis=1)
        k = int((e < max_error).sum())
        if k > best_n:
            best_n, best = k, (s, R, t)
    if best is None:
        return align_umeyama_sim3(p_es, p_gt)
    s, R, t = best
    e = np.linalg.norm(p_gt - (s * (R @ p_es.T).T + t), axis=1)
    inl = e < max_error
    if inl.sum() >= 3:
        s, R, t = align_umeyama_sim3(p_es[inl], p_gt[inl])
        e = np.linalg.norm(p_gt - (s * (R @ p_es.T).T + t), axis=1)
        inl = e < max_error
        if inl.sum() >= 3:
            s, R, t = align_umeyama_sim3(p_es[inl], p_gt[inl])
    return s, R, t


WINDOWS = (10, 25, 50, 100)
AUC_PCT = 5.0


def sequence_errors(p_es, p_gt, windows=WINDOWS, min_points=3):
    """Per-pose windowed errors, plus the normalised full-trajectory errors."""
    p_es = np.asarray(p_es, dtype=np.float64)
    p_gt = np.asarray(p_gt, dtype=np.float64)
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p_gt, axis=0), axis=1))])
    L = float(cum[-1])
    out = {}
    for W in windows:
        if W > L:
            out[W] = []
            continue
        half = W / 2.0
        errs = []
        for i in range(len(p_gt)):
            lo = max(0.0, min(cum[i] - half, L - W))
            m = (cum >= lo) & (cum <= lo + W)
            if int(m.sum()) < min_points:
                continue
            s, R, t = align_umeyama_sim3(p_es[m], p_gt[m])
            errs.append(float(np.linalg.norm(p_gt[i] - (s * R @ p_es[i] + t))))
        out[W] = errs
    s, R, t = ransac_sim3(p_es, p_gt)
    e = np.linalg.norm(p_gt - (s * (R @ p_es.T).T + t), axis=1)
    out['full'] = list(e / max(L, 1e-9)) if L > 0 else []
    out['path_length'] = L
    return out


def report(pool, windows=WINDOWS):
    """Pooled error lists -> the five W-AUC percentages of VidMap's Table 1."""
    row = {}
    for W in windows:
        e = pool.get(W, [])
        row[f'{W}m'] = 100.0 * error_auc(e, [W * AUC_PCT / 100.0])[0] if e else None
    e = pool.get('full', [])
    row['full'] = 100.0 * error_auc(e, [AUC_PCT / 100.0])[0] if e else None
    return row


def counts(pool, windows=WINDOWS):
    """How many per-pose errors each column was computed from."""
    out = {f'{W}m': len(pool.get(W, [])) for W in windows}
    out['full'] = len(pool.get('full', []))
    return out


def evaluate(seq, poses):
    """W-AUC errors and the RANSAC Sim(3) ATE for one sequence, on its posed frames."""
    p_es = poses[seq.gt_index][:, :3, 3]
    p_gt = np.asarray(seq.poses)[:, :3, 3]
    errs = sequence_errors(p_es, p_gt)
    sc, R, t = ransac_sim3(p_es, p_gt)
    ate = float(np.sqrt(np.mean(np.sum((sc * (R @ p_es.T).T + t - p_gt) ** 2, 1))))
    return {'scene': seq.scene, 'n_scored': int(len(seq.gt_index)),
            'path_length': errs['path_length'],
            'per_seq_wauc': report({k: errs[k] for k in list(WINDOWS) + ['full']}),
            'ate_sim3_m': round(ate, 3), 'sim3_scale': round(float(sc), 4),
            'errors': {str(k): [float(x) for x in errs[k]] for k in list(WINDOWS) + ['full']}}


def _pools(results):
    keys = list(WINDOWS) + ['full']
    by_scene = {}
    for v in results.values():
        if 'errors' not in v:
            continue
        d = by_scene.setdefault(v['scene'], {k: [] for k in keys})
        for k in keys:
            d[k] += v['errors'][str(k)]
    allp = {k: sum((d[k] for d in by_scene.values()), []) for k in keys}
    return by_scene, allp


def metrics(results, scenes):
    """The pooled W-AUC table, per scene and over all sequences, as VidMap reports it."""
    cols = [f'{k}m' for k in WINDOWS] + ['full']
    by_scene, allp = _pools(results)

    def row(pool):
        r = report(pool)
        return {c: (None if r[c] is None else round(r[c], 2)) for c in cols}

    per_scene = {s: row(by_scene[s]) for s in scenes if s in by_scene}

    def mean_col(c):
        v = [m[c] for m in per_scene.values() if m[c] is not None]
        return round(float(np.mean(v)), 2) if v else None

    return {'pooled': row(allp), 'per_scene': per_scene,
            'mean_over_scenes': {c: mean_col(c) for c in cols},
            'n_sequences': sum(1 for v in results.values() if 'errors' in v),
            'n_errors_per_column': counts(allp)}


def table(results, scenes):
    """The W-AUC table as text."""
    f = lambda v: f'{v:>10.1f}' if v is not None else f'{"--":>10}'  # noqa: E731
    m = metrics(results, scenes)
    cols = [f'{k}m' for k in WINDOWS] + ['full']
    lines = [f"{'scene':<8}{'seqs':>6}" + ''.join(f'{c:>10}' for c in cols)]
    for s, r in m['per_scene'].items():
        n = sum(1 for v in results.values() if v.get('scene') == s and 'errors' in v)
        lines.append(f'{s:<8}{n:>6}' + ''.join(f(r[c]) for c in cols))
    lines.append(f"{'POOLED':<8}{m['n_sequences']:>6}" + ''.join(f(m['pooled'][c]) for c in cols))
    lines.append(f"{'n_err':<8}{'':>6}" + ''.join(f"{m['n_errors_per_column'][c]:>10d}"
                                                  for c in cols))
    if m['per_scene']:
        lines.append(f"{'MEAN':<8}{len(m['per_scene']):>6}"
                     + ''.join(f(m['mean_over_scenes'][c]) for c in cols))
    return '\n'.join(lines)
