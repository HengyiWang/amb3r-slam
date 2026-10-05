"""Absolute trajectory error through evo: Sim(3)-aligned, SE(3)-aligned, and on keyframes."""

import json
import os

import numpy as np


def _ate(gt, est, out_dir, label, correct_scale, plot_mode, plot=True):
    """RMSE of the translation error after aligning `est` onto `gt` (evo's Umeyama);
    with `plot`, also writes `stats_<label>.json` and `evo_2dplot_<label>.png`."""
    from evo.core import metrics
    from evo.core.trajectory import PosePath3D

    ref, traj = PosePath3D(poses_se3=gt), PosePath3D(poses_se3=est)
    _, _, s = traj.align(ref, correct_scale=correct_scale)
    ape = metrics.APE(metrics.PoseRelation.translation_part)
    ape.process_data((ref, traj))
    rmse = ape.get_statistic(metrics.StatisticsType.rmse)
    if plot:
        import evo.tools.plot as eplot
        import matplotlib.pyplot as plt
        stats = ape.get_all_statistics()
        with open(os.path.join(out_dir, f'stats_{label}.json'), 'w') as f:
            json.dump(stats, f, indent=4)
        mode = getattr(eplot.PlotMode, plot_mode)
        fig = plt.figure(figsize=(7, 7))
        ax = eplot.prepare_axis(fig, mode)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_title(f'ATE RMSE: {rmse}')
        eplot.traj(ax, mode, ref, '--', 'gray', 'gt')
        eplot.traj_colormap(ax, traj, ape.error, mode, min_map=stats['min'],
                            max_map=stats['max'])
        ax.legend()
        plt.savefig(os.path.join(out_dir, f'evo_2dplot_{label}.png'), dpi=90)
        plt.close(fig)
    return float(rmse), s


def evaluate(seq, poses, keyframes, out_dir, plot_mode='xy'):
    gt = np.asarray(seq.poses)
    ate, s = _ate(gt, poses, out_dir, seq.name, True, plot_mode)
    try:
        ate_se3 = _ate(gt, poses, out_dir, f'{seq.name}_se3', False, plot_mode, plot=False)[0]
    except Exception:                                             
        ate_se3 = None
    kf = np.asarray(keyframes, dtype=int)
    ate_kf = None
    if len(kf) > 1:
        try:
            ate_kf = _ate(gt[kf], poses[kf], out_dir, f'{seq.name}_kf', True, plot_mode)[0]
        except Exception as e:                                    
            print(f'  [warn] keyframe ATE unavailable: {type(e).__name__}: {e}', flush=True)
    return {'ate': ate, 'ate_kf': ate_kf, 'ate_se3': ate_se3,
            'sim3_scale': float(np.asarray(s).reshape(-1)[0])}
