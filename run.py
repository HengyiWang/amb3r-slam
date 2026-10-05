"""Runs AMB3R-SLAM on one dataset and modality, and writes per-sequence results,
trajectories, the resolved config and a summary table under `--out` (default
outputs/<dataset>_<sensor>_<model>).

    python run.py --dataset tum --data_path $TUM [--sensor rgbd]
"""

import argparse
import json
import os
import time
import traceback

import numpy as np
import torch

from amb3r_slam import ROOT, datasets
from amb3r_slam import eval as evaluation
from amb3r_slam.model import MODELS, load_model
from amb3r_slam.pipeline import AMB3R_SLAM

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def parse_overrides(pairs):
    """['a.b=1', 'c=false'] -> nested dict with parsed scalars; later pairs win."""
    out = {}
    for p in pairs or []:
        k, v = p.split('=', 1)
        try:
            val = json.loads(v)
        except json.JSONDecodeError:
            val = {'true': True, 'false': False, 'null': None}.get(v.lower(), v)
        node = out
        parts = k.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = val
    return out


def arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='tum', choices=sorted(datasets.DATASETS))
    ap.add_argument('--data_path', default=None, help="default: the dataset's own root")
    ap.add_argument('--scenes', nargs='*', default=None)
    ap.add_argument('--only', nargs='*', default=None,
                    help='run only sequences whose name contains one of these')
    ap.add_argument('--sensor', default='mono', choices=('mono', 'rgbd', 'stereo', 'lidar'),
                    help='mono; rgbd or stereo: metric depth from the depth sensor or the '
                         'stereo pair; lidar: ICP odometry, the model for loop closure')
    ap.add_argument('--stride', type=int, default=None,
                    help="keep every n-th frame; default: the dataset's own (2 for TUM and "
                         'Bonn, else 1)')
    ap.add_argument('--max_frames', type=int, default=None)
    ap.add_argument('--model_name', default='da3', choices=sorted(MODELS))
    ap.add_argument('--ckpt_path', default=None,
                    help="local path or Hugging Face id; default: the model's own")
    ap.add_argument('--fp32', action='store_true',
                    help="keep the model's transformer encoder in fp32 (default: bf16 "
                         'weights, fp32 heads)')
    ap.add_argument('--fps', type=float, default=None,
                    help="input frame rate, which sets the submap geometry; default: the "
                         "dataset's own rate, or at least 10 Hz")
    ap.add_argument('--res_hw', default=None,
                    help="explicit input resolution WxH, overriding the dataset's rule")
    ap.add_argument('--calib', action='store_true',
                    help="give the system the dataset's camera intrinsics (default: "
                         'uncalibrated; images are undistorted either way)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--config', default=os.path.join(ROOT, 'amb3r_slam', 'configs',
                                                     'base.yaml'))
    ap.add_argument('--override', nargs='*', default=None)
    ap.add_argument('--out', default=None,
                    help='output directory; default: outputs/<dataset>_<sensor>_<model>')
    ap.add_argument('--save_map', action='store_true',
                    help="also write each sequence's map as a coloured point cloud "
                         '(map_<seq>.ply): full-resolution points, low-confidence, '
                         'depth-edge and sky pixels removed')
    ap.add_argument('--save_every', type=int, default=2,
                    help='with --save_map, keep one of every this many mapped frames (default 2)')
    ap.add_argument('--pixel_stride', type=int, default=4,
                    help='with --save_map, the pixel stride of the second, lighter cloud '
                         '(map_<seq>_stride.ply)')
    ap.add_argument('--map_max_points', type=int, default=0,
                    help='with --save_map, thin the cloud to at most this many points (0: all)')
    ap.add_argument('--verbose', action='store_true', help='per-stage progress')
    datasets.add_options(ap)
    return ap.parse_args()


def main():
    args = arguments()
    ds = datasets.get(args.dataset)(args, args.model_name)

    resolution = ds.resolution()
    if args.res_hw:
        w, h = (int(v) for v in str(args.res_hw).lower().split('x'))
        if w % ds.patch or h % ds.patch:
            raise SystemExit(f'--res_hw {args.res_hw} is not a multiple of patch {ds.patch}')
        resolution = (w, h)
    print(f'{ds.name} input resolution: '
          + ('per sequence' if resolution is None else f'{resolution[0]}x{resolution[1]}'),
          flush=True)

    out_dir = args.out or os.path.join(
        ROOT, 'outputs', f'{args.dataset}_{args.sensor}_{args.model_name}')
    os.makedirs(out_dir, exist_ok=True)

    model = load_model(args.model_name, args.ckpt_path, args.device, fp32=args.fp32).eval()
    modality = args.sensor
    overrides = parse_overrides(args.override)
    pipeline = AMB3R_SLAM(model, cfg_path=args.config, modality=modality,
                          overrides=overrides, fps=args.fps or ds.rate(), device=args.device)

    from omegaconf import OmegaConf
    with open(os.path.join(out_dir, 'config.json'), 'w') as f:
        json.dump({'resolved': OmegaConf.to_container(pipeline.cfg, resolve=True),
                   'modality': modality, 'fps': args.fps or ds.rate(),
                   'overrides': overrides, 'args': vars(args)}, f, indent=1)
    print(f'config: {modality} ({type(pipeline.backend).__name__}) | size '
          f'{pipeline.cfg.backend.size} step {pipeline.cfg.backend.step} frame_stride '
          f'{pipeline.cfg.backend.frame_stride} -> {out_dir}/config.json', flush=True)

    res_path = os.path.join(out_dir, 'results.json')
    results = {}

    def save():
        with open(res_path, 'w') as f:
            json.dump(results, f, indent=1, default=float)

    for seq in ds.sequences(resolution):
        if not ds.wanted(seq.name):
            continue
        n = seq.images.shape[1]
        print(f'\n=== {seq.name}: {n} frames ===', flush=True)
        try:
            torch.cuda.reset_peak_memory_stats()
            pipeline.reset(calib_K=(torch.from_numpy(np.asarray(seq.K)).float()
                                    if args.calib and seq.K is not None else None))
            ds.attach(seq, pipeline.backend, pipeline.cfg, modality, resolution)
            if args.save_map:
                from amb3r_slam.tools.export import MapRecorder
                pipeline.backend.recorder = MapRecorder(
                    pipeline.cfg.backend.frame_stride, args.save_every, args.pixel_stride,
                    max_points=args.map_max_points)
            t0 = time.time()
            mem = pipeline.run(seq.images, verbose=args.verbose)
            dt = time.time() - t0
            poses = mem.poses[:n].numpy()
            st = pipeline.stats
            np.savez_compressed(
                os.path.join(out_dir, f"traj_{seq.name.replace('/', '_')}.npz"),
                pred=poses, gt=(np.zeros((0, 4, 4)) if seq.poses is None
                                else np.asarray(seq.poses)),
                gt_index=(np.arange(0) if seq.gt_index is None else seq.gt_index),
                kf=np.array(mem.keyframes),
                submap_scale=np.array([[float(sm.metric_scale or 0.0), min(sm.frames),
                                        max(sm.frames)] for sm in mem.submaps
                                       if len(sm.frames)], dtype=np.float64).reshape(-1, 3))
            if args.save_map:
                ply = os.path.join(out_dir, f"map_{seq.name.replace('/', '_')}.ply")
                npts = pipeline.backend.recorder.save(ply, mem)
                pipeline.backend.recorder = None
                print(f'  map: {npts[0]} points -> {ply}, {npts[1]} at pixel stride '
                      f'{args.pixel_stride} -> {ply[:-4]}_stride.ply' if npts else
                      '  map: this backend keeps no submap points', flush=True)
            rec = {'scene': seq.scene, 'frames': int(n), 'keyframes': len(mem.keyframes),
                   'time_s': dt, 'fps': n / max(dt, 1e-9),
                   'loops': int(st.get('n_loops', 0)),
                   'loop_reject_reasons': st.get('loop_reject_reasons', {}),
                   'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9}
            if seq.poses is None:
                rec['no_gt'] = True
                print(f'  {seq.name}: no ground truth, trajectory saved', flush=True)
            else:
                rec.update(evaluation.evaluate(ds, seq, poses, mem.keyframes, out_dir))
                score = (f"ATE={rec['ate']:.4f} kfATE={rec['ate_kf']}" if 'ate' in rec
                         else f"ATE(sim3)={rec['ate_sim3_m']:.3f} W-AUC={rec['per_seq_wauc']}")
                print(f"--> {seq.name}: {score} fps={rec['fps']:.2f} loops={rec['loops']}",
                      flush=True)
            results[seq.name] = rec
        except Exception as e:                                    # noqa: BLE001
            traceback.print_exc()
            results[seq.name] = {'scene': seq.scene, 'error': f'{type(e).__name__}: {e}'}
        finally:
            torch.cuda.empty_cache()
        save()

    if 'wauc' in ds.metrics:
        from amb3r_slam.eval.wauc import metrics
        with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics(results, list(ds.scenes)), f, indent=1)
    text = evaluation.summary(ds, results, os.path.basename(os.path.normpath(out_dir)))
    print('\n' + text, flush=True)
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write(text + '\n')


if __name__ == '__main__':
    main()
