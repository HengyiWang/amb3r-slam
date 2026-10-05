import os

import torch
from omegaconf import OmegaConf

from amb3r_slam import ROOT


MODALITIES = ('mono', 'rgbd', 'stereo', 'lidar')


def backend_class(modality):
    if modality == 'lidar':
        from amb3r_slam.lidar.backend import LidarBackend
        return LidarBackend
    if modality in ('rgbd', 'stereo'):
        from amb3r_slam.metric.backend import MetricBackend
        return MetricBackend
    from amb3r_slam.backend import HierarchicalBackend
    return HierarchicalBackend


def _layer(cfg_dir, name, _seen=None):
    _seen = _seen or set()
    assert name not in _seen, f'circular _include through {name!r}'
    _seen.add(name)
    path = os.path.join(cfg_dir, f'{name}.yaml')
    if not os.path.exists(path):
        return OmegaConf.create({})
    cfg = OmegaConf.load(path)
    includes = cfg.pop('_include', []) or []
    out = OmegaConf.create({})
    for inc in includes:
        out = OmegaConf.merge(out, _layer(cfg_dir, inc, _seen))
    return OmegaConf.merge(out, cfg)


def _unknown_keys(cfg, overrides, prefix=''):
    out = []
    for k, v in overrides.items():
        path = f'{prefix}{k}'
        if k not in cfg:
            out.append(path)
        elif isinstance(v, dict) and OmegaConf.is_dict(cfg[k]):
            out += _unknown_keys(cfg[k], v, path + '.')
    return out


class AMB3R_SLAM:
    def __init__(self, model, cfg_path=None, modality=None, overrides=None, fps=None,
                 device='cuda'):
        cfg_dir = os.path.join(ROOT, 'amb3r_slam', 'configs')
        self.cfg = OmegaConf.load(cfg_path or os.path.join(cfg_dir, 'base.yaml'))
        if modality:
            assert modality in MODALITIES, f'unknown modality {modality!r}'
            self.cfg = OmegaConf.merge(self.cfg, _layer(cfg_dir, modality))
        self.modality = modality or 'mono'
        if overrides:
            unknown = _unknown_keys(self.cfg, overrides)
            if unknown:
                raise ValueError(
                    f'--override names keys the {self.modality} configuration does not '
                    f'have: {", ".join(unknown)}')
            self.cfg = OmegaConf.merge(self.cfg, OmegaConf.create(overrides))
        self.cfg.device = str(device)
        self._submap_geometry(fps)
        self.model = model.to(self.cfg.device)
        self.device = self.cfg.device
        self.stats = {}
        self.backend = backend_class(self.modality)(self.model, self.cfg)
        self._fe_model = None

    def _submap_geometry(self, fps):
        """Frame stride k from the frame rate, and the frame counts that follow from it."""
        b = self.cfg.backend
        k = int(b.frame_stride)
        if fps:
            rate = float(b.frame_rate)
            k = max(1, int(round(k * min(float(fps), rate) / rate)))
        b.frame_stride = k
        b.step = int(b.new_views) * k
        b.size = int(b.views) * k
        b.warmup_min = b.step
        self.cfg.frontend.max_gap = b.step

    def reset(self, calib_K=None):
        """Per-sequence inputs: intrinsics (3, 3) or None; also clears sensor handles."""
        self.backend.reset(calib_K)

    def frontend(self):
        if self.modality == 'lidar':
            from amb3r_slam.lidar.frontend import LidarFrontEnd
            if self.backend.lidar_reg is None:
                raise RuntimeError('--lidar needs LiDAR scans, and this dataset attached none')
            return LidarFrontEnd(self.backend.lidar_reg, self.cfg)
        from amb3r_slam.frontend import FrontEnd
        return FrontEnd(self.frontend_model(), self.cfg, self.backend.calib_K)

    def frontend_model(self):
        """The camera front-end's smaller model (DA3-Small), loaded once;
        the backend's model when `frontend.ckpt` is empty."""
        ck = self.cfg.frontend.ckpt
        if not ck:
            return self.model
        if self._fe_model is None:
            from amb3r_slam.model import load_model
            self._fe_model = load_model('da3', str(ck), self.device, fp32=True).eval()
        return self._fe_model

    @torch.no_grad()
    def run(self, images, verbose=True):
        """images: (1, T, 3, H, W) in [-1, 1], streamed frame by frame. Returns a
        `BackendResult`."""
        assert images.min() >= -1 and images.max() <= 1
        imgs = images[0]
        fe = self.frontend()
        self.backend.begin(imgs, fe, verbose)
        for f in range(imgs.shape[0]):
            pose = fe.track(imgs, f)
            self.backend.add_frame(f, pose, fe.last_conf)
        res = self.backend.finish()
        self.stats = res.stats
        return res
