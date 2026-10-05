"""The backend with a metric depth sensor: RGB-D and stereo (paper section 3.4)."""

from amb3r_slam.backend import HierarchicalBackend
from amb3r_slam.metric.depth import submap_metric_scale


class MetricBackend(HierarchicalBackend):
    """Submaps in metres, from a per-frame depth sensor."""

    def __init__(self, model, cfg):
        super().__init__(model, cfg)
        self.depth_source = None

    def reset(self, calib_K=None):
        super().reset(calib_K)
        self.depth_source = None

    def begin(self, imgs, fe, verbose=False):
        if self.depth_source is None:
            raise RuntimeError('this modality needs a depth source, and none is attached '
                               'for this dataset; see its Dataset.attach')
        super().begin(imgs, fe, verbose)

    def _to_metric(self, pts, poses, conf, frames):
        if self.depth_source is None:
            return pts, poses
        sd = self.depth_source.frame_depth(frames)
        if sd.shape[-2:] != pts.shape[1:3]:
            return pts, poses
        s, _ = self._submap_scale(pts, poses, conf, sd, frames)
        if s is None:
            return pts, poses
        poses = poses.clone()
        poses[:, :3, 3] *= s
        self._last_metric = True
        return pts * s, poses

    def _submap_scale(self, pts, poses, conf, sensor_depth, frames):
        """(scale, info) taking this pass into metres, scale None when unmeasurable."""
        return submap_metric_scale(pts, poses, sensor_depth, self.cfg.backend.metric_depth,
                                   conf=conf)

    def _scale_known(self, a, b):
        return (bool(self.cfg.backend.metric_depth.fix_scale)
                and a.metric and b.metric)
