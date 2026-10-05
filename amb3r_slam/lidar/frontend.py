import numpy as np

from amb3r_slam.lidar.registration import se3_inv


class LidarOdometry:
    def __init__(self, scans, max_points_per_voxel=None):
        self.scans = scans
        self.max_points_per_voxel = max_points_per_voxel
        self._icp = None
        self._poses = []

    def __len__(self):
        return len(self.scans)

    def pose(self, i):
        """(4, 4) pose of scan ``i``, registering scans up to ``i`` if not yet done."""
        if self._icp is None:
            self._icp = self._make()
        while len(self._poses) <= int(i):
            xyz = np.asarray(self.scans.kiss_points(len(self._poses)), np.float64)
            if xyz.size == 0:
                self._poses.append(self._poses[-1].copy() if self._poses else np.eye(4))
                continue
            self._icp.register_frame(xyz, np.array([]))
            self._poses.append(np.asarray(self._icp.last_pose, np.float64).copy())
        return self._poses[int(i)]

    def relative(self, i, j):
        """(4, 4) transform taking scan `j` into scan `i`, or None outside the sequence."""
        i, j = int(i), int(j)
        if not (0 <= i < len(self.scans) and 0 <= j < len(self.scans)):
            return None
        return np.linalg.inv(self.pose(i)) @ self.pose(j)

    def _make(self):
        from kiss_icp.config import KISSConfig
        from kiss_icp.kiss_icp import KissICP
        cfg = KISSConfig()
        cfg.data.max_range = float(self.scans.kiss_max_range)
        cfg.data.min_range = float(self.scans.kiss_min_range)
        cfg.data.deskew = False
        cfg.mapping.voxel_size = cfg.data.max_range / 100.0
        if self.max_points_per_voxel:
            cfg.mapping.max_points_per_voxel = int(self.max_points_per_voxel)
        return KissICP(config=cfg)


class LidarFrontEnd:
    """The camera pose of each frame from the LiDAR odometry of its scan."""

    def __init__(self, lidar_reg, cfg):
        self.scans = lidar_reg['scans']
        self.f2s = lidar_reg.get('frame_to_scan')
        self.T_vc = se3_inv(np.asarray(lidar_reg['T_cv'], np.float64))
        self.odometry = LidarOdometry(self.scans, cfg.backend.lidar.kiss_max_points_per_voxel)
        self.last = np.eye(4)
        self.last_conf = None

    def scan_of(self, f):
        return self.f2s[f] if self.f2s is not None else f

    def track(self, imgs, f):
        """(4, 4) camera-to-world pose of frame ``f``; a frame without a scan keeps the
        previous pose."""
        s = self.scan_of(f)
        if s is not None and 0 <= int(s) < len(self.scans):
            self.last = self.odometry.pose(int(s)) @ self.T_vc
        return self.last
