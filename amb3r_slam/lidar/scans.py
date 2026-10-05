"""The interface the LiDAR block requires of a point-cloud source."""


class ScanSource:
    """Indexable source of per-frame point clouds; ``kiss_min_range`` / ``kiss_max_range``
    (metres) bound the returns the odometry uses."""

    kiss_min_range = 5.0
    kiss_max_range = 100.0

    def __len__(self):
        raise NotImplementedError

    def cloud(self, i):
        raise NotImplementedError

    def kiss_points(self, i):
        raise NotImplementedError
