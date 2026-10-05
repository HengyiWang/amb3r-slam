"""Benchmark datasets, one module each; importing the package registers them all."""

from amb3r_slam.datasets.common import DATASETS, Dataset, Sequence  # noqa: F401
from amb3r_slam.datasets import (  # noqa: F401
    bonn, crocodl, demo, eth3d, euroc, kitti, lamar, lamaria, spires, tum, vbr)


def get(name):
    return DATASETS[name]


def add_options(parser):
    """Every dataset's own command-line flags, each declared once."""
    seen = set()
    for cls in DATASETS.values():
        for flag, kw in cls.options:
            if flag not in seen:
                parser.add_argument(flag, **kw)
                seen.add(flag)
