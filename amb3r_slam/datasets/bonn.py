"""Bonn RGB-D Dynamic: TUM's layout and depth scale, so TUM's reader."""

from amb3r_slam.datasets.common import register
from amb3r_slam.datasets.tum import Tum


@register
class Bonn(Tum):
    name = 'bonn'
