"""AMB3R-SLAM."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def thirdparty(name=''):
    """Put ``thirdparty/<name>``, a checkout rather than an installed package, on the path."""
    path = os.path.join(ROOT, 'thirdparty', name)
    if path not in sys.path:
        sys.path.insert(0, path)
    return path
