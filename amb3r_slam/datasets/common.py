"""What every dataset shares: lazy frame stacks, the sequence record, and the base class."""

from dataclasses import dataclass

import numpy as np
import torch

#: Default input size per model, before a dataset's own aspect rule.
MODEL_RES = {'da3': (518, 392), 'omega': (512, 384)}


class _LazyFrames:
    """A (N, 3, H, W) stack held as uint8 and converted to [-1, 1] on access."""

    __slots__ = ('u8', 'shape')

    def __init__(self, u8):
        self.u8 = u8
        self.shape = tuple(u8.shape)

    def __len__(self):
        return self.shape[0]

    def min(self):
        return -1.0

    def max(self):
        return 1.0

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.cpu().numpy()
        sub = self.u8[idx]
        t = torch.from_numpy(np.ascontiguousarray(sub)).float()
        return t / 255.0 * 2.0 - 1.0


class _LazyBatch:
    """`images` as the pipeline expects it: index [0] to get the frame stack."""

    __slots__ = ('frames', 'shape')

    def __init__(self, frames):
        self.frames = frames
        self.shape = (1,) + frames.shape

    def min(self):
        return -1.0

    def max(self):
        return 1.0

    def __getitem__(self, i):
        assert i == 0, 'only a single-sequence batch is supported'
        return self.frames


@dataclass
class Sequence:
    """One sequence to run: images (1, T, 3, H, W) in [-1, 1] and what it is scored against.

    ``poses`` are (G, 4, 4) camera-to-world ground truth for the images ``gt_index``
    (all images when None), or None for a test split. ``frames`` maps each image to its
    source frame index; ``K`` is (3, 3) at the input resolution, or None.
    """
    name: str
    images: object
    poses: object = None
    K: object = None
    frames: object = None
    gt_index: object = None
    scene: str = ''


DATASETS = {}


def register(cls):
    DATASETS[cls.name] = cls
    return cls


class Dataset:
    """A benchmark: where it lives, how its images are sized and loaded, which sensors it
    offers, and how it is scored.

    ``fps`` is the source frame rate, None when it is at least 10 Hz; ``stride`` the default
    frame subsampling (``--stride`` overrides it); ``options`` are the command-line flags only
    this dataset reads, as ``(flag, argparse kwargs)``.
    """

    name = None
    root = None
    scenes = ()
    fps = None
    stride = 1
    metrics = ('ate',)
    plot_mode = 'xy'
    options = ()

    def __init__(self, args, model_name):
        self.args = args
        self.model_name = model_name
        self.patch = 16 if model_name == 'omega' else 14
        self.width = MODEL_RES[model_name][0]
        self.stride = int(args.stride or self.stride)
        self.root = args.data_path or self.root
        if self.root is None:
            raise SystemExit(f'--data_path is required for {self.name}')

    def resolution(self):
        """Input (W, H): the model's default size, snapped to its patch grid."""
        w0, h0 = MODEL_RES[self.model_name]
        return w0, max(2 * self.patch, int(round(h0 / self.patch)) * self.patch)

    def rate(self):
        """Frame rate of the loaded stream, or None when it is at least 10 Hz."""
        return None if self.fps is None else self.fps / self.stride

    def wanted(self, name):
        """Whether ``--only`` selects this sequence (by substring of its name)."""
        only = getattr(self.args, 'only', None)
        return not only or any(o in name for o in only)

    def sequences(self, resolution):
        raise NotImplementedError

    def attach(self, seq, backend, cfg, modality, resolution):
        """Connect the sequence's own sensors -- depth, scans -- to the backend."""
        if modality != 'mono':
            raise SystemExit(f'{self.name} provides no {modality} sensor')
