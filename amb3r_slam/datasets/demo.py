import os

import numpy as np

from amb3r_slam.datasets.common import Dataset, Sequence, _LazyBatch, _LazyFrames, register

IMAGE_EXTS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def image_files(root):
    names = sorted(f for f in os.listdir(root) if f.lower().endswith(IMAGE_EXTS))
    if not names:
        raise SystemExit(f'no images ({", ".join(IMAGE_EXTS)}) in {root}')
    return [os.path.join(root, n) for n in names]


@register
class Demo(Dataset):
    """A folder of frames from one camera. Pass ``--fps`` if it is below 10 Hz."""
    name = 'demo'

    def resolution(self):
        """The model's default width along the longer side, the other side by aspect."""
        import cv2
        h0, w0 = cv2.imread(image_files(self.root)[0]).shape[:2]
        long_, short = self.width, int(round(self.width * min(h0, w0) / max(h0, w0)
                                             / self.patch)) * self.patch
        short = max(2 * self.patch, short)
        return (long_, short) if w0 >= h0 else (short, long_)

    def sequences(self, resolution):
        import cv2
        W, H = resolution
        files = image_files(self.root)[::self.stride][:self.args.max_frames or None]
        imgs = np.empty((len(files), 3, H, W), dtype=np.uint8)
        for i, f in enumerate(files):
            im = cv2.imread(f)[:, :, ::-1]
            imgs[i] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA).transpose(2, 0, 1)
        name = os.path.basename(os.path.normpath(self.root))
        yield Sequence(name, _LazyBatch(_LazyFrames(imgs)))
