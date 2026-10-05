"""Loop detection: retrieval only."""

import os
import numpy as np
import torch


class _Suppression:
    """Non-maximum suppression around closures, in (query, match) probe indices.

    A proposal suppresses its neighbourhood while it awaits verification (`propose`) and
    keeps doing so only if it is verified (`resolve`); a rejected one releases it.
    """

    def _suppression_init(self, radius):
        self.nms = int(radius)
        self.accepted = []
        self.inflight = []

    def _suppressed(self, n, j):
        return any((n - a) ** 2 + (j - b) ** 2 < self.nms ** 2
                   for a, b in self.accepted + self.inflight)

    def propose(self, i, j):
        self.inflight.append((int(i), int(j)))

    def resolve(self, i, j, ok):
        self.inflight.remove((int(i), int(j)))
        if ok:
            self.accepted.append((int(i), int(j)))


class DBoWDetector(_Suppression):
    """ORB bag-of-words retrieval through DBoW2 (``dpretrieval``)."""

    def __init__(self, cfg, repo_root):
        import dpretrieval

        vocab = cfg['vocab']
        if not os.path.isabs(vocab):
            vocab = os.path.join(repo_root, vocab)
        if not os.path.exists(vocab):
            raise FileNotFoundError(
                f"ORB vocabulary not found at {vocab}. Download ORBvoc.txt.tar.gz from "
                "https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/master/Vocabulary/")
        self.db = dpretrieval.DPRetrieval(vocab, int(cfg['nms']))
        self.thresh = float(cfg['thresh'])
        self.num_repeat = int(cfg['num_repeat'])
        self._suppression_init(cfg['nms'])
        self.n = 0
        self.hits = []
        self.island_k = int(cfg['island_k'])
        self.island_min = int(cfg['island_min'])
        self.island_radius = int(cfg['island_radius'])
        self.island_floor = float(cfg['island_floor'])
        self.recent = []

    @staticmethod
    def _to_uint8(img):
        """(3, H, W) in [-1, 1] -> (H, W, 3) uint8 contiguous."""
        arr = torch.round((img.float() + 1.0) * 127.5).clamp(0, 255).byte()
        return np.ascontiguousarray(arr.permute(1, 2, 0).cpu().numpy())

    def detect(self, img, min_gap):
        """Insert a keyframe and return a loop candidate keyframe index, or None."""
        n = self.n
        self.db.insert_image(self._to_uint8(img))
        score, j, _matches = self.db.query(n)
        self.n += 1
        if self.island_k > 0 and j >= 0 and score >= self.thresh * self.island_floor:
            self.recent.append((n, j, float(score)))
            self.recent = [r for r in self.recent if n - r[0] < self.island_k]

        if j < 0 or score < self.thresh or (n - j) < min_gap:
            return self._island(n, min_gap) if self.island_k > 0 else None

        if self._suppressed(n, j):
            return None

        self.hits.append((n, j))
        if len(self.hits) < self.num_repeat:
            return None
        recent = self.hits[-self.num_repeat:]
        if recent[-1][0] - recent[0][0] != self.num_repeat - 1:
            return None
        return int(max(recent[0][1], 1))

    def _island(self, n, min_gap):
        """The largest mutually-consistent group among the recent matches, or None."""
        if len(self.recent) < self.island_min:
            return None
        best = None
        for _, jc, _s in self.recent:
            grp = [r for r in self.recent if abs(r[1] - jc) <= self.island_radius]
            if len(grp) < self.island_min:
                continue
            if best is None or len(grp) > len(best) or (
                    len(grp) == len(best) and sum(g[2] for g in grp) > sum(b[2] for b in best)):
                best = grp
        if best is None:
            return None
        if not any(r[0] == n for r in best):
            return None
        j = max(best, key=lambda g: g[2])[1]
        if (n - j) < min_gap:
            return None
        if self._suppressed(n, j):
            return None
        return int(max(j, 1))


class _GlobalDescriptorDetector(_Suppression):
    """Cosine retrieval over one global descriptor per keyframe; subclasses pick the model."""

    def __init__(self, model, cfg, device):
        self.model = model.to(device).eval()
        self.device = device
        self.size = tuple(cfg['image_size'])
        self.thresh = float(cfg['similarity_threshold'])
        self.desc = []
        self._suppression_init(cfg['nms'])

    @torch.no_grad()
    def _describe(self, img):
        import torch.nn.functional as F
        x = ((img.float()[None] + 1.0) / 2.0).to(self.device)
        x = F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        return F.normalize(self.model((x - mean) / std).float().flatten(), dim=0)

    def detect(self, img, min_gap):
        d = self._describe(img)
        n = len(self.desc)
        self.desc.append(d)
        if n < min_gap:
            return None
        db = torch.stack(self.desc[: n - min_gap + 1])
        sims = db @ d
        best = int(sims.argmax())
        if float(sims[best]) < self.thresh:
            return None
        if self._suppressed(n, best):
            return None
        return best


class SaladDetector(_GlobalDescriptorDetector):
    """DINOv2-SALAD global descriptors."""

    def __init__(self, cfg, repo_root, device='cuda'):
        from amb3r_slam import thirdparty
        thirdparty()
        from salad import SaladModel
        ckpt = cfg['ckpt'] if os.path.isabs(cfg['ckpt']) else os.path.join(repo_root, cfg['ckpt'])
        model = SaladModel()
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        super().__init__(model, cfg, device)


class MegaLocDetector(_GlobalDescriptorDetector):
    """MegaLoc global descriptors; the weights come from the HF hub."""

    def __init__(self, cfg, repo_root, device='cuda'):
        from amb3r_slam import thirdparty
        thirdparty('MegaLoc')
        from hubconf import get_trained_model
        super().__init__(get_trained_model(), cfg, device)


def build_detector(cfg, repo_root, device='cuda'):
    if not cfg.enable:
        return None
    method = cfg.method
    sub = dict(cfg[method], nms=int(cfg.nms))
    if method == 'dbow':
        return DBoWDetector(sub, repo_root)
    if method == 'salad':
        return SaladDetector(sub, repo_root, device)
    if method == 'megaloc':
        return MegaLocDetector(sub, repo_root, device)
    raise ValueError(f"unknown loop detection method: {method}")
