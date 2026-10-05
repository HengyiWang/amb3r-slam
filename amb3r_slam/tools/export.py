"""Export a run's map as a coloured point cloud from the full-resolution submap passes."""

import numpy as np
import torch
import torch.nn.functional as F

from amb3r_slam.tools.geometry import quat_to_rmat


def write_ply(path, xyz, rgb):
    """Binary PLY with float32 positions and uint8 colours."""
    v = np.empty(len(xyz), dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
                                  ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    v['x'], v['y'], v['z'] = xyz.T
    v['red'], v['green'], v['blue'] = rgb.T
    with open(path, 'wb') as f:
        f.write((f'ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n'
                 'property float x\nproperty float y\nproperty float z\n'
                 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
                 'end_header\n').encode())
        f.write(v.tobytes())


def depth_edge(depth, rtol=0.03, k=3):
    """(H, W) True where the depth jumps by more than ``rtol`` (relative) within a k x k
    window -- the pixels that straddle a discontinuity (VGGT-Omega's `depth_edge`)."""
    d = F.pad(depth[None, None], (k // 2,) * 4, mode='replicate')
    hi = F.max_pool2d(d, k, stride=1)[0, 0]
    lo = -F.max_pool2d(-d, k, stride=1)[0, 0]
    return (hi - lo) / depth.abs().clamp_min(1e-6) > rtol


class MapRecorder:
    def __init__(self, frame_stride, save_every=2, pixel_stride=4, conf_drop=0.2,
                 edge_rtol=0.03, max_points=0):
        self.every = max(1, int(save_every) * int(frame_stride))
        self.stride = max(1, int(pixel_stride))
        self.conf_drop, self.edge_rtol, self.max_points = conf_drop, edge_rtol, int(max_points)
        self.best = {}

    def add(self, sm, images):
        pts, conf, sky = sm.dense
        mid = 0.5 * (sm.start + sm.end - 1)
        for i, f in enumerate(sm.frames):
            slot, dist = f // self.every, abs(f - mid)
            if slot in self.best and self.best[slot][0] <= dist:
                continue
            p, c = pts[i].float(), conf[i].float()
            T = sm.poses[i].float().to(p.device)
            depth = ((p - T[:3, 3]) @ T[:3, :3])[..., 2]
            keep = (c >= torch.quantile(c.flatten(), self.conf_drop)) & (depth > 0)
            keep &= ~depth_edge(depth, self.edge_rtol)
            if sky is not None:
                keep &= ~sky[i].to(keep.device)
            col = ((images[f].permute(1, 2, 0).to(keep.device) + 1.0) * 127.5).clamp(0, 255)
            grid = torch.zeros_like(keep)
            grid[::self.stride, ::self.stride] = True
            self.best[slot] = (dist, sm, p[keep].half().cpu(), col[keep].byte().cpu(),
                               grid[keep].cpu())

    def drop(self, sm):
        """Forget the frames held for ``sm`` (a short warm-up submap being replaced)."""
        self.best = {s: v for s, v in self.best.items() if v[1] is not sm}

    def _cloud(self, xyz, rgb):
        if self.max_points and len(xyz) > self.max_points:
            idx = np.linspace(0, len(xyz) - 1, self.max_points).astype(np.int64)
            xyz, rgb = xyz[idx], rgb[idx]
        return xyz, rgb

    def save(self, path, result):
        """World points through each submap's final Sim(3), written to ``path`` (every
        pixel) and ``<path>_stride.ply``. Returns the two point counts, or None when the
        result has no submaps to place them with."""
        if not result.submaps or result.submap_poses is None or not self.best:
            return None
        index = {id(sm): k for k, sm in enumerate(result.submaps)}
        xyz, rgb, on = [], [], []
        for f in sorted(self.best):
            _, sm, p, col, g_on = self.best[f]
            k = index.get(id(sm))
            if k is None:
                continue
            g = result.submap_poses[k].double().cpu()
            R, t, s = quat_to_rmat(g[3:7]), g[:3], g[7]
            xyz.append((s * p.double() @ R.T + t).float().numpy())
            rgb.append(col.numpy())
            on.append(g_on.numpy())
        xyz, rgb, on = (np.concatenate(x) for x in (xyz, rgb, on))
        counts = []
        for out, sel in ((path, slice(None)), (path[:-len('.ply')] + '_stride.ply', on)):
            x, r = self._cloud(xyz[sel], rgb[sel])
            write_ply(out, x, r)
            counts.append(len(x))
        return tuple(counts)
