import os

import torch
import torch.nn as nn

from amb3r_slam import ROOT, thirdparty


def _checkpoint(ckpt):
    """A local path as given; a Hugging Face id resolves to `checkpoints/<name>` when that
    copy exists, else stays an id for `from_pretrained` to fetch."""
    if os.path.isabs(ckpt) or os.path.exists(os.path.join(ROOT, ckpt)):
        return ckpt if os.path.isabs(ckpt) else os.path.join(ROOT, ckpt)
    local = os.path.join(ROOT, 'checkpoints', ckpt.rstrip('/').split('/')[-1])
    return local if os.path.exists(local) else ckpt


def _c2w(w2c_34):
    """(..., 3, 4) world-to-camera -> (..., 4, 4) camera-to-world."""
    bottom = torch.zeros_like(w2c_34[..., :1, :])
    bottom[..., 0, 3] = 1
    return torch.linalg.inv(torch.cat([w2c_34, bottom], dim=-2))


class DA3(nn.Module):
    """Depth Anything 3: poses, depth and intrinsics from any number of views. The
    transformer backbones hold bf16 weights unless ``fp32``; the heads stay fp32."""

    DEFAULT = 'depth-anything/DA3NESTED-GIANT-LARGE-1.1'

    def __init__(self, ckpt=None, device='cuda', fp32=False):
        super().__init__()
        thirdparty('depth_anything_3/src')
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(_checkpoint(ckpt or self.DEFAULT))
        self.model = self.model.to(device).eval()
        if not fp32:
            from amb3r_slam.tools.model_dtype import cast_bf16
            cast_bf16(self._encoders())
        # The nested model's metric branch predicts sky, uses it internally, and drops it;
        # keep the last pass's mask (N, H, W, True = sky) for map export.
        self.last_sky = None
        metric = getattr(getattr(self.model, 'model', None), 'da3_metric', None)
        if metric is not None:
            metric.register_forward_hook(self._keep_sky)

    def _keep_sky(self, _module, _inputs, out):
        from depth_anything_3.utils.alignment import compute_sky_mask
        sky = getattr(out, 'sky', None)
        self.last_sky = None if sky is None else ~compute_sky_mask(sky, threshold=0.3)[0]

    def _encoders(self):
        heads = ('cam_dec', 'cam_enc', 'head', 'gs_head', 'dpt')
        return [(n, m) for n, m in self.named_modules()
                if n.endswith('backbone') and not set(n.split('.')) & set(heads)]

    def forward(self, views):
        from depth_anything_3.utils.geometry import unproject_depth
        imgs = (views['images'] + 1.0) / 2.0
        mean = torch.tensor([0.485, 0.456, 0.406], device=imgs.device).view(1, 1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=imgs.device).view(1, 1, 3, 1, 1)
        raw = self.model.forward((imgs - mean) / std, None, None, [], False, use_ray_pose=False,
                                 ref_view_strategy=views.get('ref_view_strategy', 'first'))
        pred = self.model._convert_to_prediction(raw)
        pred = self.model._align_to_input_extrinsics_intrinsics(None, None, pred, True)

        c2w = _c2w(torch.from_numpy(pred.extrinsics).to(imgs.device))
        K = torch.from_numpy(pred.intrinsics).unsqueeze(0).to(imgs.device)
        depth = torch.from_numpy(pred.depth).unsqueeze(0).unsqueeze(-1).to(imgs.device)
        # With a known camera, keep only the predicted depth and cast it along the real rays.
        if views.get('gt_intrinsics') is not None:
            K = views['gt_intrinsics'].to(imgs.device).float()
        pts = unproject_depth(depth, K, c2w.unsqueeze(0))
        return {'pose': c2w.unsqueeze(0).float(),
                'world_points': pts.float(),
                'world_points_conf': torch.from_numpy(pred.conf).unsqueeze(0).to(imgs.device).float(),
                'intrinsics': K.float(),
                # the nested model's up-to-scale -> metric factor
                'scale_factor': float(getattr(pred, 'scale_factor', None) or 1.0)}


class VGGTOmega(nn.Module):
    """VGGT-Omega: poses and depth; its intrinsics are used only to unproject. The
    aggregator holds bf16 weights unless ``fp32``; the heads stay fp32."""

    DEFAULT = 'checkpoints/vggt_omega_1b_512.pt'

    def __init__(self, ckpt=None, device='cuda', fp32=False):
        super().__init__()
        thirdparty('vggt-omega')
        from vggt_omega.models import VGGTOmega as Net
        from vggt_omega.utils.pose_enc import pose_encoding_to_extri_intri
        self.model = Net().eval().to(device)
        self.model.load_state_dict(torch.load(_checkpoint(ckpt or self.DEFAULT),
                                              map_location='cpu'))
        self._decode = pose_encoding_to_extri_intri
        self.device = device
        if not fp32:
            from amb3r_slam.tools.model_dtype import cast_bf16
            cast_bf16([('aggregator', self.model.aggregator)])

    def forward(self, views):
        images = (views['images'].to(self.device) + 1.0) / 2.0
        with torch.inference_mode():
            out = self.model(images)
        w2c, K = self._decode(out['pose_enc'], out['images'].shape[-2:])
        depth = out['depth'][..., 0]
        B, N, H, W = depth.shape
        y, x = torch.meshgrid(torch.arange(H, device=depth.device, dtype=depth.dtype),
                              torch.arange(W, device=depth.device, dtype=depth.dtype),
                              indexing='ij')
        fx, fy = K[:, :, 0, 0, None, None], K[:, :, 1, 1, None, None]
        cx, cy = K[:, :, 0, 2, None, None], K[:, :, 1, 2, None, None]
        cam = torch.stack([(x - cx) / fx * depth, (y - cy) / fy * depth, depth], dim=-1)
        R_T = w2c[:, :, :3, :3].transpose(-1, -2)
        t = w2c[:, :, :3, 3][:, :, None, None, :]
        pts = torch.einsum('bnij,bnhwj->bnhwi', R_T, cam - t)
        return {'pose': _c2w(w2c).float(),
                'world_points': pts.float(),
                'world_points_conf': out['depth_conf'].float()}


MODELS = {'da3': DA3, 'omega': VGGTOmega}


def load_model(name, ckpt=None, device='cuda', fp32=False):
    return MODELS[name](ckpt, device, fp32=fp32)


class SLAMModel:
    """One forward pass over a few views, returning dense poses, points and confidence.
    """

    def __init__(self, model, device, points_per_frame, patch=None):
        self.model = model
        self.device = device
        self.points_per_frame = int(points_per_frame)
        self.patch = patch

    @torch.no_grad()
    def __call__(self, images, ref, calib_K=None):
        """images (F, 3, H, W) in [-1, 1] -> (poses, pts, conf, K, metric factor), dense."""
        p = self.patch
        if p and (images.shape[-1] % p or images.shape[-2] % p):
            import torch.nn.functional as F
            H0, W0 = images.shape[-2:]
            size = (max(2 * p, int(round(H0 / p)) * p), max(2 * p, int(round(W0 / p)) * p))
            images = F.interpolate(images.float(), size=size, mode='bilinear',
                                   align_corners=False).clamp(-1, 1)
        views = {'images': images[None].to(self.device), 'ref_view_strategy': str(ref)}
        if calib_K is not None:
            views['gt_intrinsics'] = calib_K.to(self.device).expand(
                images.shape[0], 3, 3).float()[None]
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            res = self.model(views)
        Ki = res.get('intrinsics')
        return (res['pose'][0].float(), res['world_points'][0].float(),
                res['world_points_conf'][0].float(), None if Ki is None else Ki[0].float(),
                float(res.get('scale_factor', 1.0) or 1.0))

    def stride(self, h, w):
        """Pixel stride that keeps about `points_per_frame` points of an (h, w) frame."""
        return max(1, int(round((h * w / self.points_per_frame) ** 0.5)))

    def subsample(self, pts, conf):
        """Keep about `points_per_frame` points per frame, in half precision."""
        stride = self.stride(pts.shape[1], pts.shape[2])
        pts = pts[:, ::stride, ::stride].reshape(pts.shape[0], -1, 3).contiguous()
        conf = conf[:, ::stride, ::stride].reshape(conf.shape[0], -1).contiguous()
        return pts.half(), conf.half()
