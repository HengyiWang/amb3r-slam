"""Sim(3) helpers."""

import torch


def quat_to_rmat(q):
    """(..., 4) xyzw -> (..., 3, 3)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    R = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
    ], dim=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def rmat_to_quat(R):
    """(..., 3, 3) -> (..., 4) xyzw. Shepperd's method, branch-free via stacking."""
    batch = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    t0 = 1 + m00 + m11 + m22
    t1 = 1 + m00 - m11 - m22
    t2 = 1 - m00 + m11 - m22
    t3 = 1 - m00 - m11 + m22
    ts = torch.stack([t0, t1, t2, t3], dim=-1)
    pivot = ts.argmax(dim=-1)
    s = torch.sqrt(ts.gather(-1, pivot[:, None]).squeeze(-1).clamp_min(1e-12)) * 2

    cand = torch.stack([
        torch.stack([(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s], -1),
        torch.stack([0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s], -1),
        torch.stack([(m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s], -1),
        torch.stack([(m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s], -1),
    ], dim=1)
    q = cand.gather(1, pivot[:, None, None].expand(-1, 1, 4)).squeeze(1)
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = torch.where(q[:, 3:4] < 0, -q, q)
    return q.reshape(*batch, 4)


def sim3_from_matrix(T, scale=None):
    """(..., 4, 4) rigid transform (+ optional scale) -> (..., 8) Sim3 data."""
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    if scale is None:
        scale = R.det().clamp_min(1e-12) ** (1.0 / 3.0)
    scale = torch.as_tensor(scale, dtype=T.dtype, device=T.device).expand(T.shape[:-2])
    R = R / scale[..., None, None]
    q = rmat_to_quat(R)
    return torch.cat([t, q, scale[..., None]], dim=-1)


def sim3_to_se3_matrix(g):
    """(..., 8) -> (..., 4, 4) rigid part only (scale dropped from rotation, kept in t)."""
    t, q = g[..., :3], g[..., 3:7]
    T = torch.zeros(*g.shape[:-1], 4, 4, dtype=g.dtype, device=g.device)
    T[..., :3, :3] = quat_to_rmat(q)
    T[..., :3, 3] = t
    T[..., 3, 3] = 1.0
    return T


def sim3_inv(g):
    t, q, s = g[..., :3], g[..., 3:7], g[..., 7:]
    q_inv = torch.cat([-q[..., :3], q[..., 3:]], dim=-1)
    s_inv = 1.0 / s.clamp_min(1e-12)
    R_inv = quat_to_rmat(q_inv)
    t_inv = -s_inv * torch.einsum('...ij,...j->...i', R_inv, t)
    return torch.cat([t_inv, q_inv, s_inv], dim=-1)


def quat_mul(a, b):
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ], dim=-1)


def sim3_mul(a, b):
    """Compose: (a * b) acts as a(b(x))."""
    ta, qa, sa = a[..., :3], a[..., 3:7], a[..., 7:]
    tb, qb, sb = b[..., :3], b[..., 3:7], b[..., 7:]
    q = quat_mul(qa, qb)
    s = sa * sb
    t = ta + sa * torch.einsum('...ij,...j->...i', quat_to_rmat(qa), tb)
    return torch.cat([t, q, s], dim=-1)


def quat_slerp(qa, qb, t):
    """(..., 4) xyzw slerp; ``t`` broadcasts against the batch dims."""
    dot = (qa * qb).sum(-1, keepdim=True)
    qb = torch.where(dot < 0, -qb, qb)
    dot = dot.abs().clamp(max=1.0)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    small = sin_theta < 1e-6
    t = t[..., None] if t.dim() == qa.dim() - 1 else t
    wa = torch.where(small, 1.0 - t, torch.sin((1.0 - t) * theta) / sin_theta.clamp_min(1e-12))
    wb = torch.where(small, t, torch.sin(t * theta) / sin_theta.clamp_min(1e-12))
    q = wa * qa + wb * qb
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def sim3_interp(ga, gb, t):
    """Interpolate two Sim(3) elements: slerp rotation, lerp translation, geometric scale."""
    t = torch.as_tensor(t, dtype=ga.dtype, device=ga.device)
    tt = t[..., None] if t.dim() == ga.dim() - 1 else t
    q = quat_slerp(ga[..., 3:7], gb[..., 3:7], t)
    trans = (1 - tt) * ga[..., :3] + tt * gb[..., :3]
    s = torch.exp((1 - tt.squeeze(-1)) * torch.log(ga[..., 7].clamp_min(1e-12))
                  + tt.squeeze(-1) * torch.log(gb[..., 7].clamp_min(1e-12)))
    return torch.cat([trans, q, s[..., None]], dim=-1)


def average_transforms(transforms, weights):
    """Weighted mean of (N, 4, 4) rigid transforms: translations averaged linearly,
    rotations as sign-aligned quaternions (AMB3R's coordinate alignment)."""
    import numpy as np
    from scipy.spatial.transform import Rotation

    if torch.sum(weights) > 0:
        weights = weights / torch.sum(weights)
    else:
        weights = torch.ones_like(weights) / len(weights)
    t = torch.sum(weights.view(-1, 1) * transforms[:, :3, 3], dim=0)

    q = Rotation.from_matrix(transforms[:, :3, :3].cpu().numpy()).as_quat()
    for i in range(1, len(q)):
        if np.dot(q[0], q[i]) < 0:
            q[i] *= -1
    q = np.sum(weights.cpu().numpy().reshape(-1, 1) * q, axis=0)
    q /= np.linalg.norm(q)

    T = torch.eye(4, device=transforms.device, dtype=transforms.dtype)
    T[:3, :3] = torch.from_numpy(Rotation.from_quat(q).as_matrix()).to(
        transforms.device, dtype=transforms.dtype)
    T[:3, 3] = t
    return T
