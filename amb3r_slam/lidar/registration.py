import numpy as np

_HAVE_O3D = None


def _o3d():
    global _HAVE_O3D
    import open3d as o3d
    _HAVE_O3D = True
    return o3d


_DEVS = {}


def _dev(name):
    """Cached `o3d.core.Device`; constructing one per call is not free at this rate."""
    if name not in _DEVS:
        _DEVS[name] = _o3d().core.Device(name)
    return _DEVS[name]


def se3_inv(T):
    R, t = T[:3, :3], T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def _place_map(scans, i, rel_pose, win, voxel, radius, device, cuda, min_pts, normals):
    """Local map around scan `i`, expressed in scan `i`'s own frame."""
    import open3d as o3d
    pts = []
    for k in range(int(i) - int(win), int(i) + int(win) + 1):
        if k < 0 or k >= len(scans):
            continue
        T = np.eye(4) if k == i else rel_pose(int(i), int(k))
        if T is None:
            continue
        P = np.asarray(scans.cloud(int(k)).points)
        if len(P) == 0:
            continue
        pts.append(P @ T[:3, :3].T + T[:3, 3])
    if not pts:
        return None
    P = np.concatenate(pts, 0)
    P = P[np.linalg.norm(P, axis=1) < float(radius)]
    if len(P) < min_pts:
        return None
    if cuda:
        m = o3d.t.geometry.PointCloud(
            o3d.core.Tensor(np.ascontiguousarray(P), o3d.core.float32, _dev(device)))
        m = m.voxel_down_sample(float(voxel))
        if len(m.point.positions) < min_pts:
            return None
        if normals:
            m.estimate_normals(max_nn=30, radius=1.0)
        return m
    m = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P))
    m = m.voxel_down_sample(float(voxel))
    if len(m.points) < min_pts:
        return None
    if normals:
        m.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30))
    return m


def register_places(scans, ia, ib, T_init, rel_pose, win=10, voxel=0.5, max_corr=1.5,
                    iters=60, backend='cuda', device='CUDA:0', radius=75.0,
                    min_map_pts=500, dist='p2pl'):
    """Refine a loop transform by registering the place at `ib` onto the place at `ia`."""
    import open3d as o3d
    cuda = (str(backend) == 'cuda')
    normals = (dist != 'p2p')
    ma = _place_map(scans, ia, rel_pose, win, voxel, radius, device, cuda,
                    min_map_pts, normals)
    mb = _place_map(scans, ib, rel_pose, win, voxel, radius, device, cuda,
                    min_map_pts, False)
    if ma is None or mb is None:
        return None, 0.0, {'reason': 'place map too small'}
    T0 = np.asarray(T_init, np.float64)
    if cuda:
        import open3d.t.pipelines.registration as treg
        est = (treg.TransformationEstimationPointToPoint() if dist == 'p2p'
               else treg.TransformationEstimationPointToPlane())
        r = treg.icp(mb, ma, float(max_corr),
                     o3d.core.Tensor(T0, o3d.core.float64, _dev(device)), est,
                     treg.ICPConvergenceCriteria(max_iteration=int(iters)))
        T, fit = r.transformation.cpu().numpy().astype(np.float64), float(r.fitness)
        rmse = float(r.inlier_rmse)
    else:
        est = (o3d.pipelines.registration.TransformationEstimationPointToPoint()
               if dist == 'p2p'
               else o3d.pipelines.registration.TransformationEstimationPointToPlane())
        res = o3d.pipelines.registration.registration_icp(
            mb, ma, float(max_corr), T0, est,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(iters)))
        T, fit = np.asarray(res.transformation, np.float64), float(res.fitness)
        rmse = float(res.inlier_rmse)
    dt = float(np.linalg.norm(T[:3, 3] - T0[:3, 3]))
    dR = T0[:3, :3].T @ T[:3, :3]
    dr = float(np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return T, fit, {'d_trans': dt, 'd_rot_deg': dr, 'inlier_rmse': rmse}


