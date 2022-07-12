from utils.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
import os
import numpy as np
import torch
from tqdm import tqdm
import kaolin.ops.spc as spc_ops
import trimesh
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
# Get the color map by name:
cm = plt.get_cmap('jet')

try:
    import open3d as o3d
except:
    print("run without open3d")


def o3d_load(file_pred, file_trgt, scene_config, is_mesh, bbx_name, save_dir):
    gt_min = np.array(scene_config[bbx_name][0])
    gt_max = np.array(scene_config[bbx_name][1])
    sfm_to_gt = np.array(scene_config['sfm2gt'])
    bbox_gt = o3d.geometry.AxisAlignedBoundingBox(min_bound=gt_min[:3], max_bound=gt_max[:3])

    pcd_trgt = o3d.io.read_point_cloud(file_trgt)
    pcd_trgt = pcd_trgt.crop(bbox_gt)
    verts_trgt = np.asarray(pcd_trgt.points)

    if is_mesh:
        # pcd_pred = o3d.io.read_point_cloud(file_pred)
        mesh_pred = o3d.io.read_triangle_mesh(file_pred)
        # transform to gt coordinates
        pcd_pred_pts = np.asarray(mesh_pred.vertices)
        pcd_pred_pts = np.concatenate((pcd_pred_pts, np.ones((pcd_pred_pts.shape[0], 1))), axis=-1)
        verts_pred = (sfm_to_gt[:3] @ pcd_pred_pts.T).T
        mesh_pred.vertices = o3d.utility.Vector3dVector(verts_pred)

        pcd_pred = mesh_pred.crop(bbox_gt)
        # pcd_pred = pcd_pred.sample_points_uniformly(number_of_points=int(verts_trgt.shape[0]) // 10)
        # verts_pred = np.asarray(pcd_pred.vertices)
        pcd_pred = pcd_pred.sample_points_uniformly(number_of_points=int(verts_trgt.shape[0]) * 10)
        verts_pred = np.asarray(pcd_pred.points)
    else:
        pcd_pred = o3d.io.read_point_cloud(file_pred)
        verts_pred = np.asarray(pcd_pred.points)
        # verts_pred = np.concatenate((verts_pred, np.ones((verts_pred.shape[0], 1))), axis=-1)
        # verts_pred = (sfm_to_gt[:3] @ verts_pred.T).T
        pcd_pred.points = o3d.utility.Vector3dVector(verts_pred)
        pcd_pred = pcd_pred.crop(bbox_gt)
        verts_pred = np.asarray(pcd_pred.points)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(verts_trgt)
    o3d.io.write_point_cloud(f"{save_dir}/down_gt.ply", gt_pcd)

    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(verts_pred[:, :3])
    o3d.io.write_point_cloud(f"{save_dir}/down_pred_in_gt.ply", gt_pcd)

    return verts_pred, verts_trgt


def trimesh_load(file_pred, file_trgt, scene_config, is_mesh, bbx_name, save_dir):
    
    sfm_to_gt = np.array(scene_config['sfm2gt'])

    pcd_trgt = trimesh.load(file_trgt)
    verts_trgt = np.asarray(pcd_trgt.vertices)
    verts_trgt = bbx_crop(verts_trgt, scene_config[bbx_name])
    pcd_trgt_bbx = trimesh.PointCloud(verts_trgt)
    pcd_trgt_bbx.export(f"{save_dir}/down_gt.ply")

    pcd_pred = trimesh.load(file_pred)
    verts_pred = np.asarray(pcd_pred.vertices)
    verts_pred = np.concatenate((verts_pred, np.ones((verts_pred.shape[0], 1))), axis=-1)
    verts_pred = (sfm_to_gt[:3] @ verts_pred.T).T 
    pcd_pred_bbx = trimesh.PointCloud(verts_pred)

    verts_pred = bbx_crop(verts_pred, scene_config[bbx_name])
    pcd_pred_bbx = trimesh.PointCloud(verts_pred)
    pcd_pred_bbx.export(f"{save_dir}/down_pred_in_gt.ply")

    return verts_pred, verts_trgt
    

def _compute(dist1, dist2, threshold):
    precision = max(np.mean((dist2 < threshold).astype('float')), 1e-6)
    recal = max(np.mean((dist1 < threshold).astype('float')), 1e-6)
    fscore = 2 * precision * recal / (precision + recal)
    metrics = {'dist1': np.mean(dist2),
               'dist2': np.mean(dist1),
               'prec': precision,
               'recal': recal,
               'fscore': fscore,
               }
    print(f"****metrics for threshold {threshold:.2f}****")
    print(metrics)
    print("******************************************")
    return metrics


def bbx_crop(points, bbx):
    bbx_min = np.array(bbx[0])
    bbx_max = np.array(bbx[1])
    
    dim_sfm = bbx_max - bbx_min
    bbx_origin = bbx_min + (bbx_max - bbx_min) / 2
    scale = dim_sfm / 2
    points_normalized = (points - bbx_origin) / scale

    mask = np.prod((points_normalized > -1), axis=-1, dtype=bool) & np.prod((points_normalized < 1), axis=-1, dtype=bool)
    return points[mask]


def visualize_error(pc, dists, save_dir, threshold):
    max_dist = threshold * 3
    dists = np.minimum(dists, max_dist) / max_dist
    rgbs = cm(dists.reshape(-1, 1))[:, :, :3].reshape(-1, 3)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(rgbs)
    o3d.io.write_point_cloud(save_dir, pcd)


def nn_correspondance(verts1, verts2, use_o3d=True):
    """ for each vertex in verts2 find the nearest vertex in verts1
    Args:
        nx3 np.array's
    Returns:
        ([indices], [distances])
    """

    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances
    if use_o3d:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(verts1)
        kdtree = o3d.geometry.KDTreeFlann(pcd)

        for vert in tqdm(verts2):
            _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
            indices.append(inds[0])
            distances.append(np.sqrt(dist[0]))
    else:
        kdtree = KDTree(verts1)
        for index, vert in enumerate(verts2):
            dist, inds = kdtree.query(vert, 1)
            indices.append(inds)
            distances.append(dist)

    return np.array(indices), np.array(distances)


def filtered_sfm(data_dir, sfm_to_gt, track_length=200, reproj_error=0.5, save_path=None):
    pts3d = read_points3d_binary(os.path.join(data_dir, 'points3D.bin'))
    pts3d_filtered = []
    for pts_id, pts in pts3d.items():
        if len(pts.point2D_idxs) > track_length and pts.error < reproj_error:
            pts3d_filtered.append(pts.xyz)

    pts3d_filtered = np.vstack(pts3d_filtered)
    pts3d_filtered = np.concatenate((pts3d_filtered, np.ones((pts3d_filtered.shape[0], 1))), axis=-1)
    pts3d_filtered = (sfm_to_gt[:3] @ pts3d_filtered.T).T 

    if not save_path is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts3d_filtered)
        o3d.io.write_point_cloud(save_path, pcd)

    return pts3d_filtered


def point_crop(src_pts, tsr_pts, voxel_size, bbx, batch_size=8, save_path=None, device=0):
    """crop point cloud by a set of voxels

    Args:
        src_pts (tensor): source point cloud to crop
        tsr_pts (tensor): voxel center
        voxel_size (tensor): voxel size
    """
    # dimensions
    bbx_min, bbx_max =  np.array(bbx[0]),  np.array(bbx[1])
    dim = np.max(bbx_max - bbx_min)

    scene_origin = bbx_min + (bbx_max - bbx_min) / 2
    scale = dim / 2

    tar_ind = src_pts.shape[0]
    all_pts = np.concatenate([src_pts, tsr_pts], axis=0)

    points_normalized = (all_pts - scene_origin) / scale
    points_normalized = torch.from_numpy(points_normalized).to(device)
    res = int(np.floor(2 * scale / voxel_size))
    quantized_pc = torch.floor(res * (points_normalized + 1.0) / 2.0).short()

    all_morton = spc_ops.points_to_morton(quantized_pc)
    src_morton, tar_morton = all_morton[:tar_ind], all_morton[tar_ind:]

    tar_morton = torch.unique(tar_morton, dim=0)

    _ind = torch.arange(0, tar_ind).long()
    all_mask = torch.zeros(tar_ind, dtype=torch.bool)
    for i in tqdm(range(0, tar_morton.size()[0], batch_size)):
        mask_sumed = torch.sum((tar_morton[i:i+batch_size, None] - src_morton[None, :]) == 0, axis=0) > 0
        all_mask = all_mask | mask_sumed.cpu()
    indices = _ind[all_mask]

    filtered = src_pts[indices]
    if not save_path is None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered)
        o3d.io.write_point_cloud(save_path, pcd)

    return src_pts[indices]