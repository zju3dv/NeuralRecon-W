import sys

sys.path.insert(1, ".")

import open3d as o3d

from itertools import repeat, product
import torch
import numpy as np
from loguru import logger
import os
from utils.colmap_utils import read_points3d_binary
from kaolin.ops import spc
import kaolin.render.spc as spc_render
import yaml
from tqdm import tqdm


def filter_p3d_w_track_length(points3D, min_track_length=12):
    points = []
    for id, p in points3D.items():
        if p.point2D_idxs.shape[0] > min_track_length:
            points.append(p.xyz)
    return np.array(points)


def expand_points(points, voxel_size):
    """
    A naive version of the sparse dilation.
    """
    # a cube with size=3 and step=1.
    cube_grids_3 = list(product(*zip([-1, -1, -1], [0, 0, 0], [1, 1, 1])))
    # add the offsets to the points.
    points_expanded = [
        points + np.array(grid_point) * voxel_size for grid_point in cube_grids_3
    ]
    points_expanded = np.concatenate(points_expanded, axis=0)
    return np.unique(points_expanded, axis=0)


def gen_octree_from_sfm(
    recontruct_path,
    min_track_length,
    voxel_size,
    sfm_path="sparse",
    device=0,
    visualize=False,
    expand=True,
    radius=1.0,
):
    # read 3d points from sfm result, and filter them
    point_path = os.path.join(recontruct_path, f"dense/{sfm_path}/points3D.bin")
    points_3d = read_points3d_binary(point_path)
    points_ori = []
    for id, p in points_3d.items():
        if p.point2D_idxs.shape[0] > min_track_length:
            points_ori.append(p.xyz)
    points = np.array(points_ori)
    logger.debug(
        f"Points filtered from raw point cloud: {points.shape[0]}/{len(points_3d)}"
    )

    if visualize:
        print(f"saving... at samples/voxel_vis_source.ply")
        os.makedirs("samples", exist_ok=True)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        o3d.io.write_point_cloud(f"samples/voxel_vis_source.ply", gt_pcd)

    return gen_octree(
        recontruct_path, points, voxel_size, device, visualize, expand, radius
    )


def gen_octree(
    recontruct_path,
    points,
    voxel_size,
    device=0,
    visualize=False,
    expand=1,
    radius=1.0,
    in_sfm=True,
):
    scene_config_path = os.path.join(recontruct_path, "config.yaml")
    # read scene config
    with open(scene_config_path, "r") as yamlfile:
        scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # scene_origin = scene_config['origin']

    # transform bbx to sfm coordinates system
    if in_sfm:
        sfm_to_gt = np.array(scene_config["sfm2gt"])
        gt_to_sfm = np.linalg.inv(sfm_to_gt)
        sfm_vert1 = (
            gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][0]) + gt_to_sfm[:3, 3]
        )
        sfm_vert2 = (
            gt_to_sfm[:3, :3] @ np.array(scene_config["eval_bbx"][1]) + gt_to_sfm[:3, 3]
        )
        bbx_min = np.minimum(sfm_vert1, sfm_vert2)
        bbx_max = np.maximum(sfm_vert1, sfm_vert2)
    else:
        bbx_min = np.array(scene_config["eval_bbx"][0])
        bbx_max = np.array(scene_config["eval_bbx"][1])

    # dimensions
    dim = np.max(bbx_max - bbx_min)

    # points dialation
    for _ in range(expand):
        points = expand_points(points, voxel_size)

    # normalize cropped area to [-1, -1]
    scene_origin = bbx_min + (bbx_max - bbx_min) / 2
    scale = dim / 2 * radius
    points_normalized = (points - scene_origin) / scale

    # filter out points out of [-1, 1]
    mask = np.prod((points_normalized > -1), axis=-1, dtype=bool) & np.prod(
        (points_normalized < 1), axis=-1, dtype=bool
    )
    points_filtered = points_normalized[mask]
    logger.debug(
        f"number of points for voxel generation: {points_filtered.shape[0]}/{points_normalized.shape[0]}"
    )

    if visualize:
        print(f"saving... at samples/voxel_vis_norm.ply")
        os.makedirs("samples", exist_ok=True)
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(
            points_normalized[:, :3] * scale + scene_origin
        )
        o3d.io.write_point_cloud(f"samples/voxel_vis_norm.ply", gt_pcd)

        print(f"saving... at samples/voxel_vis_filtered.ply")
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(
            points_filtered[:, :3] * scale + scene_origin
        )
        o3d.io.write_point_cloud(f"samples/voxel_vis_filtered.ply", gt_pcd)

    # calculate level
    points_filtered = torch.from_numpy(points_filtered).to(device)  # N, 3
    level = int(np.floor(np.log2(2 * scale / voxel_size)))
    logger.debug(f"level: {level} for expected voxel size: {voxel_size}")

    quantized_pc = spc.points.quantize_points(points_filtered, level)
    octree = spc.unbatched_points_to_octree(quantized_pc, level)

    if visualize:
        # vis spc
        print(f"saving... at samples/spc_points_level_{level}_expand_{expand}.ply")

        points, pyramid, prefix = octree_to_spc(octree)
        os.makedirs("samples", exist_ok=True)
        level_pointx = points[pyramid[1, level] : pyramid[1, level + 1], :3].cpu()
        pyramid = pyramid.cpu()

        level_pointx = (
            (level_pointx / (2 ** (pyramid.size()[1] - 2))) * 2 - 1.0
        ) * scale + scene_origin
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(level_pointx.cpu().numpy())
        o3d.io.write_point_cloud(
            f"samples/spc_points_level_{level}_expand_{expand}.ply", gt_pcd
        )

    return octree, scene_origin, scale, level


def octree_to_spc(octree):
    lengths = torch.tensor([len(octree)], dtype=torch.int32)
    _, pyramid, prefix = spc.scan_octrees(octree, lengths)
    points = spc.generate_points(octree, pyramid, prefix)
    pyramid = pyramid[0]
    return points, pyramid, prefix


def convert_to_dense(octree, level):
    points, pyramid, prefix = octree_to_spc(octree)
    level_pointx = points[pyramid[1, level] : pyramid[1, level + 1], :3]
    occ = torch.ones_like(level_pointx[:, 0]).unsqueeze(1)
    dense = spc.to_dense(points, pyramid.unsqueeze(0), occ.float(), level)
    return dense.squeeze()


def level_upgrade(
    octree,
    octree_origin,
    octree_scale,
    src_level,
    target_level,
    recontruct_path,
    visualize=False,
):
    # upsample octree
    device = octree.device
    dense = convert_to_dense(octree, src_level).cpu()
    low_dim = dense.size()[0]

    # dense to sparse
    sparse_ind = torch.nonzero(dense > 0)  # n, 3
    sparse_num = sparse_ind.size()[0]

    # upsample
    up_level = target_level - src_level
    up_times = 2**up_level

    target_dim = int(low_dim * (2**up_level))
    print(
        f"target dim: {target_level}, upsampled {up_times} times, original dim {low_dim}, sparse num {sparse_num}"
    )

    sparse_ind_up = sparse_ind.repeat_interleave(up_times**3, dim=0) * up_times
    up_kernel = torch.arange(0, up_times, 1)
    up_kernal = torch.stack(
        torch.meshgrid(up_kernel, up_kernel, up_kernel), dim=-1
    ).reshape(
        -1, 3
    )  # up_times**3, 3
    expand_kernal = up_kernal.repeat([sparse_num, 1])  # sparse_num * up_times**3, 3

    sparse_ind_up = sparse_ind_up + expand_kernal

    # to sfm coordinate
    target_voxel_size = 2 / (2**target_level) * octree_scale
    vol_origin = octree_origin - octree_scale

    xyz_sfm = sparse_ind_up * target_voxel_size + vol_origin

    return gen_octree(
        recontruct_path,
        xyz_sfm.cpu().numpy(),
        target_voxel_size,
        device=device,
        visualize=visualize,
        expand=False,
    )


def level_downgrade(
    octree,
    octree_origin,
    octree_scale,
    src_level,
    target_level,
    recontruct_path,
    visualize=False,
):
    device = octree.device
    # downsample octree
    points, pyramid, prefix = octree_to_spc(octree)
    down_level_pts = (
        points[pyramid[1, target_level] : pyramid[1, target_level + 1], :3]
        .cpu()
        .numpy()
    )

    # to sfm
    xyz_sfm = (
        (down_level_pts / (2 ** (target_level))) * 2 - 1.0
    ) * octree_scale + octree_origin

    target_voxel_size = 2 / (2**target_level) * octree_scale

    return gen_octree(
        recontruct_path,
        xyz_sfm,
        target_voxel_size,
        device=device,
        visualize=visualize,
        expand=False,
    )


def octree_level_adjust(
    octree,
    octree_origin,
    octree_scale,
    src_level,
    target_level,
    recontruct_path,
    visualize,
):
    if target_level > src_level:
        return level_upgrade(
            octree,
            octree_origin,
            octree_scale,
            src_level,
            target_level,
            recontruct_path,
            visualize,
        )
    elif target_level < src_level:
        return level_downgrade(
            octree,
            octree_origin,
            octree_scale,
            src_level,
            target_level,
            recontruct_path,
            visualize,
        )
    else:
        return octree, octree_origin, octree_scale, src_level


def get_near_far(
    rays_o,
    rays_d,
    octree,
    scene_origin,
    scale,
    level,
    spc_data=None,
    visualize=False,
    ind=0,
    with_exit=False,
    return_pts=False,
):
    """
    'rays_o': ray origin in sfm coordinate system
    'rays_d': ray direction in sfm coordinate system
    'octree': spc

    'with_exit': set true to obtain accurate far. Default to false as this will perform aabb twice
    """
    # Avoid corner cases. issuse in kaolin: https://github.com/NVIDIAGameWorks/kaolin/issues/490
    rays_d = rays_d.clone() + 1e-7
    rays_o = rays_o.clone() + 1e-7

    if spc_data is None:
        points, pyramid, prefix = octree_to_spc(octree)
    else:
        points, pyramid, prefix = (
            spc_data["points"],
            spc_data["pyramid"],
            spc_data["prefix"],
        )

    # transform origin from sfm to kaolin [-1, 1]
    rays_o_normalized = (rays_o - scene_origin) / scale

    # legacy kaolin version of v0.9.1
    # nugs = spc_render.unbatched_raytrace(octree, points, pyramid, prefix,
    #                                          rays_o_normalized.float(), rays_d.float(), level) # (ind ray, ind point)

    # rays_near, _, _ = spc_render.unbatched_ray_aabb(nugs, points, rays_o_normalized, rays_d, level)
    # rays_far, _, _ = spc_render.unbatched_ray_aabb(torch.flip(nugs, [0]), points, rays_o_normalized, rays_d, level)

    rays_pid = torch.ones_like(rays_o_normalized[:, :1]) * -1
    rays_near = torch.zeros_like(rays_o_normalized[:, :1])
    rays_far = torch.zeros_like(rays_o_normalized[:, :1])

    ray_index, pt_ids, depth_in_out = spc_render.unbatched_raytrace(
        octree,
        points,
        pyramid,
        prefix,
        rays_o_normalized,
        rays_d,
        level,
        return_depth=True,
        with_exit=with_exit,
    )
    ray_index = ray_index.long()
    if not with_exit:
        # if no exit, far will be the entry point of the last intersecting. This is an inaccurate far, but will be more efficient
        depth_in_out = torch.cat([depth_in_out, depth_in_out], axis=1)

    near_index, near_count = torch.unique_consecutive(ray_index, return_counts=True)

    if ray_index.size()[0] == 0:
        print("[WARNING] batch has 0 intersections!!")
        if return_pts:
            return rays_near, rays_far, rays_pid
        else:
            return rays_near, rays_far

    near_inv = torch.roll(torch.cumsum(near_count, dim=0), shifts=1)
    near_inv[0] = 0

    far_index, far_count = torch.unique_consecutive(
        torch.flip(ray_index, [0]), return_counts=True
    )
    far_inv = torch.roll(torch.cumsum(far_count, dim=0), shifts=1)
    far_inv[0] = 0
    far_inv = ((ray_index.size()[0] - 1) - far_inv).long()

    rays_pid[near_index] = pt_ids[near_inv].reshape(-1, 1).float()
    rays_near[near_index] = depth_in_out[near_inv, :1]
    rays_far[far_index] = depth_in_out[far_inv, 1:]

    valid = (rays_near) > 1e-4
    rays_near[~valid] = 0
    rays_far[~valid] = 0
    rays_pid[~valid] = -1

    near_points_sfm = rays_o + rays_d * rays_near * scale
    far_points_sfm = rays_o + rays_d * rays_far * scale

    error_mask = (rays_far[valid] - rays_near[valid]) < -1e-4
    assert (
        torch.sum(error_mask) == 0
    ), f"invalid near far from intersection : {torch.sum(error_mask)}/{error_mask.size()[0]}"

    if visualize:
        os.makedirs(f"samples/spc_intersect/{ind}", exist_ok=True)
        print(f"saving... at samples/spc_intersect/{ind}")
        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(near_points_sfm.cpu().numpy())
        o3d.io.write_point_cloud(
            f"samples/spc_intersect/{ind}/near_level_{level}.ply", gt_pcd
        )

        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(far_points_sfm.cpu().numpy())
        o3d.io.write_point_cloud(
            f"samples/spc_intersect/{ind}/far_level_{level}.ply", gt_pcd
        )

    error_mask = (rays_far[valid] - rays_near[valid]) < -1e-4
    if torch.sum(error_mask) > 0:
        print(
            rays_far[valid][error_mask],
            rays_near[valid][error_mask],
            rays_far[valid][error_mask] - rays_near[valid][error_mask],
        )
    assert (
        torch.sum(error_mask) == 0
    ), f"invalid near far from intersection : {torch.sum(error_mask)}/{error_mask.size()[0]}. Maybe try to reduce intersection batch size?"

    if return_pts:
        return rays_near * scale, rays_far * scale, rays_pid
    else:
        return rays_near * scale, rays_far * scale


# debugging
if __name__ == "__main__":

    from datasets import dataset_dict
    from torch.utils.data import DataLoader

    voxel_size = 0.1
    recontruct_path = "/nas/datasets/IMC/phototourism/training_set/brandenburg_gate"
    min_track_length = 50
    octree, scene_origin, scale, level = gen_octree_from_sfm(
        recontruct_path, min_track_length, voxel_size, visualize=True
    )
    # gen fake ray origin and direction
    rays_o = (
        torch.from_numpy(np.array([-0.2874, -0.1635, -0.6985]).reshape(1, 3)).cuda()
        * 9.8
    )
    rays_d = torch.from_numpy(np.array([-0.0242, -0.0924, 0.9954]).reshape(1, 3)).cuda()

    kwargs = {
        "root_dir": "/nas/datasets/IMC/phototourism/training_set/brandenburg_gate",
        "split": "test_train",
    }
    kwargs["with_semantics"] = False
    kwargs["img_downscale"] = 1

    dataset = dataset_dict["phototourism"](**kwargs)

    test_dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=True
    )

    for idx, sample in enumerate(tqdm(test_dataloader)):
        rays = sample["rays"][0].cuda()
        ts = sample["ts"][0].cuda()
        B = rays.shape[0]

        w, h = sample["img_wh"][0]
        chunk = 256
        for i in tqdm(range(0, B, chunk)):
            batch_rays = rays[i : i + chunk]
            near, far = get_near_far(
                batch_rays[:, 0:3],
                batch_rays[:, 3:6],
                octree,
                torch.from_numpy(scene_origin).cuda(),
                scale,
                level,
                visualize=True,
            )
