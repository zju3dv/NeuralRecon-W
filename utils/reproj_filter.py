import sys
sys.path.insert(1, '.')

import argparse
import json
import os

import numpy as np
import trimesh
from tools.reproj_error import get_image_id, get_entrinsics, get_intrinsic
from utils.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
from utils.kaolin_renderer import kaolin_renderer
from utils.pyrender_renderer import pyrender_renderer

import open3d as o3d
import ray
from tqdm import tqdm
import cv2
import os
import yaml
import matplotlib.pyplot as plt
import glob
import pandas as pd

# Get the color map by name:
cm = plt.get_cmap('gist_rainbow')

os.environ['PYOPENGL_PLATFORM'] = 'egl'
def parse_args():
    parser = argparse.ArgumentParser(description="NeuralRecon ScanNet Testing")
    parser.add_argument('--src_file', type=str,
                        default='results/phototourism/bg_voxel_5_step-20211104_030249_iter_300000/mesh/extracted_mesh_level_10_colored.ply',
                        help='mesh file path')
    parser.add_argument('--target_file', type=str,
                        default='results/phototourism/bg_voxel_5_step-20211104_030249_iter_300000/mesh/extracted_mesh_level_10_colored.ply',
                        help='point cloud to be filtered')
    parser.add_argument('--data_path', type=str,
                        default="/nas/datasets/IMC/phototourism/training_set/brandenburg_gate",
                        help='camera poses in colmap format')
    parser.add_argument('--output_path', type=str,
                        default='results/phototourism/bg_voxel_5_step-20211104_030249_iter_300000/mesh',
                        help='output path')  
    parser.add_argument('--gt', default=False, action="store_true",
                        help='whether target pc/mesh is in gt coordinates system')
    parser.add_argument('--visualize', default=False, action="store_true",
                        help='whether to store render results')
    parser.add_argument('--voxel_size', type=float, default=0.01,
                        help='voxel size in world coordinate system')   

    # ray config
    parser.add_argument('--n_cpus', type=int, default=1)
    parser.add_argument('--n_gpus', type=int, default=4)
    return parser.parse_args()


args = parse_args()


def get_image_id(imdata, data_dir):
    img_path_to_id = {}
    img_id_to_name={}
    img_ids = []
    for v in imdata.values():
        img_path_to_id[v.name] = v.id
        img_id_to_name[v.id] = v.name
        img_ids += [v.id]
    return img_ids, img_id_to_name, img_path_to_id

def get_train_ids(data_path, img_ids_all, img_path_to_id):
    tsv = glob.glob(os.path.join(data_path, '*.tsv'))[0]
    files = pd.read_csv(tsv, sep='\t')
    files = files[~files['id'].isnull()] # remove data without id
    files.reset_index(inplace=True, drop=True)

    img_ids_file = []
    for filename in list(files['filename']):
        id_ = img_path_to_id[filename]
        img_ids_file += [id_]

    img_ids = [id_ for i, id_ in enumerate(img_ids_file) if files.loc[i, 'split']=='train']

    return img_ids

def sfm2gt(src_pts):
    scene_config_path = os.path.join(args.data_path, 'config.yaml')
    with open(scene_config_path, "r") as yamlfile:
        scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    src_pts_homo = np.concatenate((src_pts, np.ones((src_pts.shape[0], 1))), axis=-1).T
    sfm_to_gt = np.array(scene_config['sfm2gt'])
    sfm_pts = (sfm_to_gt[:3] @ src_pts_homo).T
    return sfm_pts

def get_sfm2gt():
    scene_config_path = os.path.join(args.data_path, 'config.yaml')
    with open(scene_config_path, "r") as yamlfile:
        scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    sfm_to_gt = np.array(scene_config['sfm2gt'])
    return sfm_to_gt

def load_mesh_to_render(src_file):
    try:
        mesh = trimesh.load(src_file, process=False)
        if not args.gt:
            mesh.vertices = sfm2gt(mesh.vertices)
        render = pyrender_renderer(mesh=mesh)
        print("***************************")
        print("******reproject mesh******")
        print("***************************")
    except:
        src_pc = o3d.io.read_point_cloud(src_file)
        src_pts = np.asarray(src_pc.points)
        if not args.gt:
            src_pts = sfm2gt(src_pts)
        render = kaolin_renderer(args.data_path, src_pts, args.voxel_size)

        # mesh_opengl = pyrender.Mesh.from_points(src_pts)  
        # radius = 0.01
        # sm = trimesh.creation.uv_sphere(radius=radius, count=[4,4])
        # sm.visual.vertex_color = [1.0, 0.0, 0.0]
        # tfs = np.tile(np.eye(4), (len(src_pts), 1, 1))
        # tfs[:, :3, 3] = src_pts
        # mesh_opengl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
        # render = pyrender_renderer(mesh=mesh_opengl)

        print("***************************")
        print("***reproject point cloud***")
        print("***************************")

    return render
    

def reproject(height, width, depth, intrinsic, pose, valid_mask):
    # project to camera coordinates
    cor_h = np.concatenate([np.arange(0, height, 1).reshape((height, 1))] * width, axis=1)  # (h, w)
    cor_w = np.concatenate([np.arange(0, width, 1).reshape((1, width))] * height, axis=0)  # (h, w)
    coordinates_raw = np.stack([cor_w, cor_h, np.ones_like(depth)], axis=0)  # (3, h, w)

    # preserve only valid points
    coordinates = coordinates_raw[:, valid_mask]  # (3, h x w)
    depth_valid = depth[valid_mask].reshape(1, -1)  # (1, h x w)

    # point cloud in camera coordinates
    coordinates = np.linalg.inv(intrinsic) @ coordinates  # (3, h x w)
    pc_cam = (coordinates * depth_valid)  # (3, h x w)

    # point cloud in world coordinates
    homo_pc = np.concatenate([pc_cam, np.ones((1, pc_cam.shape[1]))], axis=0)  # (4, h x w)
    pc_world = pose[:3, :] @ homo_pc  # (3, h x w)
    pc_xyz = pc_world.T

    return pc_xyz

def render_visualize(depth, rgb, reprojects, name):
    # remove name suffix
    name = name[:-4]

    depth_dir = os.path.join(args.output_path, "render", "depth")
    os.makedirs(depth_dir, exist_ok=True)
    cv2.imwrite(f'{depth_dir}/{name}.jpg',cm(depth)[:, :, :3] * 255)

    rgb_dir = os.path.join(args.output_path, "render", "rgb")
    os.makedirs(rgb_dir, exist_ok=True)
    cv2.imwrite(f'{rgb_dir}/{name}.jpg',rgb)

    pc_dir = os.path.join(args.output_path, "render", "reprojects")
    os.makedirs(pc_dir, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(reprojects)
    o3d.io.write_point_cloud(f'{pc_dir}/{name}.ply', pcd)

@ray.remote(num_cpus=1, num_gpus=0.5)
def reprojection_worker(intrinsics, extrinsics, whs, names):
    # transform to gt coordinate
    extrinsics = extrinsics @ np.linalg.inv(get_sfm2gt())

    pcd = o3d.io.read_point_cloud(args.target_file)
    if not args.gt:
        pcd.points = o3d.utility.Vector3dVector(sfm2gt(np.asarray(pcd.points)))

    # construct kdtree for fast look up nearest neighbor
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    # color and vertex from target pc
    colors = np.asarray(pcd.colors)
    vertices = np.asarray(pcd.points)
    if colors.shape != vertices.shape:
        print("No color found in target point cloud")
        colors = np.zeros_like(vertices)

    # load source mesh/pc and generate renderer
    renderer = load_mesh_to_render(args.src_file)

    points_all = np.zeros((0, 6))
    if hasattr(renderer, "get_index"):
        indices_all = np.zeros((renderer.voxel_num), dtype=bool)
    else:
        indices_all = np.zeros((vertices.shape[0]), dtype=bool)

    item_num = len(intrinsics)

    for i in tqdm(range(0, item_num)):
        pose = np.linalg.inv(extrinsics[i])
        intrinsic = intrinsics[i]
        width, height = whs[i]

        try:
            rgb, depth = renderer(height, width, intrinsic, pose)
            valid_mask = depth > 0
        except BaseException as err:
            print(f"Unexpected {err=}, {type(err)=}")
            print(f"[WARNING] intersection failed at {names[i]} with pose")
            raise

        # project to world coordinates
        pc_xyz = reproject(height, width, depth, intrinsic, pose, valid_mask)

        if args.visualize:
            render_visualize(depth, rgb, pc_xyz, names[i])

        if np.sum(valid_mask) < 10:
            print(f"[WARNING] invalid view at {names[i]}")

        if hasattr(renderer, "get_index"):
            indices_all[renderer.pids.reshape(-1)] = True
        else:
            indices=[]
            # get correspondence
            for point in pc_xyz:
                _, inds, dist = kdtree.search_knn_vector_3d(point, 1)
                if np.sqrt(dist[0]) < 2 * np.sqrt(2) * args.voxel_size:
                    indices_all[inds[0]] = True

    if hasattr(renderer, "get_index"):
        indices = renderer.get_index(indices_all)
    else:
        indices = indices_all

    pc_rgb = colors[indices]
    pc_xyz = vertices[indices]
    points_all = np.hstack([pc_xyz, pc_rgb])

    return points_all


def split_list(_list, n):
    assert len(_list) >= n
    ret = [[] for _ in range(n)]
    for idx, item in enumerate(_list):
        ret[idx % n].append(item)
    return ret


if __name__ == "__main__":
    reconstuct_path = 'dense/sparse'
    print(f"result will be saved to {args.output_path}")
    os.makedirs(args.output_path, exist_ok=True)

    imdata = read_images_binary(os.path.join(args.data_path, reconstuct_path, 'images.bin'))
    camdata = read_cameras_binary(os.path.join(args.data_path, reconstuct_path, 'cameras.bin'))
    img_ids_all, img_id_to_name, img_path_to_id = get_image_id(imdata, args.data_path)
    img_ids = get_train_ids(args.data_path, img_ids_all, img_path_to_id)
    print(f"views to process: {len(img_ids)}")

    entrinsics = get_entrinsics(imdata, img_ids)
    entrinsics_dict = {id_: entrinsics[i] for i, id_ in enumerate(img_ids)}
    intrinsics_dict, whs_dict = get_intrinsic(camdata, img_ids, imdata)

    # dict to list
    entrinsics_list = [entrinsics_dict[k] for k in img_ids]
    intrinsics_list = [intrinsics_dict[k] for k in img_ids]
    whs_list = [whs_dict[k] for k in img_ids]
    name_list = [img_id_to_name[k] for k in img_ids]

    # generate splits
    all_proc = args.n_cpus
    ray.init(num_cpus=args.n_cpus, num_gpus=args.n_gpus)
    ray_worker_ids = []
    entrinsic_split = split_list(entrinsics_list, all_proc)
    intrinsic_split = split_list(intrinsics_list, all_proc)
    whs_split = split_list(whs_list, all_proc)
    name_split = split_list(name_list, all_proc)

    for w_idx in range(all_proc):
        # results = [reprojection_worker(intrinsic_split[w_idx], entrinsic_split[w_idx], whs_split[w_idx], name_split[w_idx])]
        ray_worker_ids.append(reprojection_worker.remote(intrinsic_split[w_idx], entrinsic_split[w_idx], whs_split[w_idx], name_split[w_idx]))

    results = ray.get(ray_worker_ids)
    pc_colored = np.zeros((0, 6))
    for points_all in results:
        pc_colored = np.unique(np.vstack([pc_colored, points_all]), axis=0)
    
    pcd_vector =  o3d.utility.Vector3dVector(pc_colored[:, :3])
    color_vector =  o3d.utility.Vector3dVector(pc_colored[:, 3:])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_vector)
    pcd.colors = o3d.utility.Vector3dVector(color_vector)

    o3d.io.write_point_cloud(os.path.join(args.output_path, "reprojected.ply"), pcd)