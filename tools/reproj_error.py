# 1. find valid points: long track length + low reproduction error
# 2. select one view in a track, find the corresponding point in new pc
# 3. calculate rejection error in other points in the track
import sys
sys.path.insert(1, '.')
import open3d as o3d
import yaml
import imageio
from argparse import ArgumentParser
from utils.colmap_utils import \
    read_cameras_binary, read_images_binary, read_points3d_binary
import os
import numpy as np
import open3d as o3d
import glob
import pandas as pd
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

def get_gt_point(pcd, cam_pose, cam_intrinsic, track_pts2D):
    '''
    args:
        pcd: tensor (num_points, 3)
        cam_pose: tensor (num_points, 4, 4)
        cam_intrinsic: tensor (num_points, 3, 3)
    returns:
        pixel_depth: np.array (num_pixels, 3) component: (pixel_x, pixel_y, depth)
    '''
    num_gt = pcd.shape[0]
    pcd = torch.cat([pcd, torch.ones(num_gt, 1).cuda()], -1) # (batch_size, num_points, 4)
    pcd_in_cam = torch.linalg.inv(cam_pose) @ pcd.transpose(0, 1) # (batch_size, 4, num_points)
    pcd_in_cam[:, 0] /= pcd_in_cam[:, 3]
    pcd_in_cam[:, 1] /= pcd_in_cam[:, 3]
    pcd_in_cam[:, 2] /= pcd_in_cam[:, 3]
    pcd_in_cam = (pcd_in_cam)[:, :3, :]  # (batch_size, 3, num_points)
    pcd_projected = torch.bmm(cam_intrinsic, pcd_in_cam).permute(0, 2, 1) # (batch_size, num_points, 3)

    pcd_projected[:, :, 0] /= pcd_projected[:, :, 2]
    pcd_projected[:, :, 1] /= pcd_projected[:, :, 2]

    # select points from ray
    pc_ray_mask = torch.round(pcd_projected[:, :, :2]) == torch.round(track_pts2D[:, -2:].unsqueeze(1))
    pc_ray_mask = pc_ray_mask[:, :, 0] * pc_ray_mask[:, :, 1] * (pcd_projected[:, :, 2] >= 0)
    if torch.sum(pc_ray_mask) == 0:
        print("Invalid points detected!!")
    pcd_projected[~pc_ray_mask] = torch.max(pcd_projected[pc_ray_mask][:, 2]) + 10

    gt_pts3D_id = torch.argmin(pcd_projected[:, :, 2], dim=1, keepdim=True).squeeze()

    return pcd[gt_pts3D_id, :]

def get_image_id(imdata, data_dir):
    files = sorted(os.listdir(os.path.join(data_dir, 'dense', 'images')))[2:]
    img_path_to_id = {}
    img_id_to_name={}
    for v in imdata.values():
        img_path_to_id[v.name] = v.id
        img_id_to_name[v.id] = v.name
    img_ids = []
    image_paths = {}  # {id: filename}
    for filename in files:
        id_ = img_path_to_id[filename]
        image_paths[id_] = filename
        img_ids += [id_]
    return img_ids, img_id_to_name

def get_entrinsics(imdata, img_ids):
    N_images = len(img_ids)
    entrinsics = np.zeros([N_images, 4, 4])
    entrinsics[:, 3, 3] = 1
    for id, img_id in enumerate(img_ids):
        im = imdata[img_id]
        entrinsics[id, :3, :3] = im.qvec2rotmat()
        entrinsics[id, :3, 3] = im.tvec

    return entrinsics

def get_intrinsic(camdata, img_ids, imdata):
    Ks = {}
    whs = {}
    for id_ in img_ids:
        cam = camdata[imdata[id_].camera_id]
        Ks[id_] = np.array([
            [cam.params[0], 0, cam.params[2]],
            [0, cam.params[1], cam.params[3]],
            [0, 0, 1]
        ], dtype=np.float32)
        whs[id_] = cam.width ,cam.height

    return Ks, whs

def reproject_vis(gt_reprojets_2D, track_pts2D, whs_dict, img_id_to_name):
    """
    take two projections, visualize them on the map
    Args:
        gt_reprojets_2D:
        track_pts2D: N_points, 4(img_id, pixel_id, x, y)

    Returns: None

    """
    img_save_list = torch.unique(track_pts2D[:, 0]).cpu().numpy()
    print("visualize imgs...")
    for img_id in tqdm(img_save_list):
        wh = torch.from_numpy(np.array([whs_dict[img_id][0], whs_dict[img_id][1]]))
        img = torch.zeros([wh[1], wh[0], 3])
        mask = track_pts2D[:, 0] == img_id

        sfm_project = torch.max(torch.min(track_pts2D[mask][:, -2:].long(), wh-1), 0 * wh)
        gt_project = torch.max(torch.min(gt_reprojets_2D[mask][:, :2].long(), wh-1), 0 * wh)

        img[gt_project[:, 1], gt_project[:, 0]] = torch.tensor([0, 255, 0], dtype=img.dtype)
        img[sfm_project[:, 1], sfm_project[:, 0]] = torch.tensor([255, 0, 0], dtype=img.dtype)

        os.makedirs("reprojects", exist_ok=True)
        imageio.imwrite(os.path.join('reprojects', f'{img_id_to_name[img_id]}.png'), img.cpu().numpy().astype(np.uint8))
    return

def image_reproj_error(imdata, pts3d, img_ids, entrinsics_dict, intrinsics_dict):
    pts3d_array = torch.ones(max(pts3d.keys()) + 1, 4).cuda()
    for pts_id, pts in tqdm(pts3d.items()):
        pts3d_array[pts_id, :3] = torch.from_numpy(pts.xyz).cuda()

    img_errors = torch.ones(len(img_ids), 1)
    for id, img_id in tqdm(enumerate(img_ids)):
        img = imdata[img_id]
        entrinsic = torch.from_numpy(entrinsics_dict[img_id]).float().cuda()
        intrinsic = torch.from_numpy(intrinsics_dict[img_id]).float().cuda()
        img_p3d = pts3d_array[img.point3D_ids]
        projected = intrinsic @ entrinsic[:3] @ img_p3d.T
        projected[:2, :] /= projected[2, :].unsqueeze(0)
        img_2d = torch.from_numpy(img.xys)
        img_error = torch.linalg.norm(projected[:2, :].T.cpu() - img_2d, dim=-1)
        img_errors[id] = torch.sum(img_error) / img_error.size()[0]
        # print(f"img_error: {img_errors[id]}")

    return img_errors




def gt_reproject_error(data_dir, gt_pcd_path, sfm_to_gt, reconstuct_path, track_length=200, reproj_error=0.4, batch_size=2, img_reproj_error=300):
    # 0. read data from bin
    # reconstuct_path = 'dense_ws_filtered_tkl200_mrep.5'
    # reconstuct_path = 'dense/sparse'
    imdata = read_images_binary(os.path.join(data_dir, reconstuct_path, 'images.bin'))
    camdata = read_cameras_binary(os.path.join(data_dir, reconstuct_path, 'cameras.bin'))
    pts3d = read_points3d_binary(os.path.join(data_dir, reconstuct_path, 'points3D.bin'))

    gt_pcd = torch.from_numpy(np.array(o3d.io.read_point_cloud(gt_pcd_path).points)).float().cuda()
    img_ids, img_id_to_name = get_image_id(imdata, data_dir)
    entrinsics = get_entrinsics(imdata, img_ids)
    entrinsics_dict = {id_: entrinsics[i] for i, id_ in enumerate(img_ids)}
    cam2gt = sfm_to_gt @ np.linalg.inv(entrinsics)  # (N_images, 4, 4)
    cam2gt_dict = {id_: cam2gt[i] for i, id_ in enumerate(img_ids)}
    intrinsics_dict, whs_dict = get_intrinsic(camdata, img_ids, imdata)

    image_error = image_reproj_error(imdata, pts3d, img_ids, entrinsics_dict, intrinsics_dict)
    img_ids_filtered = [id_ for i, id_ in enumerate(img_ids) if image_error[i] < img_reproj_error]
    print(f"selected {len(img_ids_filtered)} view for testing.")

    # 1. find valid points: long track length + low reproduction error
    pts3d_filtered = []
    track_lengths = []
    track_pts2D = []
    track_cameras = []
    track_pts3D_id = []
    track_cam2gt = []
    track_intrinsics = []
    track_pts3D = []
    for pts_id, pts in pts3d.items():
        if len(pts.point2D_idxs) > track_length and pts.error < reproj_error:
            point2D_xy = []
            point2D_img_id = []
            point2D_track_cameras = []
            point2D_track_cam2gt = []
            point2D_track_intrinsics = []

            # parse point2D and cameras corrdinates from
            for image_id, point2D_id in zip(pts.image_ids, pts.point2D_idxs):
                if image_id not in img_ids_filtered:
                    continue
                point2D_xy.append(imdata[image_id].xys[point2D_id])
                w2c_sfm = intrinsics_dict[image_id] @ (entrinsics_dict[image_id] @ np.linalg.inv(sfm_to_gt))[:3]
                point2D_track_cameras.append(w2c_sfm)
                point2D_track_cam2gt.append(cam2gt_dict[image_id])
                point2D_img_id.append(np.array([image_id, point2D_id]))
                point2D_track_intrinsics.append(intrinsics_dict[image_id])
            if len(point2D_img_id) > 0:
                track_lengths.append(len(point2D_img_id))
                track_pts2D.append(np.hstack([np.vstack(point2D_img_id), np.vstack(point2D_xy)]))

                track_cameras.append(np.stack(point2D_track_cameras, axis=0))
                track_cam2gt.append(np.stack(point2D_track_cam2gt, axis=0))
                track_intrinsics.append(np.stack(point2D_track_intrinsics, axis=0))

                pts3d_filtered.append(pts)
                track_pts3D_id.append(pts_id)
                track_pts3D.append(pts.xyz)

    track_lengths = torch.from_numpy(np.vstack(track_lengths)).squeeze()
    track_pts2D = torch.from_numpy(np.vstack(track_pts2D)) # all_tracts, 4(img_id, pixel_id, point2D_idxs, pixel_id)
    track_cameras = torch.from_numpy(np.vstack(track_cameras))
    track_pts3D_id = torch.from_numpy(np.vstack(track_pts3D_id))
    track_pts3D_id = torch.repeat_interleave(track_pts3D_id.squeeze(), track_lengths.squeeze())
    track_cam2gt = torch.from_numpy(np.vstack(track_cam2gt))
    track_intrinsics = torch.from_numpy(np.vstack(track_intrinsics))
    track_reference_points = torch.roll(torch.cumsum(track_lengths, dim=0), shifts=1)
    track_reference_points[0] = 0
    track_pts3D = torch.from_numpy(np.vstack(track_pts3D))

    # tracks -> point3D_id, tracks_length
    #        -> point3D_id, img_id, track_x, track_y -- errors views
    #        -> point3D_id, img_id, x, y  -- element of reference 2D point

    gt_reprojets_3D = torch.ones([track_reference_points.size()[0], 4])
    for ind in tqdm(range(0, track_reference_points.size()[0], batch_size)):
        gt_reprojets_3D[ind:ind+batch_size] = get_gt_point(gt_pcd,
                                                           track_cam2gt[track_reference_points][ind:ind+batch_size].float().cuda(),
                                                           track_intrinsics[track_reference_points][ind:ind+batch_size].float().cuda(),
                                                           track_pts2D[track_reference_points][ind:ind+batch_size].float().cuda()) # point3D_id, img_id, gt_x, gt_y, gt_z
    os.makedirs(f"samples/reproject", exist_ok=True)
    sfm_pcd = o3d.geometry.PointCloud()
    sfm_pcd.points = o3d.utility.Vector3dVector(track_pts3D.detach().cpu().numpy())
    o3d.io.write_point_cloud(f"samples/reproject/colmap_sfm.ply", sfm_pcd)

    gt_proj = (np.linalg.inv(sfm_to_gt)[:3] @ gt_reprojets_3D.detach().cpu().numpy().T).T
    gt_pcd = o3d.geometry.PointCloud()
    gt_pcd.points = o3d.utility.Vector3dVector(gt_proj)
    o3d.io.write_point_cloud(f"samples/reproject/gt.ply", gt_pcd)

    gt_reprojets_3D = torch.repeat_interleave(gt_reprojets_3D, track_lengths, dim=0)
    gt_reprojets_2D = torch.bmm(track_cameras.float(), gt_reprojets_3D.unsqueeze(-1)).squeeze(-1)
    gt_reprojets_2D[:, :2] /= gt_reprojets_2D[:, 2].unsqueeze(1)
    errors = torch.linalg.norm(gt_reprojets_2D[:, :2] - track_pts2D[:, -2:], dim=-1)

    plt.plot(np.arange(0, errors.size()[0]), errors.squeeze().detach().cpu().numpy())
    plt.savefig('reproj_error.png')

    # errors = errors[errors < 100]
    loss = torch.sum(errors) / errors.size()[0]
    print(f"avg re-projection error {loss}, {errors.size()[0]}/{gt_reprojets_2D.size()[0]}")

    reproject_vis(gt_reprojets_2D, track_pts2D, whs_dict=whs_dict, img_id_to_name=img_id_to_name)

    return loss

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='/nas/datasets/IMC/phototourism/training_set/brandenburg_gate',
                        help='path to data folder')
    parser.add_argument('--gt_pcd_path', type=str,
                        default="/nas/datasets/OpenHeritage3D/pro/brandenburg_gate/bg_sampled_0.01_cropped.ply",
                        help='target point cloud')
    parser.add_argument('--reconstuct_path', type=str,
                        default="dense/sparse",
                        help='reconstruction work space')
    parser.add_argument('--track_length', type=int, default='200',
                    help='track length threshold')
    parser.add_argument('--reproj_error', type=float, default='0.4',
                    help='reproj error threshold')
    parser.add_argument('--batch_size', type=int, default='2',
                    help='batch size when projectin point cloud')
    parser.add_argument('--img_reproj_error', type=float, default='300',
                    help='filter out image with reproj error above this threshold')

    return parser.parse_args()       

if __name__ == "__main__":
    args = get_opts()

    # read scene config
    with open(os.path.join(args.data_dir, 'config.yaml'), "r") as yamlfile:
        scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    gt_reproject_error(args.data_dir, args.gt_pcd_path, np.array(scene_config['sfm2gt']), args.reconstuct_path, args.track_length, args.reproj_error, args.batch_size, args.img_reproj_error)