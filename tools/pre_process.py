# this folder is dedicated to transform original colmap data into our data

"""
Input data structure:
- demo
 - colmap
   - 0
 - images
   - VID00
   - VID01
   ...
   
Output data structure:
- demo
  - VID00
    - config.yaml
    - semantic maps
    - dense
      - sparse
      - images
  - VID01
"""

import sys
sys.path.insert(1, '.')
from utils.colmap_utils import read_images_binary, write_images_binary, read_points3d_binary
import numpy as np
import tqdm as tqdm
import os
import subprocess
import shutil
import yaml
from argparse import ArgumentParser

def bbx_selection(sfm_points):
    """auto scene origin and radius selection

    Args:
        sfm_points (numpy.array): sfm points
    returns:
        bbx (numpy.array): (3 x 2), [min. max]
        origin (numpy.array): (1, 3), [x,y,z]
    """
    bbx = np.concatenate([np.percentile(sfm_points, q=4.00, axis=0).reshape(1,3), np.percentile(sfm_points, q=96.0, axis=0).reshape(1,3)], axis=0)
    origin = np.mean(bbx, axis=0)
    return bbx, origin


def colmap_overwrite(colmap_path, image_list):
    """auto scene origin and radius selection

    Args:
        colmap_path (str): colmap sparse folder path
        image_list (List): image folder list
    """
    image_path =  os.path.join(colmap_path, 'images.bin')
    images = read_images_binary(image_path)
    images_new = {}
    for key in images.keys():
        colmap_folder = images[key].name.rsplit('/', 1)[-2] if len(images[key].name.rsplit('/', 1)) > 1 else ''
        image_name = images[key].name.rsplit('/', 1)[-1]
        if colmap_folder != '':
            if colmap_folder in image_list:
                images_new[key] = images[key]
                new_name = f"{colmap_folder}_{image_name}"
                images_new[key] = images_new[key]._replace(name=new_name)
        else:
            images_new[key] = images[key]

    write_images_binary(images_new, image_path)
    
    
def copy_file(src, dest, group_list, colmap_dir, img_dir):
    """copy file make as our

    Args:
        src (src): source folder
        dest (src): destination folder
    """
    
    scene_name = src.split('/')[-1]
    dest = os.path.join(dest, scene_name)
    
    
    undistort_path = os.path.join(dest, 'undistort')
    os.makedirs(undistort_path, exist_ok=True)
    undistort_img_path = os.path.join(dest, 'undistort/images')
    
    
    origin_image_path = os.path.join(src, img_dir)
    colmap_path = os.path.join(src, colmap_dir)
    # image undistorter
    subprocess.call([
        'colmap',
        "image_undistorter",
        "--image_path", origin_image_path,
        "--input_path", colmap_path,
        "--output_path",  undistort_path,
        "--output_type",'COLMAP'
    ])
    
    point_path =  os.path.join(colmap_path, 'points3D.bin')
    points_3d = read_points3d_binary(point_path)
    points_ori = []
    for id, p in points_3d.items():
        if p.point2D_idxs.shape[0] > 2:
            points_ori.append(p.xyz)
    points = np.array(points_ori)
    
    sfm_points = points
    
    for group in group_list:
        group_name = '_'.join(group)
        group_path = os.path.join(dest, group_name if group_name != '' else 'split_0' )
        dest_colmap = os.path.join(group_path, 'dense', 'sparse')
        
        # copy and rewrite colmap 
        shutil.copytree(os.path.join(undistort_path, 'sparse'), dest_colmap, dirs_exist_ok=True)
        colmap_overwrite(dest_colmap, group)
        
        # write config
        generate_config('_'.join([scene_name, group_name]) if group_name != '' else scene_name, group_path, sfm_points)
        
        # write images
        dest_image_path = os.path.join(group_path, 'dense', 'images')
        for image_folder_name in group:
            os.makedirs(dest_image_path, exist_ok=True)
            orgin_img_folder = os.path.join(undistort_img_path, image_folder_name)
            ori_imag_names = os.listdir(orgin_img_folder)
            new_img_names = ['_'.join([group_name, _img_name]) if group_name != '' else _img_name for _img_name in ori_imag_names]
            
            for ori_img, dest_img in zip(ori_imag_names, new_img_names):
                shutil.copy(os.path.join(orgin_img_folder, ori_img), os.path.join(dest_image_path, dest_img))
                
def generate_config(scene_name, save_path, sfm_points):
    # genrate origin and bbx
    bbx, origin= bbx_selection(sfm_points)
    sfm2gt = np.eye(4)
    scale = (np.max(bbx[1] - bbx[0]) / 2).item()
    level = 5
    
    print(origin.tolist())
    config_dict = {
        'name': scene_name,
        'origin': origin.tolist(),
        'radius': scale * 2,
        'eval_bbx': bbx.tolist(),
        'sfm2gt': sfm2gt.tolist(),
        'min_track_length': 2,
        'eval_bbx_detail': bbx.tolist(),
        'voxel_size': 2 / (2 ** level) * scale - 0.0001
        }
    
    print(f"config: {config_dict}")
    
    config_path = os.path.join(save_path, 'config.yaml')
    with open(config_path, 'w') as outfile:
        yaml.dump(config_dict, outfile, default_flow_style=False, sort_keys=False)
        

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--src', type=str,
                        default='/nas/dataset/static_recon/SLR/scene/two_seq',
                        help='source folder')
    parser.add_argument('--dest', type=str,
                        default="/nas/users/chenxi/recon_data",
                        help='detination path')
    parser.add_argument('--split', type=str,
                        default="plain",
                        help='split sub image folders into split, it can be "plain", "none", or "folder1,folder2#folder4"')
    parser.add_argument('--colmap_dir', type=str,
                        default='sparse/0',
                        help='colmap folder')
    parser.add_argument('--img_dir', type=str,
                        default='images',
                        help='source image folder')
    return parser.parse_args()     

def gen_split(src, split, img_dir):
    if split == 'none':
        return [['']]
    if split == 'plain':
        return [[group] for group in os.listdir(os.path.join(src, img_dir))]
    else:
        return [group.split(',') for group in split.split('#')]


if __name__ == "__main__":
    args = get_opts()
    group_list = gen_split(args.src, args.split, args.img_dir)
    print("images groups: ", group_list)
    copy_file(args.src, args.dest, group_list, args.colmap_dir, args.img_dir)