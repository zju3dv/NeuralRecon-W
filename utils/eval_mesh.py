import sys
sys.path.insert(1, '.')

import numpy as np
import torch
import yaml
import os
import json
from argparse import ArgumentParser

from utils.eval_utils import \
    visualize_error, nn_correspondance, filtered_sfm, point_crop, bbx_crop, o3d_load, trimesh_load, _compute


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--file_pred', type=str,
                        default='/home/chenxi/code/nerf_pl/results/phototourism/neusw_s1_trans_uncertainty_weight_max-20211021_204722_iter_255000/mesh/extracted_mesh_res_1024_radius_1.0_colored.ply',
                        help='ply file path for prediction')
    parser.add_argument('--file_trgt', type=str,
                        default="/nas/datasets/OpenHeritage3D/pro/brandenburg_gate/bg_sampled_0.05_cropped.ply",
                        help='ply file path for ground truth')
    parser.add_argument('--scene_config_path', type=str,
                        default='/nas/datasets/IMC/phototourism/training_set/brandenburg_gate/config.yaml',
                        help='scene config path') 
    parser.add_argument('--mesh', default=False, action="store_true",
                        help='whether prediction is mesh')
    parser.add_argument('--threshold',  type=str, default="0.1",
                        help='threshold for precision and recall in cm, in order of [start,end,interval]')
    parser.add_argument('--bbx_name', type=str, default='eval_bbx',
                        help='area to eval')

    parser.add_argument('--sfm_path', type=str,
                        help='if set, eval will use sfm points to crop both gt and prediction')   
    parser.add_argument('--track_lenth', type=float,
                        help='track length threshold for sfm points')   
    parser.add_argument('--reproj_error', type=float,
                        help='mean reprojection error threshold for sfm points')   
    parser.add_argument('--voxel_size', type=float,
                        help='voxel size for sfm points to crop point clouds')   

    parser.add_argument('--save_name', type=str,
                        help='visualization save path')  

    return parser.parse_args()               


def eval_mesh(file_pred, file_trgt, scene_config, is_mesh, threshold=.1, bbx_name='eval_bbx', use_o3d=True, save_name="eval"):
    """eval two point clouds

    Args:
        file_pred (str): ply file path for prediction
        file_trgt (str): ply file path for ground truth
        scene_config (str): scene reconstruction path
        is_mesh (bool): whether prediction is a mesh
        threshold (float, optional): distance threshold. Defaults to .1.
        bbx_name (bool, optional): name of the evaluation bounding box

    Returns:
        dict: mesh metrics(dist1, dist2, precision, recall, fscore)
    """

    save_dir = '/'.join(file_pred.split('/')[:-1])
    save_dir = os.path.join(save_dir, "eval_"+save_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"results will save in {save_dir}")

    print("loading eval data...")
    if use_o3d:
        verts_pred, verts_trgt = o3d_load(file_pred, file_trgt, scene_config, is_mesh, bbx_name, save_dir)
    else:
        verts_pred, verts_trgt = trimesh_load(file_pred, file_trgt, scene_config, is_mesh, bbx_name, save_dir)

    if "sfm_path" in scene_config.keys():
        sfm_filtered = filtered_sfm(scene_config["sfm_path"], sfm_to_gt=np.array(scene_config['sfm2gt']), \
                        track_length=scene_config['eval_tl'], reproj_error=scene_config['eval_error'], save_path=f"{save_dir}/sfm_points.ply")
        print(f"filtered points: {sfm_filtered.shape[0]}")

        verts_pred = point_crop(verts_pred, sfm_filtered, scene_config['eval_voxel'], scene_config[bbx_name], batch_size=8, save_path=f"{save_dir}/pred_filtered.ply")
        verts_trgt = point_crop(verts_trgt, sfm_filtered, scene_config['eval_voxel'], scene_config[bbx_name], batch_size=8, save_path=f"{save_dir}/target_filtered.ply")

    print("start evaluating...")
    _, dist1 = nn_correspondance(verts_pred, verts_trgt, use_o3d)
    _, dist2 = nn_correspondance(verts_trgt, verts_pred, use_o3d)

    if not isinstance(threshold, list):
        threshold = [threshold]

    fscores = []
    precs = []
    recals = []
    for i, _ in enumerate(threshold):
        save_path =  os.path.join(save_dir, "visualize", f"{threshold[i]:.2f}")
        os.makedirs(save_path, exist_ok=True)
        # visualize error
        if use_o3d:
            visualize_error(verts_pred, dist2, f"{save_path}/error_pred_precision.ply", threshold[i])
            visualize_error(verts_trgt, dist1, f"{save_path}/error_gt_recal.ply", threshold[i])
        
        metrics = _compute(dist1, dist2, threshold[i])
        
        rslt_file = os.path.join(save_path, 'metrics.json')
        json.dump(metrics, open(rslt_file, 'w'))

        fscores.append(metrics["fscore"])
        precs.append(metrics["prec"])
        recals.append(metrics["recal"])

    all_metrics = {
        "thresholds": threshold,
        "fscores": fscores,
        "precs": precs,
        "recals": recals
    }

    rslt_file = os.path.join(save_dir, 'metrics.json')
    json.dump(all_metrics, open(rslt_file, 'w'))

    print(f"fscores: {fscores}")
    print(f"precs: {precs}")
    print(f"recals: {recals}")

    return metrics


if __name__ == "__main__":
    args = get_opts()
    args.threshold = [float(num.strip()) for num in args.threshold.split(',')]
    args.threshold = list(np.arange(args.threshold[0], args.threshold[1], args.threshold[2]))
    print(f"thresholds to eval: {args.threshold}")

    # read scene config
    with open(args.scene_config_path, "r") as yamlfile:
        scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    if args.sfm_path:
        print(f"crop with sfm in {args.sfm_path}")
        scene_config["sfm_path"] = args.sfm_path
        scene_config['eval_tl'] = args.track_lenth
        scene_config['eval_error'] = args.reproj_error
        scene_config['eval_voxel'] = args.voxel_size
    use_o3d = True

    try:
        import open3d as o3d
    except:
        use_o3d = False
    
    eval_mesh(args.file_pred, args.file_trgt, scene_config, args.mesh, threshold=args.threshold, bbx_name=args.bbx_name, use_o3d=use_o3d, save_name=args.save_name)