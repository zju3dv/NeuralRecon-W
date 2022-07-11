import sys
sys.path.insert(1, '.')
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os

from argparse import ArgumentParser

from collections import defaultdict
import numpy as np

from lightning_modules.neuconw_system import NeuconWSystem
from models.nerf import NeRF, PosEmbedding
from config.defaults import get_cfg_defaults
from datasets import dataset_dict

from utils import load_ckpt
from utils.comm import *
from utils.visualization import extract_mesh

from tools.prepare_data.generate_voxel import convert_to_dense, gen_octree_from_sfm
import yaml

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--cfg_path', type=str, help='config path')
    parser.add_argument('--dataset_name', type=str, default='phototourism',
                        choices=['blender', 'phototourism'],
                        help='which dataset to validate')
    parser.add_argument('--eval_level', type=int, default=-1,
                        help='level og precision')
    parser.add_argument('--mesh_size', type=int, default=128,
                        help='resolution of mesh, (N, N, N)')
    parser.add_argument('--mesh_origin', type=str, default="0, 0, 0",
                        help='origin of mesh, (x, y, z)')
    parser.add_argument('--mesh_radius', type=float, default=1.0,
                        help='radius pf mesh')
    parser.add_argument('--vertex_color', default=False, action="store_true",
                        help='whether add color to mesh')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')

    parser.add_argument('--chunk', type=int, default=16384,
                        help='chunk size to split the input to avoid OOM')

    parser.add_argument('--chunk_rgb', type=int, default=256,
                        help='chunk size to split when evaluating color net for mesh vertex colors')

    parser.add_argument('--ckpt_path', type=str, default='ckpts/brandenburg_scale1_nerfw/epoch=1.ckpt',
                        help='pretrained checkpoint path to load')

    parser.add_argument('--local_rank',
                        default=0,
                        type=int,
                        help='node rank for distributed training')

    return parser.parse_args()

def gen_grid_spc(scene_config, data_path, eval_level, device=0):
    # decompose voxel config
    min_track_length = scene_config['min_track_length']
    voxel_size = scene_config['voxel_size']
    
    # obtain training octree
    octree, octree_origin, octree_scale, octree_level = gen_octree_from_sfm(data_path, min_track_length, voxel_size, device=device)

    # upsample octree
    dense = convert_to_dense(octree, octree_level).cpu()
    low_dim = dense.size()[0]

    # dense to sparse
    sparse_ind = torch.nonzero(dense > 0) # n, 3
    sparse_num = sparse_ind.size()[0]

    # upsample
    up_level = eval_level - octree_level
    up_times = 2 ** up_level

    eval_dim = int(low_dim * (2 ** up_level))
    print(f"evaluation dim: {eval_level}, upsampled {up_times} times, original dim {low_dim}, sparse num {sparse_num}")

    sparse_ind_up = sparse_ind.repeat_interleave(up_times ** 3, dim=0) * up_times
    up_kernel = torch.arange(0, up_times, 1)
    up_kernal = torch.stack(torch.meshgrid(up_kernel, up_kernel, up_kernel), dim=-1).reshape(-1, 3) # up_times**3, 3
    expand_kernal = up_kernal.repeat([sparse_num, 1]) # sparse_num * up_times**3, 3

    sparse_ind_up = sparse_ind_up + expand_kernal

    # to sfm coordinate
    eval_voxel_size = 2 / (2 ** eval_level) * octree_scale
    vol_origin = octree_origin - octree_scale 

    xyz_sfm = sparse_ind_up * eval_voxel_size + vol_origin

    sparse_data = {}
    sparse_data['sparse_vol'] = xyz_sfm
    sparse_data['voxel_size'] = eval_voxel_size
    sparse_data['dim'] = eval_dim
    sparse_data['vol_origin'] = vol_origin

    return sparse_data

if __name__ == "__main__":
    args = get_opts()
    config = get_cfg_defaults()
    config.merge_from_file(args.cfg_path)
    scene_config_path = os.path.join(config.DATASET.ROOT_DIR, "config.yaml")
    with open(scene_config_path, "r") as yamlfile:
        scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
    # resolve mesh origin
    args.mesh_origin = [float(corr.strip()) for corr in args.mesh_origin.split(',')]

    save_name = '_'.join(args.ckpt_path.split('/')[-2:]).replace('.ckpt', '')
    dir_name = f'results/{args.dataset_name}/{save_name}/mesh'
    os.makedirs(dir_name, exist_ok=True)
    print(f"result saved in {dir_name}")

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    print('number of gpus: {}'.format(num_gpus))
    args.DISTRIBUTED = num_gpus > 0

    if args.DISTRIBUTED:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    ckpt_path = args.ckpt_path
    neuconw_sys = NeuconWSystem(args, config, None)
    load_ckpt(neuconw_sys.embedding_a, ckpt_path, model_name='embedding_a')
    load_ckpt(neuconw_sys.neuconw, ckpt_path, model_name='neuconw')
    load_ckpt(neuconw_sys.nerf, ckpt_path, model_name='nerf')

    models_to_ddp = [neuconw_sys]
    for m in models_to_ddp:
        m.eval()
        m.cuda()

    if args.DISTRIBUTED:
        for m in models_to_ddp:
            m = DistributedDataParallel(
                m, device_ids=[args.local_rank], output_device=args.local_rank,
                # this should be removed if we update BatchNorm stats
                broadcast_buffers=False,
                find_unused_parameters=True
            )

    if args.eval_level > 0:
        sparse_data = gen_grid_spc(scene_config, config.DATASET.ROOT_DIR, args.eval_level, device=dist.get_rank())
    else:
        sparse_data = None
    mesh = extract_mesh(
        args.mesh_size, args.chunk, scene_config['radius'], scene_config['origin'],
        origin=args.mesh_origin, radius=args.mesh_radius, with_color=args.vertex_color,
        embedding_a=neuconw_sys.embedding_a((torch.ones(1, device=neuconw_sys.device) * 1123).long()),
        chunk_rgb=args.chunk_rgb, sparse_data=sparse_data, renderer = neuconw_sys.renderer)
    colored = '_colored' if args.vertex_color else ''
    if dist.get_rank() == 0:
        print("Saving mesh.....")
        if args.eval_level > 0:
            mesh.export(
                os.path.join(dir_name, f'extracted_mesh_level_{args.eval_level}{colored}.ply'))
        else:
            mesh.export(
                os.path.join(dir_name, f'extracted_mesh_res_{args.mesh_size}_radius_{args.mesh_radius}{colored}.ply'))
        print("Done!")
