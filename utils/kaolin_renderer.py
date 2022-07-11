from tools.prepare_data.generate_voxel import gen_octree, get_near_far, octree_to_spc
import torch 
from datasets.ray_utils import get_ray_directions, get_rays
import kaolin.ops.spc as spc_ops
from kornia import create_meshgrid
import numpy as np
from tqdm import tqdm
import os
import open3d as o3d

class kaolin_renderer():
    def __init__(self, data_path, pts, voxel_size, device=0, visualize=False):
        self.device = device
        self.visualize = visualize
        self.pts = pts
        self.octree, self.origin, self.scale, self.level \
            = gen_octree(data_path, pts, voxel_size, visualize=self.visualize, device=self.device, expand=0, in_sfm=False)

        self.spc_data = {}
        points, pyramid, prefix = octree_to_spc(self.octree)
        self.spc_data["points"] = points
        self.spc_data["pyramid"] = pyramid
        self.spc_data["prefix"] = prefix

        # self.origin = torch.from_numpy(self.origin).to(self.device).float()
        self.counter=0

        self.pt2vid = self.vertex_table(pyramid, points)
        self.voxel_num = pyramid[0, self.level]
        self.base_ind = pyramid[1, self.level]


    def gen_rays(self, height, width, intrinsic, pose):
        grid = create_meshgrid(height, width, normalized_coordinates=False, device=self.device)[0]
        i, j = grid.unbind(-1)
        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        directions = \
            torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], dim=-1) # (H, W, 3)

        # Rotate ray directions from camera coordinate to the world coordinate
        rays_d = directions @ pose[:, :3].T # (H, W, 3)
        dir_norm = torch.norm(rays_d, dim=-1, keepdim=True)
        rays_d = rays_d / dir_norm

        rays_o = pose[:, 3].expand(rays_d.shape) # (H, W, 3)

        rays_d = rays_d.reshape(-1, 3)
        rays_o = rays_o.reshape(-1, 3)
        dir_norm = dir_norm.reshape(-1, 1)

        return rays_o, rays_d, dir_norm

    def get_index(self, pids, batch_size=8):
        vids = torch.nonzero(torch.from_numpy(pids)).reshape(-1).long().cuda()
        self.pt2vid = self.pt2vid.cuda()
        # vids = torch.unique(pids[pids >= 0], dim=0).long()

        # input_pts = (input_pts - self.origin) / self.scale
        # res = 2 ** self.level
        # quantized_input = torch.unique(torch.floor(res * (torch.from_numpy(input_pts) + 1.0) / 2.0).short(), dim=0)

        # points = self.spc_data["points"]
        # pyramid = self.spc_data["pyramid"]

        # points_mt = spc_ops.points_to_morton(points[vids]).sort()[0]
        # inputs_mt = spc_ops.points_to_morton(quantized_input.cuda()).sort()[0]
        # reproj_error = torch.abs(points_mt - inputs_mt).float()

        # print("reproj_error: ", torch.mean(reproj_error))
        print(vids.size(), self.pt2vid.size())

        indices = []
        _ind = torch.arange(0, self.pt2vid.size()[0]).long()
        all_mask = torch.zeros(self.pt2vid.size()[0], dtype=torch.bool)
        for i in tqdm(range(0, vids.size()[0], batch_size)):
            mask_sumed = torch.sum((vids[i:i+batch_size, None] - self.pt2vid[None, :]) == 0, axis=0) > 0
            all_mask = all_mask | mask_sumed.cpu()
        indices = _ind[all_mask]

        return indices.numpy()

    def vertex_table(self,pyramid, points):
        points_normalized = (self.pts - self.origin) / self.scale
        mask = np.prod((points_normalized > -1), axis=-1, dtype=bool) & np.prod((points_normalized < 1), axis=-1, dtype=bool)

        points_normalized = torch.from_numpy(points_normalized[mask]).to(self.device)
        res = 2 ** self.level
        quantized_pc = res * (points_normalized + 1.0) / 2.0
        quantized_pc = torch.floor(quantized_pc[mask]).short()

        mask = torch.prod((quantized_pc >= 0), dim=-1, dtype=bool) & torch.prod((quantized_pc <= res - 1), dim=-1, dtype=bool)
        quantized_pc = quantized_pc[mask]

        unique_pts, inv_ind = torch.unique(quantized_pc, dim=0, return_inverse=True)
        sort_inds = spc_ops.points_to_morton(unique_pts).sort()[1]
        unique_pts = unique_pts[sort_inds]

        level_pointx = points[pyramid[1, self.level]:pyramid[1, self.level+1], :3]

        assert torch.mean(torch.abs(level_pointx - unique_pts).float()) == 0, "error in creating index. spc corrupted"

        voxel_id = torch.ones_like(torch.from_numpy(self.pts[:, 1]).long()) * -10
        sort_inv = torch.arange(0, sort_inds.size()[0])
        sort_inv[sort_inds] = torch.arange(0, sort_inds.size()[0])
        voxel_id[mask.cpu()] = sort_inv[inv_ind].cpu()

        return voxel_id


    @torch.no_grad()
    def __call__(self, height, width, intrinsic, pose):
        intrinsic = torch.from_numpy(intrinsic).to(self.device).float()
        pose = torch.from_numpy(pose).to(self.device).float()

        rays_o, rays_d, dir_norm = self.gen_rays(height, width, intrinsic, pose[:3])

        depth_all = []
        pid_all = []
        # todo: figure out why chunck size greater or equal than 1768500 will result in error intersection
        # use 1000000 as a threshold just to be safe
        chunk_size = min(rays_o.size()[0], 1000000)
        for i in range(0, rays_o.size()[0], chunk_size):
            # generate near far from spc
            voxel_near_sfm, _, pid = get_near_far(rays_o[i:i + chunk_size], rays_d[i:i + chunk_size], \
                self.octree, torch.from_numpy(self.origin).to(self.device).float(), self.scale, self.level, self.spc_data, \
                visualize=self.visualize, ind=f"filter_{self.counter}", return_pts=True)
            voxel_near_sfm /= dir_norm[i:i + chunk_size]
            depth_all.append(voxel_near_sfm.cpu())
            pid_all.append(pid.cpu())
        
        depth = torch.cat(depth_all, dim=0).reshape(height, width).numpy()
        self.pids = (torch.cat(pid_all, dim=0).reshape(height, width) - self.base_ind).long().numpy() 

        # generate pesudo color
        base_color = np.array([0, 0, 255])
        color = np.tile(base_color, (height, width, 1))
        color[depth <= 0] = 0

        self.counter += 1

        return color, depth+0.02