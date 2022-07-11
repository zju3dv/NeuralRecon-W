import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
import torch
import trimesh
from skimage import measure
from tqdm import tqdm
from utils.comm import get_world_size, get_rank
import torch.distributed as dist
import torch.distributed

def visualize_depth(depth, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = depth.cpu().numpy()
    x = np.nan_to_num(x) # change nan to 0
    mi = np.min(x) # get minimum depth
    ma = np.max(x)
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = Image.fromarray(cv2.applyColorMap(x, cmap))
    x_ = T.ToTensor()(x_) # (3, H, W)
    return x_

def get_local_split(data, world_size, local_rank):
    if data.shape[0] % world_size != 0:
        xyz_padded = torch.cat((data, \
            torch.zeros((data.shape[0] // world_size + 1) * world_size - data.shape[0], data.shape[1]).to(data.device)), 0)
    else:
        xyz_padded = data
    local_split_length = xyz_padded.shape[0] // world_size
    local_data = xyz_padded[local_rank*local_split_length: (local_rank+1)*local_split_length]
    return local_data

def extract_mesh(dim, chunk, scene_radius, scene_origin, origin=None, radius=1.0,
                       with_color=False, embedding_a=None, chunk_rgb=256, sparse_data=None, renderer=None):
    if origin is None:
        origin = [0, 0, 0]
    sphere_origin = np.array(origin)
    if sparse_data is None:
        vol_origin = sphere_origin - radius
        voxel_size = 2 * radius / (dim - 1)

        x = torch.linspace(sphere_origin[0] - radius, sphere_origin[0] + radius, dim)
        y = torch.linspace(sphere_origin[1] - radius, sphere_origin[1] + radius, dim)
        z = torch.linspace(sphere_origin[2] - radius, sphere_origin[2] + radius, dim)

        xyz_ = torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3)
    else:
        sparse_vol = sparse_data['sparse_vol'].float()
        voxel_size_sfm = sparse_data['voxel_size']
        dim = sparse_data['dim']
        vol_origin_sfm = sparse_data['vol_origin']

        ind = torch.round((sparse_vol - vol_origin_sfm) / voxel_size_sfm).long()
        scene_origin = torch.from_numpy(np.array(scene_origin)).float()
        vol_origin_sfm = torch.from_numpy(np.array(vol_origin_sfm)).float()
        
        # to training coordinates
        xyz_ = (sparse_vol - scene_origin) / scene_radius
        vol_origin = (vol_origin_sfm - scene_origin) / scene_radius

        voxel_size = voxel_size_sfm / scene_radius

    # multi-gpu 
    world_size = get_world_size()
    local_rank = get_rank()
    local_xyz_ = get_local_split(xyz_, world_size, local_rank)

    if local_rank == 0:
        print("start evaluating sdf...")
    with torch.no_grad():
        B = local_xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, chunk), disable=local_rank!=0):
            new_sdf = renderer.sdf(local_xyz_[i:i + chunk].reshape(-1, 1, 3).cuda())
            out_chunks += [new_sdf.detach().cpu()]
        sdf = torch.cat(out_chunks, dim=0)
    if world_size > 1:
        sdf_gathered = [torch.zeros(B, dtype=torch.float, device=local_rank) for _ in range(world_size)]
        dist.all_gather(sdf_gathered, sdf.cuda())
    else:
        sdf_gathered = [sdf]
        
    if local_rank == 0:
        sdf_ = torch.cat(sdf_gathered, 0).cpu().numpy()
        sdf = sdf_[:xyz_.shape[0]]
        print(f'max sdf: {np.max(sdf.reshape(-1))}, min sdf: {np.min(sdf.reshape(-1))}', np.max(sdf.reshape(-1)))
        print(f'start marching cubes')

        if sparse_data is None:
            sdf = sdf.reshape((dim, dim, dim))
            mask_dense = None
        else:
            sdf_dense = torch.ones(dim, dim, dim).float().numpy()
            sdf_dense[ind[:, 0], ind[:, 1], ind[:, 2]] = sdf.reshape(-1)
            vol_origin = vol_origin.numpy()
            scene_origin = scene_origin.numpy()
            sdf = sdf_dense.reshape((dim, dim, dim))
            
            mask_dense = torch.zeros(dim, dim, dim).float().bool()
            mask_dense[ind[:, 0], ind[:, 1], ind[:, 2]] = True
            
            # A points is valid if and only if all it's 8 corners are valid
            mask_dense = mask_dense & \
                torch.roll(mask_dense, shifts=1, dims=0) & torch.roll(mask_dense, shifts=1, dims=1) & torch.roll(mask_dense, shifts=1, dims=2) & \
                torch.roll(mask_dense, shifts=[1,1], dims=[0,1]) & torch.roll(mask_dense, shifts=[1,1], dims=[0,2]) & torch.roll(mask_dense, shifts=[1,1], dims=[1,2]) & \
                torch.roll(mask_dense, shifts=[1,1,1], dims=[0,1,2]) 
                
            mask_dense = mask_dense.numpy() > 0

        verts, faces, norms, vals = measure.marching_cubes(sdf, level=0, mask=mask_dense)
        if local_rank == 0:
            print("radius: ", (np.max(verts[0]) - np.min(verts[0])) * voxel_size)

        verts = verts * voxel_size + vol_origin  # voxel grid coordinates to training coordinates
        verts_w = verts * scene_radius + scene_origin # training coordinates to world coordinates
        
        if not with_color:
            mesh = trimesh.Trimesh(vertices=verts_w, faces=faces, vertex_normals=norms)

    if with_color:
        if local_rank == 0:
            print("generating rgb ...")
            B = verts.shape[0]
            verts = torch.from_numpy(verts).float() # (B, 3)
            rays_d = torch.zeros_like(verts).detach().cpu()

            rays_d[:,  2] = 1   # (B, 3)
            a_embedded = embedding_a.repeat(B, 1).detach().cpu()  # (B, N_a)
            all_inputs = torch.cat([verts, rays_d, a_embedded], 1)
            objects = [all_inputs]
        else:
            objects = [None]

        dist.broadcast_object_list(objects, src=0)
        all_inputs = objects[0]
        local_inputs = get_local_split(all_inputs, world_size, local_rank).unsqueeze(1)
        out_chunks = []
        for i in tqdm(range(0, local_inputs.shape[0], chunk_rgb), disable=local_rank!=0):
            static_rgb = renderer.rgb(local_inputs[i:i + chunk_rgb][:, :, :3].cuda(), local_inputs[i:i + chunk_rgb][:, :, 3:6].cuda(),
                local_inputs[i:i + chunk_rgb][:, :, 6:6+embedding_a.size()[1]].cuda())
            out_chunks += [static_rgb.reshape(-1, 3).detach().cpu()]
        rgbs = torch.cat(out_chunks, 0) * 255
        
        B = local_inputs.shape[0]
        rgbs_gathered = [torch.zeros(B, 3, dtype=torch.float, device=local_rank) for _ in range(world_size)]
        dist.all_gather(rgbs_gathered, rgbs.cuda())
    
        if local_rank == 0:
            rgbs_ =  torch.cat(rgbs_gathered, 0).cpu().numpy()
            rgbs = rgbs_[:all_inputs.shape[0]]
            vertex_colors = trimesh.visual.color.VertexColor(colors=rgbs, obj=None).vertex_colors
            mesh = trimesh.Trimesh(vertices=verts_w, faces=faces, vertex_normals=norms, vertex_colors=vertex_colors)

    if local_rank == 0:
        return mesh