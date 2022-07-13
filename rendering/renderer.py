import open3d as o3d
import torch
import torch.nn.functional as F
import numpy as np
from datasets.mask_utils import get_label_id_mapping
import os
from tools.prepare_data.generate_voxel import (
    get_near_far,
    gen_octree_from_sfm,
    octree_to_spc,
)
import yaml


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    device = weights.device
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(
            0.0 + 0.5 / n_samples, 1.0 - 0.5 / n_samples, steps=n_samples, device=device
        )
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1], device=device) + [n_samples])

    # Invert CDF
    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples


class NeuconWRenderer:
    def __init__(
        self,
        nerf,
        neuconw,
        embeddings,
        n_samples,
        n_importance,
        n_outside,
        up_sample_steps,
        perturb,
        origin,
        radius,
        s_val_base=0,
        spc_options=None,
        sample_range=None,
        boundary_samples=None,
        nerf_far_override=False,
        render_bg=True,
        trim_sphere=True,
        save_sample=False,
        save_step_sample=False,
        mesh_mask_list=None,
        floor_normal=False,
        depth_loss=False,
        floor_labels=None,
    ):

        self.nerf = nerf
        self.neuconw = neuconw
        self.embeddings = embeddings

        self.n_samples = n_samples
        self.n_importance = n_importance
        self.n_outside = n_outside
        self.up_sample_steps = up_sample_steps
        self.perturb = perturb
        self.s_val_base = s_val_base

        self.boundary_samples = boundary_samples
        self.nerf_far_override = nerf_far_override

        self.octree_data = None
        self.sample_range = sample_range
        self.fine_octree_data = None
        if self.nerf_far_override:
            self.recontruct_path = spc_options["recontruct_path"]
            self.min_track_length = spc_options["min_track_length"]
            self.voxel_size = spc_options["voxel_size"]

        self.sfm_to_gt = np.eye(4)

        # read unit sphere origin and radius from scene config
        scene_config_path = os.path.join(spc_options["recontruct_path"], "config.yaml")
        if os.path.isfile(scene_config_path):
            with open(scene_config_path, "r") as yamlfile:
                scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)
            origin = scene_config["origin"]
            radius = scene_config["radius"]
            self.sfm_to_gt = torch.from_numpy(np.array(scene_config["sfm2gt"]))
        self.origin = torch.from_numpy(np.array(origin))
        self.radius = radius

        self.render_bg = render_bg

        self.floor_normal = floor_normal
        self.floor_labels = floor_labels

        self.depth_loss = depth_loss

        self.save_sample = save_sample
        self.trim_sphere = trim_sphere
        self.mesh_mask_list = mesh_mask_list

        # If saving sampling of each up sample step,
        # don't forget to start ray from center of image to make sure ray intersect some surface.
        # Currently, we will save up-sampled new points in 'samples/new-z'
        # and colored visualization of weight distribution in 'samples/steps'.
        self.save_step_sample = save_step_sample
        self.save_step_itr = 0

        if self.save_sample:
            self.itr = 0
            self.insiders = None
            self.outsiders = None

    def get_octree(self, device):
        octree, scene_origin, scale, level = gen_octree_from_sfm(
            self.recontruct_path, self.min_track_length, self.voxel_size, device=device
        )

        octree_data = {}
        octree_data["octree"] = octree
        octree_data["scene_origin"] = torch.from_numpy(scene_origin).cuda()
        octree_data["scale"] = scale
        octree_data["level"] = level
        spc_data = {}
        points, pyramid, prefix = octree_to_spc(octree)
        spc_data["points"] = points
        spc_data["pyramid"] = pyramid
        spc_data["prefix"] = prefix

        octree_data["spc_data"] = spc_data

        return octree_data

    def render_core_outside(
        self,
        rays_o,
        rays_d,
        z_vals,
        sample_dist,
        nerf,
        background_rgb=None,
        a_embedded=None,
    ):
        """
        Render background
        """
        device = rays_o.device

        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist.expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = (
            rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        )  # batch_size, n_samples, 3

        dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(
            1.0, 1e10
        )
        pts = torch.cat(
            [pts / dis_to_center, 1.0 / dis_to_center], dim=-1
        )  # batch_size, n_samples, 4

        dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

        pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
        dirs = dirs.reshape(-1, 3)

        a_embedded_expand = (
            a_embedded[:, None, :]
            .expand(batch_size, n_samples, -1)
            .reshape(dirs.size()[0], -1)
            if a_embedded is not None
            else None
        )

        density, sampled_color = nerf(pts, dirs, a_embedded_expand)
        alpha = 1.0 - torch.exp(
            -F.softplus(density.reshape(batch_size, n_samples)) * dists
        )
        alpha = alpha.reshape(batch_size, n_samples)
        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [torch.ones([batch_size, 1], device=device), 1.0 - alpha + 1e-7], -1
                ),
                -1,
            )[:, :-1]
        )
        sampled_color = sampled_color.reshape(batch_size, n_samples, 3)
        color = (weights[:, :, None] * sampled_color).sum(dim=1)
        if background_rgb is not None:
            color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

        return {
            "color": color,
            "sampled_color": sampled_color,
            "alpha": alpha,
            "weights": weights,
        }

    def save_samples_step(self, pts, weights, save_name, dir_name="steps"):
        pts_world = (pts * self.radius).view(-1, 3) + self.origin

        # set point cloud color
        ## weight[0.0, 0.1]   --   light blue
        ## weight [0.1, 0.9]  --   dark blue
        ## weight [0.9, 1.0]  --   purple
        colors = torch.zeros_like(pts_world)  # N, 3
        colors[weights < 0.1, :] = torch.tensor(
            [0, 255, 255], dtype=colors.dtype, device=pts_world.device
        )
        colors[(weights > 0.1) & (weights < 0.9), :] = torch.tensor(
            [0, 0, 255], dtype=colors.dtype, device=pts_world.device
        )
        colors[weights > 0.9, :] = torch.tensor(
            [127, 0, 255], dtype=colors.dtype, device=pts_world.device
        )

        print(f"Saving samples at samples/{dir_name}/step_{self.save_step_itr} ...")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world.detach().cpu().numpy())
        pcd.colors = o3d.utility.Vector3dVector(colors.detach().cpu().numpy() / 255.0)
        os.makedirs(f"samples/{dir_name}/step_{self.save_step_itr}", exist_ok=True)
        o3d.io.write_point_cloud(
            f"samples/{dir_name}/step_{self.save_step_itr}/{save_name}.ply", pcd
        )

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s, step):
        """
        Up sampling give a fixed inv_s
        """
        device = sdf.device
        batch_size, n_samples = z_vals.shape
        pts = (
            rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
        )  # n_rays, n_samples, 3
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)
        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------
        prev_cos_val = torch.cat(
            [torch.zeros([batch_size, 1], device=device), cos_val[:, :-1]], dim=-1
        )
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        dist = next_z_vals - prev_z_vals
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        # transient alpha
        # alpha = alpha_s + alpha_t
        weights = (
            alpha
            * torch.cumprod(
                torch.cat(
                    [torch.ones([batch_size, 1], device=device), 1.0 - alpha + 1e-7], -1
                ),
                -1,
            )[:, :-1]
        )

        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()

        if self.save_step_sample:
            # # start points
            # self.save_samples_step(pts[:, :-1][weights < 0.1].view(-1, 3), f"{step}_{0.0}_{0.1}")
            # # end points
            # self.save_samples_step(pts[:, :-1][((weights > 0.1) & (weights < 0.9))].view(-1, 3), f"{step}_{0.1}_{0.9}")
            # self.save_samples_step(pts[:, :-1].reshape(-1, 3), f"{step}_all")

            # colored
            self.save_samples_step(
                pts[:, :-1].reshape(-1, 3),
                weights.reshape(
                    -1,
                ),
                f"{step}_colored",
            )

            pts_new = rays_o[:, None, :] + rays_d[:, None, :] * z_samples[..., :, None]
            self.save_samples_step(
                pts_new.reshape(-1, 3),
                torch.zeros_like(z_samples).reshape(
                    -1,
                ),
                f"{step}",
                dir_name="new_z",
            )
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        if not last:
            _n_rays, _n_samples, _ = pts.size()
            new_sdf = self.sdf(pts).reshape(_n_rays, _n_samples)
            # # print("cat_z_vals ", new_sdf.size(), sdf.size())
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = (
                torch.arange(batch_size)[:, None]
                .expand(batch_size, n_samples + n_importance)
                .reshape(-1)
            )
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        return z_vals, sdf

    def render_depth(self, alphas, z_vals):
        # print('z_vals: {}, alphas: {}'.format(z_vals.size(), alphas.size()))
        transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones([alphas.size()[0], 1], device=alphas.device),
                    1.0 - alphas + 1e-7,
                ],
                -1,
            ),
            -1,
        )[:, :-1]
        weights = alphas * transmittance
        return torch.sum(weights * z_vals, dim=-1)

    def get_near_far_octree(self, octree_data, rays_o, rays_d, near, far):
        # unpack octree
        octree = octree_data["octree"]
        octree_origin = octree_data["scene_origin"].float()
        octree_scale = octree_data["scale"]
        octree_level = octree_data["level"]
        spc_data = octree_data["spc_data"]

        # transfrom origins and direction of rays to sfm coordinate system
        rays_o_sfm = (rays_o * self.radius).view(-1, 3) + self.origin

        # generate near far from spc
        voxel_near_sfm, voxel_far_sfm = get_near_far(
            rays_o_sfm,
            rays_d,
            octree,
            octree_origin,
            octree_scale,
            octree_level,
            spc_data=spc_data,
            visualize=self.save_step_sample,
            ind=self.save_step_itr,
        )

        # transform to unit sphere
        hit_mask = voxel_near_sfm > 0
        voxel_near = (voxel_near_sfm.float().reshape(-1, 1)) / self.radius
        voxel_far = (
            voxel_far_sfm.float().reshape(-1, 1) + self.voxel_size
        ) / self.radius

        near[hit_mask] = voxel_near[hit_mask]
        far[hit_mask] = voxel_far[hit_mask]
        return near, far, hit_mask

    def get_near_far_sdf(self, octree_data, rays_o, rays_d, near, far):
        # generate samples from sdf

        # unpack octree
        octree = octree_data["octree"]
        octree_origin = octree_data["scene_origin"].float()
        octree_scale = octree_data["scale"]
        octree_level = octree_data["level"]
        train_voxel_size = octree_data["voxel_size"]
        train_spc_data = octree_data["spc_data"]

        # transfrom origins and direction of rays to sfm coordinate system
        rays_o_sfm = (rays_o * self.radius).view(-1, 3) + self.origin

        # generate near far from spc
        surface_sfm, _ = get_near_far(
            rays_o_sfm,
            rays_d,
            octree,
            octree_origin,
            octree_scale,
            octree_level,
            spc_data=train_spc_data,
            visualize=self.save_step_sample,
            ind=self.save_step_itr,
        )

        # ray has intersection with current octree
        miss_mask = surface_sfm <= 0

        # generate samples around surface
        voxel_near_sfm = surface_sfm - self.sample_range * train_voxel_size
        voxel_far_sfm = surface_sfm + self.sample_range * train_voxel_size

        # transform to unit sphere
        voxel_near = voxel_near_sfm.float().reshape(-1, 1) / self.radius
        voxel_far = voxel_far_sfm.float().reshape(-1, 1) / self.radius

        voxel_near[miss_mask] = near[miss_mask]
        voxel_far[miss_mask] = far[miss_mask]

        return voxel_near, voxel_far, ~miss_mask

    def sparse_sampler(self, rays_o, rays_d, near, far, perturb):
        """sample on spaese voxel. Including upsample on sparse voxels,
        and uniform sample on inverse depth of original near far,
        Note that input coordinates are scaled to unit sphere

        Args:
            octree_data (tensor byte): [description]
            rays_o (tensor float): [description]
            rays_d (tensor float): [description]
        """

        device = rays_o.device
        batch_size = len(rays_o)

        if self.nerf_far_override:
            if self.octree_data is None:
                self.octree_data = self.get_octree(device)
            near, far, hit_mask_sfm = self.get_near_far_octree(
                self.octree_data, rays_o, rays_d, near, far
            )

        sample_near = near
        sample_far = far

        if self.fine_octree_data is not None:
            # generate near far from sdf
            sample_near, sample_far, hit_mask_sdf = self.get_near_far_sdf(
                self.fine_octree_data, rays_o, rays_d, near, far
            )

        sample_dist = (sample_far - sample_near) / self.n_samples
        z_vals = torch.linspace(0.0, 1.0, self.n_samples, device=device)
        z_vals = sample_near + (sample_far - sample_near) * z_vals[None, :]

        z_vals_outside = None
        if self.render_bg and self.n_outside > 0:
            z_vals_outside = torch.linspace(
                1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside, device=device
            )

        if perturb > 0:
            t_rand = torch.rand([batch_size, 1], device=device) - 0.5
            z_vals = z_vals + (sample_far - sample_near) * t_rand * 2.0 / self.n_samples

            if self.render_bg and self.n_outside > 0:
                mids = 0.5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand(
                    [batch_size, z_vals_outside.shape[-1]], device=device
                )
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.render_bg and self.n_outside > 0:
            z_vals_outside = (
                far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples
            )

        # upsample inside voxel
        if self.n_importance > 0:
            with torch.no_grad():
                pts = (
                    rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                )  # N_rays, N_samples, 3
                sdf = self.sdf(pts).reshape(batch_size, self.n_samples)
                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(
                        rays_o,
                        rays_d,
                        z_vals,
                        sdf,
                        self.n_importance // self.up_sample_steps,
                        64 * 2 ** (self.s_val_base + i),
                        i,
                    )
                    z_vals, sdf = self.cat_z_vals(
                        rays_o,
                        rays_d,
                        z_vals,
                        new_z_vals,
                        sdf,
                        last=(i + 1 == self.up_sample_steps),
                    )
            n_samples = self.n_samples + self.n_importance

        if self.save_step_sample:
            self.save_step_itr += 1
            # save up to 400 iterations
            if self.save_step_itr > 400:
                exit(0)

        if self.fine_octree_data is not None and self.boundary_samples > 0:
            # to ensure boundary doesn't have noisy surface, LOL
            bound_near_num = self.boundary_samples // 2
            bound_far_num = self.boundary_samples - bound_near_num

            bound_near_z = torch.linspace(0.0, 1.0, bound_near_num + 1, device=device)[
                :-1
            ]
            bound_near_z = near + (z_vals[:, 0][:, None] - near) * bound_near_z[None, :]

            bound_far_z = torch.linspace(0.0, 1.0, bound_far_num + 1, device=device)[1:]
            bound_far_z = (
                z_vals[:, -1][:, None]
                + (far - z_vals[:, -1][:, None]) * bound_far_z[None, :]
            )

            z_vals = torch.cat([bound_near_z, bound_far_z, z_vals], dim=-1)
            z_vals, index = torch.sort(z_vals, dim=-1)

        return n_samples, z_vals, z_vals_outside, sample_dist

    def render_core(
        self,
        rays_o,
        rays_d,
        z_vals,
        sample_dist,
        a_embedded,
        cos_anneal_ratio=0,
        background_alpha=None,
        background_sampled_color=None,
        background_rgb=None,
    ):
        batch_size, n_samples = z_vals.shape
        device = rays_o.device

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, sample_dist.expand(dists[..., -1:].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = (
            rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]
        )  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        pts_ = pts.reshape(batch_size, n_samples, -1)  # N_rays, N_samples, c
        a_embedded_ = a_embedded.unsqueeze(1).expand(
            -1, n_samples, -1
        )  # N_rays, N_samples, c
        rays_d_ = rays_d.unsqueeze(1).expand(-1, n_samples, -1)  # N_rays, N_samples, c

        # inputs for NeuconW
        inputs = [pts_, rays_d_]
        inputs += [a_embedded_]
        # print([it.size() for it in inputs])

        static_out = self.neuconw(torch.cat(inputs, -1))
        rgb, inv_s, sdf, gradients = static_out

        true_cos = (dirs * gradients.reshape(-1, 3)).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        iter_cos = -(
            F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-true_cos) * cos_anneal_ratio
        )  # always non-positive

        # print("prev_cdf shape: ", sdf.size(), dists.reshape(-1, 1).size())
        # Estimate signed distances at section points
        estimated_next_sdf = sdf.reshape(-1, 1) + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf.reshape(-1, 1) - iter_cos * dists.reshape(-1, 1) * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(
            batch_size, n_samples
        )
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # depth
        depth = self.render_depth(alpha, mid_z_vals)
        # print("depth", torch.min(depth), torch.max(depth))
        # print("mid_z_vals", torch.min(mid_z_vals), torch.max(mid_z_vals))

        alpha = alpha * inside_sphere
        rgb = rgb * inside_sphere[:, :, None]
        alpha_in_sphere = alpha
        sphere_rgb = rgb

        # Render with background
        if background_alpha is not None:
            if self.save_sample:
                self.itr += 1
                # save sampled points as point cloud
                filter = (pts_norm < 1.0).view(-1, 1)
                filter_all = filter.repeat(1, 3)
                # print(f'filter_all size {filter_all.size()}')
                inside_pts = (pts[filter_all] * self.radius).view(-1, 3) + self.origin
                # print(f'filter_all val {filter_all}')
                outside_pts = (pts[~filter_all] * self.radius).view(-1, 3) + self.origin
                # print(f'total num{pts.size()}, insides num {inside_pts.size()}, outside num {outside_pts.size()}')
                # print(torch.max(pts_norm), torch.min(pts_norm), torch.sum(filter), torch.sum(1 - filter.long()))
                if self.insiders is None:
                    self.insiders = inside_pts
                    self.outsiders = outside_pts
                else:
                    self.insiders = torch.cat([self.insiders, inside_pts], dim=0)
                    self.outsiders = torch.cat([self.outsiders, outside_pts], dim=0)

                if self.itr % 200 == 0:
                    print("Saving samples...")
                    inside_pcd = o3d.geometry.PointCloud()
                    inside_pcd.points = o3d.utility.Vector3dVector(
                        self.insiders.detach().cpu().numpy()
                    )
                    o3d.io.write_point_cloud(
                        f"samples/inside_pts_sphere_{self.itr}.ply", inside_pcd
                    )

                    outside_pcd = o3d.geometry.PointCloud()
                    outside_pcd.points = o3d.utility.Vector3dVector(
                        self.outsiders.detach().cpu().numpy()
                    )
                    o3d.io.write_point_cloud(
                        f"samples/outside_sphere.ply_{self.itr}.ply", outside_pcd
                    )

                    # clear cache
                    self.insiders = None
                    self.outsiders = None

            # # print("background processed")
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (
                1.0 - inside_sphere
            )
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            rgb = (
                rgb * inside_sphere[:, :, None]
                + background_sampled_color[:, :n_samples]
                * (1.0 - inside_sphere)[:, :, None]
            )
            rgb = torch.cat([rgb, background_sampled_color[:, n_samples:]], dim=1)

            # render background
            if self.trim_sphere:
                background_alpha[:, :n_samples] = torch.zeros_like(
                    alpha[:, :n_samples]
                ) + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            transmittance_bg = torch.cumprod(
                torch.cat(
                    [
                        torch.ones([batch_size, 1], device=device),
                        1.0 - background_alpha + 1e-7,
                    ],
                    -1,
                ),
                -1,
            )[:, :-1]
            weights_bg = background_alpha * transmittance_bg
            color_bg = (background_sampled_color * weights_bg[:, :, None]).sum(dim=1)
        else:
            color_bg = None

        transmittance = torch.cumprod(
            torch.cat(
                [torch.ones([batch_size, 1], device=device), 1.0 - alpha + 1e-7], -1
            ),
            -1,
        )[:, :-1]
        weights = alpha * transmittance
        # weights_sum_s = weights_s.sum(dim=-1, keepdim=True)
        # only calculate weights inside sphere
        weights_sum = (weights[:, :n_samples] * inside_sphere).sum(dim=-1, keepdim=True)

        sphere_transmittance = torch.cumprod(
            torch.cat(
                [
                    torch.ones([batch_size, 1], device=device),
                    1.0 - alpha_in_sphere + 1e-7,
                ],
                -1,
            ),
            -1,
        )[:, :-1]
        weights_sphere = alpha_in_sphere * sphere_transmittance
        color_sphere = (sphere_rgb * weights_sphere[:, :, None]).sum(dim=1)

        # rendered normal
        normals = (gradients * weights[:, :n_samples, None]).sum(dim=1)

        color = (rgb * weights[:, :, None]).sum(dim=1)

        if background_rgb is not None:  # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        gradient_error = (
            torch.linalg.norm(
                gradients.reshape(batch_size, n_samples, 3), ord=2, dim=-1
            )
            - 1.0
        ) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (
            relax_inside_sphere.sum() + 1e-5
        )

        return {
            "color": color,
            "color_sphere": color_sphere,
            "color_bg": color_bg if color_bg is not None else torch.zeros_like(color),
            "sdf": sdf,
            "dists": dists,
            "s_val": 1.0 / inv_s,
            "mid_z_vals": mid_z_vals,
            "weights": weights,
            "weights_sum": weights_sum,
            "cdf": c.reshape(batch_size, n_samples),
            "inside_sphere": inside_sphere,
            "depth": depth,
            "gradient_error": gradient_error,
            "gradients": gradients,
            "normals": normals,
        }

    def render(
        self,
        rays,
        ts,
        label,
        perturb_overwrite=-1,
        background_rgb=None,
        cos_anneal_ratio=0.0,
    ):
        device = rays.device

        # only exceute once
        if not self.origin.is_cuda:
            self.origin = self.origin.to(device).float()
            self.sfm_to_gt = self.sfm_to_gt.to(device).float()

        # Decompose the inputs
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
        near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
        # depth only present in training
        if rays.size()[1] >= 10:
            depth_gt, depth_weight = rays[:, 8], rays[:, 9]
        else:
            depth_gt = depth_weight = torch.zeros_like(near).squeeze()

        # coordinates normalization
        # adjust ray origin to normalized origin
        rays_o = (rays_o - self.origin).float()
        # adjust to unit sphere
        near = (near / self.radius).float()
        far = (far / self.radius).float()
        rays_o = (rays_o / self.radius).float()
        depth_gt = (depth_gt / self.radius).float()

        a_embedded = self.embeddings["a"](ts)

        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        with torch.no_grad():
            n_samples, z_vals, z_vals_outside, sample_dist = self.sparse_sampler(
                rays_o, rays_d, near, far, perturb
            )

        background_alpha = None
        background_sampled_color = None

        # Background model
        if self.render_bg and self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(
                rays_o,
                rays_d,
                z_vals_feed,
                sample_dist,
                self.nerf,
                a_embedded=a_embedded,
            )

            background_sampled_color = ret_outside["sampled_color"]
            background_alpha = ret_outside["alpha"]

        # Render core
        ret_fine = self.render_core(
            rays_o,
            rays_d,
            z_vals,
            sample_dist,
            a_embedded,
            background_rgb=background_rgb,
            background_alpha=background_alpha,
            background_sampled_color=background_sampled_color,
            cos_anneal_ratio=cos_anneal_ratio,
        )

        color = ret_fine["color"]
        weights = ret_fine["weights"]
        gradients = ret_fine["gradients"]
        s_val = ret_fine["s_val"]
        gradient_error = ret_fine["gradient_error"]
        weights_sum = ret_fine["weights_sum"]

        if self.mesh_mask_list is not None:
            mask = torch.ones_like(near)
            for label_name in self.mesh_mask_list:
                mask[get_label_id_mapping()[label_name] == label] = 0
            mask_error = F.binary_cross_entropy(
                weights_sum.clip(1e-3, 1.0 - 1e-3), mask, reduction="none"
            )
        else:
            mask_error = torch.zeros_like(weights_sum)

        rendered_depth = ret_fine["depth"]
        normals = ret_fine["normals"]  # n_rays, 3
        if self.floor_normal:
            floor_normal_error, floor_y_error = self.floor_loss(
                label, normals, rays_o, rays_d, rendered_depth
            )
        else:

            floor_normal_error, floor_y_error = torch.zeros_like(
                normals
            ), torch.zeros_like(normals)

        # depth error
        if self.depth_loss and torch.sum(depth_weight > 0) > 0:
            sfm_depth_loss = (((rendered_depth - depth_gt) ** 2) * depth_weight)[
                depth_weight > 0
            ]
        else:
            sfm_depth_loss = torch.zeros_like(rendered_depth)

        return {
            "color": color,
            "color_sphere": ret_fine["color_sphere"],
            "color_bg": ret_fine["color_bg"],
            "s_val": s_val,
            "cdf_fine": ret_fine["cdf"],
            "gradients": gradients,
            "mask_error": mask_error,
            "weights": weights,
            "weights_sum": ret_fine["weights_sum"],
            "weights_max": torch.max(weights, dim=-1, keepdim=True)[0],
            "gradient_error": torch.ones(1, device=device) * gradient_error,
            "inside_sphere": ret_fine["inside_sphere"],
            "depth": ret_fine["depth"],
            "floor_normal_error": floor_normal_error,
            "floor_y_error": floor_y_error,
            "sfm_depth_loss": sfm_depth_loss,
        }

    def floor_loss(self, label, normals, rays_o, rays_d, rendered_depth):
        device = rays_o.device
        batch_size = label.size()[0]
        floor_mask = torch.zeros([batch_size], dtype=torch.bool)
        for f_label in self.floor_labels:
            floor_mask[get_label_id_mapping()[f_label] == label] = 1

        if torch.sum(floor_mask) == 0:
            return torch.zeros_like(normals), torch.zeros_like(normals)

        floor_normal = normals[floor_mask]

        # in gt coordinate system, z is perpendicular to road
        floor_normal_gt = torch.zeros_like(floor_normal)
        floor_normal_gt[:, 2] = 1
        floor_normal_gt = self.sfm_to_gt[:3, :3].T @ floor_normal_gt.permute(1, 0)
        floor_normal_gt = floor_normal_gt.permute(1, 0).contiguous()
        floor_normal_gt = floor_normal_gt / torch.linalg.norm(
            floor_normal_gt, dim=-1
        ).reshape(-1, 1)

        floor_error = F.l1_loss(floor_normal, floor_normal_gt, reduction="none")

        xyzs = rays_o + rays_d * rendered_depth.reshape(-1, 1)
        road_y = xyzs[floor_mask]
        y_error = torch.var(road_y)

        return floor_error, torch.ones_like(floor_error) * y_error

    def sdf(self, pts):
        sdf = self.neuconw.sdf(pts)
        return sdf

    def rgb(self, pts, rays_d, a_embedded):
        num_points, _, _ = pts.size()

        # inputs for NeuconW
        inputs = [pts, rays_d]
        inputs += [a_embedded]
        # print([it.size() for it in inputs])

        static_out = self.neuconw(torch.cat(inputs, -1))
        rgb, inv_s, sdf, gradients = static_out
        return rgb.reshape(num_points, 3)
