import torch
from collections import defaultdict
from pytorch_lightning import LightningModule
import numpy as np

# models
from models.neuconw import NeuconW
from models.nerf import NeRF, PosEmbedding
from rendering.renderer import NeuconWRenderer
from shutil import copyfile

# losses
from losses import loss_dict

# optimizer, scheduler, visualization
from utils import *

# metrics
from metrics import *
import os
from datasets.mask_utils import get_label_id_mapping
from utils.visualization import extract_mesh

import matplotlib.pyplot as plt
import yaml
from utils.eval_mesh import eval_mesh
from tools.prepare_data.generate_voxel import (
    convert_to_dense,
    gen_octree,
    octree_to_spc,
)
from utils.comm import get_world_size, get_rank
import torch.distributed as dist

# Get the color map by name:
cm = plt.get_cmap("gist_rainbow")


def get_local_split(data, world_size, local_rank):
    if data.shape[0] % world_size != 0:
        xyz_padded = torch.cat(
            (
                data,
                torch.zeros(
                    (data.shape[0] // world_size + 1) * world_size - data.shape[0],
                    data.shape[1],
                ).to(data.device),
            ),
            0,
        )
    else:
        xyz_padded = data
    local_split_length = xyz_padded.shape[0] // world_size
    local_data = xyz_padded[
        local_rank * local_split_length : (local_rank + 1) * local_split_length
    ]
    return local_data


class NeuconWSystem(LightningModule):
    def __init__(self, hparams, config, caches):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.config = config
        scene_config_path = os.path.join(self.config.DATASET.ROOT_DIR, "config.yaml")
        with open(scene_config_path, "r") as yamlfile:
            self.scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

        self.loss = loss_dict["neuconw"](**self.config.NEUCONW.LOSS, config=config)
        self.models_to_train = []
        self.embeddings = {}

        # todo: remove appearance and transient Embedding if not using
        self.embedding_a = torch.nn.Embedding(
            self.config.NEUCONW.N_VOCAB, self.config.NEUCONW.N_A
        )
        self.embeddings["a"] = self.embedding_a
        self.models_to_train += [self.embedding_a]

        # define NEUCONW model
        self.neuconw = NeuconW(
            sdfNet_config=self.config.NEUCONW.SDF_CONFIG,
            colorNet_config=self.config.NEUCONW.COLOR_CONFIG,
            SNet_config=self.config.NEUCONW.S_CONFIG,
            in_channels_a=self.config.NEUCONW.N_A,
            encode_a=self.config.NEUCONW.ENCODE_A,
        )

        # for background
        self.nerf = NeRF(
            D=8,
            d_in=4,
            d_in_view=3,
            W=256,
            multires=10,
            multires_view=4,
            output_ch=4,
            skips=[4],
            encode_appearance=self.config.NEUCONW.ENCODE_A_BG,
            in_channels_a=self.config.NEUCONW.N_A,
            in_channels_dir=6 * self.config.NEUCONW.COLOR_CONFIG.multires_view + 3,
            use_viewdirs=True,
        )

        self.anneal_end = self.config.NEUCONW.ANNEAL_END

        spc_options = {
            "voxel_size": self.scene_config["voxel_size"],
            "recontruct_path": self.config.DATASET.ROOT_DIR,
            "min_track_length": self.scene_config["min_track_length"],
        }
        self.renderer = NeuconWRenderer(
            nerf=self.nerf,
            neuconw=self.neuconw,
            embeddings=self.embeddings,
            n_samples=self.config.NEUCONW.N_SAMPLES,
            s_val_base=self.config.NEUCONW.S_VAL_BASE,
            n_importance=self.config.NEUCONW.N_IMPORTANCE,
            n_outside=self.config.NEUCONW.N_OUTSIDE,
            up_sample_steps=self.config.NEUCONW.UP_SAMPLE_STEP,
            perturb=1.0,
            origin=self.scene_config["origin"],
            radius=self.scene_config["radius"],
            render_bg=self.config.NEUCONW.RENDER_BG,
            mesh_mask_list=self.config.NEUCONW.MESH_MASK_LIST,
            floor_normal=self.config.NEUCONW.FLOOR_NORMAL,
            floor_labels=self.config.NEUCONW.FLOOR_LABELS,
            depth_loss=self.config.NEUCONW.DEPTH_LOSS,
            spc_options=spc_options,
            sample_range=self.config.NEUCONW.SAMPLE_RANGE,
            boundary_samples=self.config.NEUCONW.BOUNDARY_SAMPLES,
            nerf_far_override=self.config.NEUCONW.NEAR_FAR_OVERRIDE,
        )

        self.models = {"neuconw": self.neuconw, "nerf": self.nerf}
        self.models_to_train += [self.models]
        self.caches = caches

        self.ray_mask_list = self.config.NEUCONW.RAY_MASK_LIST
        self.sdf_threshold = self.config.NEUCONW.SDF_THRESHOLD
        self.update_freq = self.config.NEUCONW.UPDATE_FREQ

        if self.update_freq > 0:
            self.train_level = self.surface_level(
                self.config.NEUCONW.TRAIN_VOXEL_SIZE, self.scene_config["eval_bbx"]
            )

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0
        else:
            return np.min([1.0, self.global_step / self.anneal_end])

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, label):
        """Do batched inference on rays"""
        results = defaultdict(list)
        rendered_ray_chunks = self.renderer.render(
            rays,
            ts,
            label,
            background_rgb=torch.zeros([1, 3], device=rays.device),
            cos_anneal_ratio=self.get_cos_anneal_ratio(),
        )

        for k, v in rendered_ray_chunks.items():
            results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, dim=0)

        return results

    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.config.TRAINER, self.models_to_train)
        scheduler = get_scheduler(self.hparams, self.config.TRAINER, self.optimizer)
        if scheduler is None:
            return [self.optimizer]
        else:
            return [self.optimizer], [scheduler]

    def surface_selection(self, train_level, threshold, device=0, chunk=65536):
        scene_origin_sfm = (self.renderer.origin).float().cpu()
        scene_radius_sfm = self.renderer.radius

        # get sparse voxel coordinates
        if self.renderer.octree_data is None:
            self.renderer.octree_data = self.renderer.get_octree(device)
        octree_data = self.renderer.octree_data

        # unpack octree
        octree = octree_data["octree"]
        octree_origin = octree_data["scene_origin"].float().cpu()
        octree_scale = octree_data["scale"]
        octree_level = octree_data["level"]

        # upsample octree
        dense = convert_to_dense(octree, octree_level).cpu()
        low_dim = dense.size()[0]

        # dense to sparse
        sparse_ind = torch.nonzero(dense > 0)  # n, 3
        sparse_num = sparse_ind.size()[0]

        # upsample
        up_level = train_level - octree_level
        up_times = 2**up_level

        train_dim = int(low_dim * (2**up_level))
        print(
            f"train dim: {train_dim}, upsampled {up_times} times, original dim {low_dim}, sparse num {sparse_num}"
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
        train_voxel_size = 2 / (2**train_level) * octree_scale
        vol_origin = octree_origin - octree_scale

        xyz_sfm = sparse_ind_up * train_voxel_size + vol_origin

        # feed to sdf net to get sdf
        xyz_training = (xyz_sfm - scene_origin_sfm) / scene_radius_sfm

        # multi-gpu
        world_size = get_world_size()
        local_rank = get_rank()
        local_xyz_ = get_local_split(xyz_training, world_size, local_rank)

        B = local_xyz_.shape[0]
        out_chunks = []
        for i in tqdm(range(0, B, chunk), disable=local_rank != 0):
            new_sdf = self.renderer.sdf(
                local_xyz_[i : i + chunk].reshape(-1, 1, 3).cuda()
            )
            out_chunks += [new_sdf.detach().cpu()]
        sdf = torch.cat(out_chunks, 0).reshape(-1)

        # if multi gpu
        if self.hparams.num_gpus > 1:
            sdf_gathered = [
                torch.zeros(B, dtype=torch.float, device=device) for _ in range(world_size)
            ]
            dist.all_gather(sdf_gathered, sdf.cuda())
            sdf = torch.cat(sdf_gathered, 0).reshape(-1)[: xyz_training.size()[0]]

        # filter with threshold
        sparse_pc_sfm = xyz_sfm[sdf <= threshold].cpu().numpy()
        print(
            f"sdf filtered points {sparse_pc_sfm.shape[0]}, max sdf: {torch.min(sdf)}, min sdf: {torch.max(sdf)}"
        )

        return sparse_pc_sfm, train_voxel_size

    def octree_update(
        self, train_level, threshold, device=0, chunk=65536, visualize=False
    ):
        print("Updating sdf to octree....")

        del self.renderer.fine_octree_data

        # get suface points
        sparse_pc_sfm, train_voxel_size = self.surface_selection(
            train_level, threshold, device, chunk
        )

        # use remaining points to generate new octree
        octree_new, scene_origin, scale, level = gen_octree(
            self.renderer.recontruct_path,
            sparse_pc_sfm,
            train_voxel_size,
            device=device,
            visualize=visualize,
            expand=False,
        )

        del sparse_pc_sfm
        # update to renderer
        octree_data = {}
        octree_data["octree"] = octree_new
        octree_data["scene_origin"] = torch.from_numpy(scene_origin).to(device)
        octree_data["scale"] = scale
        octree_data["level"] = level
        octree_data["voxel_size"] = train_voxel_size

        spc_data = {}
        points, pyramid, prefix = octree_to_spc(octree_new)
        spc_data["points"] = points
        spc_data["pyramid"] = pyramid
        spc_data["prefix"] = prefix

        octree_data["spc_data"] = spc_data

        self.renderer.fine_octree_data = octree_data

        # an attempt to speed up training
        torch.cuda.empty_cache()

        print("Update successful!!")

    def surface_level(self, voxel_size, bbx):
        """calculate octree level based on voxel size in world coordinates

        Args:
            voxel_size (float): voxel size in world coordinates
            bbx (floatTensor): bounding box of scene
        """
        bbx_min = np.array(bbx[0])
        bbx_max = np.array(bbx[1])

        # dimensions
        dim = np.max(bbx_max - bbx_min)

        scene_origin = bbx_min + (bbx_max - bbx_min) / 2
        scale = dim / 2

        level = int(np.ceil(np.log2(2 * scale / voxel_size)))
        print(
            f"training octree will be in level: {level}, with respect to voxel size {2 / (2 ** level) * scale}"
        )

        return level

    def training_step(self, batch, batch_nb):
        rays, rgbs, ts, label = (
            batch["rays"],
            batch["rgbs"],
            batch["ts"],
            batch["semantics"],
        )
        # training will always read near far from cache
        self.renderer.nerf_far_override = False
        # filter rays
        # mask rays as black list
        ray_mask = torch.ones_like(ts, dtype=torch.bool)
        if self.ray_mask_list is not None:
            for label_name in self.ray_mask_list:
                ray_mask[get_label_id_mapping()[label_name] == label] = False
        rays = rays[ray_mask, :]
        ts = ts[ray_mask]
        rgbs = rgbs[ray_mask]
        label = label[ray_mask]

        results = self(rays, ts, label)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            psnr_ = psnr(results[f"color"], rgbs)

        self.log("lr", get_learning_rate(self.optimizer))
        self.log("train/loss", loss)
        for k, v in loss_d.items():
            self.log(f"train/{k}", v, prog_bar=True)
        self.log("train/psnr", psnr_, prog_bar=True)
        self.log("train/s_val", results["s_val"].mean())
        with torch.no_grad():
            if self.update_freq > 0 and (self.global_step + 1) % self.update_freq == 0:
                self.octree_update(
                    self.train_level, self.sdf_threshold, device=rays.device
                )

        if (
            self.global_step % self.config.TRAINER.SAVE_FREQ == 0
            and self.trainer.global_rank == 0
        ):
            os.makedirs(
                f"{self.config.TRAINER.SAVE_DIR}/{self.hparams.exp_name}", exist_ok=True
            )
            self.trainer.save_checkpoint(
                f"{self.config.TRAINER.SAVE_DIR}/{self.hparams.exp_name}/iter_{self.global_step}.ckpt"
            )

            if self.global_step == 0:
                # save config
                config_save_path = (
                    f"{self.config.TRAINER.SAVE_DIR}/{self.hparams.exp_name}/config"
                )
                os.makedirs(config_save_path, exist_ok=True)
                config_dir = "./config"
                config_files = os.listdir(config_dir)
                for config_file in config_files:
                    if os.path.isfile(os.path.join(config_dir, config_file)):
                        copyfile(
                            os.path.join(config_dir, config_file),
                            os.path.join(config_save_path, config_file),
                        )

        return loss

    def validation_step(self, batch, batch_nb):
        torch.set_grad_enabled(True)
        # if near far from cache is generated by intersection sfm voxel, turn this flag on in config
        self.renderer.nerf_far_override = self.config.NEUCONW.NEAR_FAR_OVERRIDE
        rays, rgbs, ts, labels = (
            batch["rays"],
            batch["rgbs"],
            batch["ts"],
            batch["semantics"],
        )
        rays = rays.squeeze()  # (H*W, 3)
        rgbs = rgbs.squeeze()  # (H*W, 3)
        labels = labels.squeeze()  # (H*W)
        ts = ts.squeeze()  # (H*W)
        B = rays.shape[0]

        results = defaultdict(list)
        for i in range(0, B, self.hparams.test_batch_size):
            rendered_ray_chunks = self(
                rays[i : i + self.hparams.test_batch_size],
                ts[i : i + self.hparams.test_batch_size],
                labels[i : i + self.hparams.test_batch_size],
            )
            for k, v in rendered_ray_chunks.items():
                results[k] += [v.detach()]
        for k, v in results.items():
            results[k] = torch.cat(v, dim=0)
        loss_d = self.loss(results, rgbs)
        loss = sum(l for l in loss_d.values())
        log = {"val_loss": loss}

        if batch_nb == 0 and self.global_step > 0:
            if self.trainer.global_rank == 0:
                WH = batch["img_wh"]
                W, H = WH[0, 0].item(), WH[0, 1].item()

                img = (
                    results[f"color"].view(H, W, 3).permute(2, 0, 1).cpu()
                )  # (3, H, W)
                img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()  # (3, H, W)
                depth = visualize_depth(results[f"depth"].view(H, W))  # (3, H, W)
                normal = (
                    (
                        torch.sum(
                            results["gradients"][:, :, :]
                            * results["weights"][
                                :, : results["gradients"].size()[1], None
                            ],
                            dim=1,
                        )
                    )
                    .view(H, W, 3)
                    .detach()
                    .cpu()
                )  # (H, W, 3)
                normal = normal / torch.linalg.norm(normal, axis=-1)[:, :, None]
                normal_mapped = (normal / 2 + 0.5).permute(2, 0, 1)  # (3, H, W)
                stack = torch.stack([img_gt, img, depth, normal_mapped])  # (3, 3, H, W)
                self.logger.experiment.add_images(
                    "val/GT_pred_depth_normal", stack, self.global_step
                )

            # save mesh
            mesh_dir = os.path.join(self.logger.save_dir, self.logger.name, "meshes")
            mesh = extract_mesh(
                dim=128,
                chunk=16384,
                scene_radius=self.scene_config["radius"],
                scene_origin=self.scene_config["origin"],
                with_color=False,
                renderer=self.renderer,
            )
            os.makedirs(mesh_dir, exist_ok=True)
            if self.trainer.global_rank == 0:
                mesh.export(
                    os.path.join(mesh_dir, "{:0>8d}.ply".format(self.global_step))
                )

            sfm_to_gt = np.array(self.scene_config["sfm2gt"])
            gt_to_sfm = np.linalg.inv(sfm_to_gt)
            sfm_vert1 = (
                gt_to_sfm[:3, :3] @ np.array(self.scene_config["eval_bbx_detail"][0])
                + gt_to_sfm[:3, 3]
            )
            sfm_vert2 = (
                gt_to_sfm[:3, :3] @ np.array(self.scene_config["eval_bbx_detail"][1])
                + gt_to_sfm[:3, 3]
            )
            eval_bbx_detail_min = np.minimum(sfm_vert1, sfm_vert2)
            eval_bbx_detail_max = np.maximum(sfm_vert1, sfm_vert2)
            eval_bbx_detail_center = (eval_bbx_detail_max + eval_bbx_detail_min) / 2
            dim_eval_bbx = np.max(eval_bbx_detail_max - eval_bbx_detail_min) / 2

            mesh_detail = extract_mesh(
                dim=256,
                chunk=16384,
                scene_radius=self.scene_config["radius"],
                scene_origin=self.scene_config["origin"],
                origin=(eval_bbx_detail_center - self.scene_config["origin"])
                / self.scene_config["radius"],
                radius=dim_eval_bbx / self.scene_config["radius"],
                with_color=False,
                renderer=self.renderer,
            )
            if self.trainer.global_rank == 0:
                mesh_detail.export(
                    os.path.join(
                        mesh_dir, "{:0>8d}_detail.ply".format(self.global_step)
                    )
                )

            gt_path = os.path.join(self.config.DATASET.ROOT_DIR, "gt.ply")
            if os.path.exists(gt_path) and self.trainer.global_rank == 0:
                eval_metrics = eval_mesh(
                    file_pred=os.path.join(
                        mesh_dir, "{:0>8d}_detail.ply".format(self.global_step)
                    ),
                    file_trgt=gt_path,
                    scene_config=self.scene_config,
                    is_mesh=False,
                    threshold=0.1,
                    bbx_name="eval_bbx_detail",
                    use_o3d=False,
                )
                print("eval_metric: ", eval_metrics)
                self.log("val/prec", eval_metrics["prec"], rank_zero_only=True)
                self.log("val/recal", eval_metrics["recal"], rank_zero_only=True)
                self.log("val/fscore", eval_metrics["fscore"], rank_zero_only=True)

        psnr_ = psnr(results[f"color"], rgbs)
        log["val_psnr"] = psnr_

        # an attempt to speed up training
        torch.cuda.empty_cache()

        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        mean_psnr = torch.stack([x["val_psnr"] for x in outputs]).mean()

        self.log("val/loss", mean_loss)
        self.log("val/psnr", mean_psnr, prog_bar=True)
