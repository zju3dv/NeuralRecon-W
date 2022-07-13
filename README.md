# Neural 3D Reconstruction in the Wild

### [Project Page](https://zju3dv.github.io/neuralrecon-w) | [Paper](https://arxiv.org/pdf/2205.12955)

<br/>

> Neural 3D Reconstruction in the Wild  
> [Jiaming Sun](https://jiamingsun.ml), [Xi Chen](https://github.com/Burningdust21), [Qianqian Wang](https://www.cs.cornell.edu/~qqw/), [Zhengqi Li](https://zhengqili.github.io/), [Hadar Averbuch-Elor](https://www.cs.cornell.edu/~hadarelor/), [Xiaowei Zhou](https://xzhou.me), [Noah Snavely](https://www.cs.cornell.edu/~snavely/)  
> SIGGRAPH 2022 (Conference Proceedings)

![demo_vid](assets/neuconw-github-teaser.gif)

## TODO List

- [x] Training (i.e., reconstruction) code.
- [x] Toolkit and pipeline to reproduce the evaluation results on the proposed Heritage-Recon dataset.
- [x] Config for reconstructing generic outdoor/indoor scenes.

## Installation

```shell
conda env create -f environment.yaml
conda activate neuconw
scripts/download_sem_model.sh
```

## Reproduce reconstruction results on Heritage-Recon

### Dataset setup

Download the [Heritage-Recon](https://drive.google.com/drive/folders/1eZvmk4GQkrRKUNZpagZEIY_z8Lsdw94v?usp=sharing) dataset and put it under `data`. You can also use [gdown](https://github.com/wkentaro/gdown) to download it in command line:

```bash
mkdir data && cd data
gdown --id 1eZvmk4GQkrRKUNZpagZEIY_z8Lsdw94v
```

Generate ray cache for all four scenes:

```bash
for SCENE_NAME in brandenburg_gate lincoln_memorial palacio_de_bellas_artes pantheon_exterior; do
  scripts/data_generation.sh data/heritage-recon/${SCENE_NAME}
done
```

### Training

To train scenes in our Heritage-Recon dataset:

```bash
# Subsutitude `SCENE_NAME` with the scene you want to reconstruct.
scripts/train.sh $EXP_NAME config/train_${SCENE_NAME}.yaml $NUM_GPU $NUM_NODE
```


### Evaluating

First, extracting mesh from a checkpoint you want to evaluate:

```bash
scripts/sdf_extract.sh $EXP_NAME config/train_${SCENE_NAME}.yaml $CKPT_PATH 10
```

The reconstructed meshes will be saved to `PROJECT_PATH/results`.

Then run the evaluation pipeline:

```bash
scripts/eval_pipeline.sh $SCENE_NAME $MESH_PATH
```

Evaluation results will be saved in the same folder with the evaluated mesh.

## Reconstructing custom data

### Data preparation

#### Automatic generation

The code takes a standard COLMAP workspace format as input, a script is provided for automatically convert a colmap workspace into our data format:

```bash
scripts/preprocess_data.sh
```

More instructions can be found in `scripts/preprocess_data.sh`

#### Manual selection

However, if you wish to select a better bounding box (i.e., reconstruction region) manually, do the following steps.

#### 1. Generate semantic maps

Generate semantic maps:

```bash
python tools/prepare_data/prepare_semantic_maps.py --root_dir $WORKSPACE_PATH --gpu 0
```

#### 2. Create scene metadata file

Create a file `config.yaml` into workspace to write metadata. The target scene needs to be normalized into a unit sphere, which require manual selection. One simple way is to use SFM key-points points from COLMAP to determine the origin and radius. Also, a bounding box is required, which can be set to `[origin-raidus, origin+radius]`, or only the region you're interested in.

```yaml
{
    name: brandenburg_gate, # scene name
    origin: [ 0.568699, -0.0935532, 6.28958 ], 
    radius: 4.6,
    eval_bbx: [[-14.95992661, -1.97035599, -16.59869957],[48.60944366, 30.66258621, 12.81980324]],
    voxel_size: 0.25,
    min_track_length: 10,
    # The following configuration is only used in evaluation, can be ignored for your own scene
    sfm2gt: [[1, 0, 0, 0],
            [ 0, 1, 0, 0],
            [ 0, 0, 1, 0],
            [ 0, 0, 0, 1]],
}
```

#### 3. Generate cache

Run the following command with a `WORKSPACE_PATH` specified:

```bash
scripts/data_generation.sh $WORKSPACE_PATH
```

After completing above steps, whether automatically or manually, the COLMAP workspace should be looking like this;

```bash
└── brandenburg_gate
  └── brandenburg_gate.tsv
  ├── cache_sgs
    └── splits
        ├── rays1_meta_info.json
        ├── rgbs1_meta_info.json
        ├── split_0
            ├── rays1.h5
            └── rgbs1.h5
        ├── split_1
        ├──.....
  ├── config.yaml
  ├── dense
    └── sparse
        ├── cameras.bin
        ├── images.bin
        ├── points3D.bin
  └── semantic_maps
      ├── 99119670_397881696.jpg
      ├── 99128562_6434086647.jpg
      ├── 99250931_9123849334.jpg
      ├── 99388860_2887395078.jpg
      ├──.....
```

### Training

Change `DATASET.ROOT_DIR` to COLMAP workspace path in `config/train.yaml`, and run:

```bash
scripts/train.sh $EXP_NAME config/train.yaml $NUM_GPU $NUM_NODE
```

Additionally, `NEUCONW.SDF_CONFIG.inside_outside` should be set to `True` if training an indoor scene (refer to `config/train_indoor.yaml`).

### Extracting mesh

```bash
scripts/sdf_extract.sh $EXP_NAME config/train.yaml $CKPT_PATH $EVAL_LEVEL
```

The reconstructed meshes will be saved to `PROJECT_PATH/results`.

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

```bibtex
@inproceedings{sun2022neuconw,
  title={Neural {3D} Reconstruction in the Wild},
  author={Sun, Jiaming and Chen, Xi and Wang, Qianqian and Li, Zhengqi and Averbuch-Elor, Hadar and Zhou, Xiaowei and Snavely, Noah},
  booktitle={SIGGRAPH Conference Proceedings},
  year={2022}
}
```

## Acknowledgement

Part of our code is derived from [nerf_pl](https://github.com/kwea123/nerf_pl) and [NeuS](https://github.com/Totoro97/NeuS), thanks to their authors for the great works.
