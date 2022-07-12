#!/bin/bash
set -e
now=$(date +"%Y%m%d_%H%M%S")
jobname="eval-$now"

#################################################
# change these according to your machine status
N_GPUS=4
N_CPUS=4
export CUDA_VISIBLE_DEVICES=0,2,5,6
##################################################

scene_name=$1
pred_dir=$2

GT_DATA_PATH='data/heritage-recon'
pred_path=${pred_dir}/mesh
pred_mesh_name="reprojected.ply"
eval_target=${GT_DATA_PATH}/${scene_name}/${scene_name}.ply
sfm_path=${GT_DATA_PATH}/${scene_name}/neuralsfm

if [ ${scene_name} == 'brandenburg_gate' ]; then
    scene_name=brandenburg_gate
    scene_abv=bg
    thresholds="0.01,1,0.01"
    track_lenth=14
    reproj_error=2
    voxel_size=2
elif [ ${scene_name} == 'lincoln_memorial' ]; then
    scene_name=lincoln_memorial
    scene_abv=lm
    thresholds="0.005,0.3,0.005"
    track_lenth=12
    reproj_error=1.6
    voxel_size=0.04
elif [ ${scene_name} == 'palacio_de_bellas_artes' ]; then
    scene_name=palacio_de_bellas_artes
    scene_abv=pba
    thresholds="0.01,1,0.01"
    track_lenth=12
    reproj_error=1.5
    voxel_size=2
elif [ ${scene_name} == 'pantheon_exterior' ]; then
    scene_name=pantheon_exterior
    scene_abv=pe
    thresholds="0.01,1,0.01"
    track_lenth=12
    reproj_error=1.4
    voxel_size=0.1
else
    echo Not suppotted scene: ${scene_name}
    return
fi


echo Evaulating ${pred_dir} ...

# filtering pred
python utils/reproj_filter.py \
  --src_file ${pred_path}/extracted_mesh_level_10_colored.ply \
  --target_file ${pred_path}/extracted_mesh_level_10_colored.ply \
  --data_path ${GT_DATA_PATH}/${scene_name} \
  --output_path ${pred_path} \
  --n_cpus ${N_CPUS} \
  --n_gpus ${N_GPUS} \
2>&1|tee log/${jobname}_filtering_pred.log


# Evaulation on pred result
python utils/eval_mesh.py \
--file_pred ${pred_path}/${pred_mesh_name} \
--file_trgt ${eval_target} \
--scene_config_path ${GT_DATA_PATH}/${scene_name}/config.yaml \
--threshold ${thresholds} \
--save_name $1_${pred_mesh_name} \
--sfm_path ${sfm_path} \
--track_lenth ${track_lenth} --reproj_error ${reproj_error} --voxel_size ${voxel_size} \
--bbx_name eval_bbx \
2>&1|tee log/${jobname}_pred.log
