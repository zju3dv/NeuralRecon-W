#!/bin/bash
set -e
now=$(date +"%Y%m%d_%H%M%S")
jobname="eval-$1-$now"

IMC_DATA_PATH='/nas/datasets/IMC'
OH3_DATA_PATH='/nas/datasets/OpenHeritage3D'
WORK_SPACE='/nas/users/chenxi/neusw_files'
N_GPUS=4
N_CPUS=8
export CUDA_VISIBLE_DEVICES=1,0,2,3



# brandenburg_gate lincoln_memorial palacio_de_bellas_artes pantheon_exterior trevi_fountain
scene_name=palacio_de_bellas_artes
scene_abv=pba
thresholds="0.01,1,0.01"


# ${OH3_DATA_PATH}/processed/${scene_name}/${scene_abv}_cropped.ply
# ${WORK_SPACE}/gt/${scene_abv}/reprojected_0.16.ply
eval_target=${OH3_DATA_PATH}/processed/${scene_name}/${scene_abv}_cropped_sampled_0.05.ply

sfm_path=${IMC_DATA_PATH}/phototourism/training_set/${scene_name}/dense/neuralsfm
track_lenth=12
reproj_error=1.5
voxel_size=2
# --track_lenth ${track_lenth} --reproj_error ${reproj_error} --voxel_size ${voxel_size} \

# surf_guided_pba_tl_5_fix_normal_resume-20211221_103636_iter_360000
# surf_guided_pba_depth_40-20220119_204059_iter_*
for ours_dir in results/phototourism/surf_guided_pba_depth_40-20220119_204059_iter_270000; do
    # echo ${ours_dir}
    pred_path=${ours_dir}/mesh
    ours_mesh_name="reprojected.ply"

    colmap_path=${WORK_SPACE}/colmap_results/${scene_abv}
    colmap_mesh_name="reprojected_d11.ply"

    vismvs_path=${WORK_SPACE}/vis_MVS_results/${scene_name}
    vismvs_mesh_name="reprojected_d13.ply "

    nerfw_path=${WORK_SPACE}/nerfw_results
    nerfw_mesh_name="${scene_name}.ply"

    # # filtering gt
    # # ropped_sampled_0.02
    # name=cropped_sampled_0.02
    # # bg_cropped_sampled_0.01 sampled_0.05_cropped
    # src_file=/nas/datasets/OpenHeritage3D/processed/${scene_name}/${scene_abv}_${name}.ply
    # trg_file=/nas/datasets/OpenHeritage3D/processed/${scene_name}/${scene_abv}_${name}.ply
    # out_path=/nas/users/chenxi/neusw_files/gt/${scene_abv}

    # python utils/reproj_filter.py \
    #   --src_file $src_file \
    #   --target_file $src_file \
    #   --data_path /nas/datasets/IMC/phototourism/training_set/${scene_name} \
    #   --output_path $out_path \
    #   --gt \
    #   --voxel_size 0.2\
    #   --visualize \
    #   --n_cpus 1 \
    #   --n_gpus 1 \
    #   2>&1|tee log/${jobname}_filtering_gt.log


    # # filtering ours
    # python utils/reproj_filter.py \
    #   --src_file ${pred_path}/extracted_mesh_level_10_colored.ply \
    #   --target_file ${pred_path}/extracted_mesh_level_10_colored.ply \
    #   --data_path ${IMC_DATA_PATH}/phototourism/training_set/${scene_name} \
    #   --output_path ${pred_path} \
    #   --n_cpus ${N_CPUS} \
    #   --n_gpus ${N_GPUS} \
    # 2>&1|tee log/${jobname}_filtering_ours.log

    # # filtering colmap
    # python utils/reproj_filter.py \
    #   --src_file ${colmap_path}/${colmap_mesh_name} \
    #   --target_file ${colmap_path}/${colmap_mesh_name} \
    #   --data_path ${IMC_DATA_PATH}/phototourism/training_set/${scene_name} \
    #   --output_path ${colmap_path} \
    #   --n_cpus ${N_CPUS} \
    #   --n_gpus ${N_GPUS} \
    # 2>&1|tee log/${jobname}_filtering_colmap.log

    # # filtering vismvs
    # python utils/reproj_filter.py \
    #   --src_file ${vismvs_path}/${vismvs_mesh_name} \
    #   --target_file ${vismvs_path}/${vismvs_mesh_name} \
    #   --data_path ${IMC_DATA_PATH}/phototourism/training_set/${scene_name} \
    #   --output_path ${vismvs_path} \
    #   --n_cpus ${N_CPUS} \
    #   --n_gpus ${N_GPUS} \
    # 2>&1|tee log/${jobname}_filtering_vismvs.log

    # # Evaluation on ours result
    # python utils/eval_mesh.py \
    # --file_pred ${pred_path}/${ours_mesh_name} \
    # --file_trgt ${eval_target} \
    # --scene_config_path ${IMC_DATA_PATH}/phototourism/training_set/${scene_name}/config.yaml \
    # --threshold ${thresholds} \
    # --save_name $1_${ours_mesh_name} \
    # --sfm_path ${sfm_path} \
    # --track_lenth ${track_lenth} --reproj_error ${reproj_error} --voxel_size ${voxel_size} \
    # --bbx_name eval_bbx \
    # 2>&1|tee log/${jobname}_ours.log


    # # Evaluation on colmap result
    # python utils/eval_mesh.py \
    # --file_pred ${colmap_path}/${colmap_mesh_name} \
    # --file_trgt ${eval_target} \
    # --scene_config_path ${IMC_DATA_PATH}/phototourism/training_set/${scene_name}/config.yaml \
    # --threshold ${thresholds} \
    # --save_name $1_${colmap_mesh_name} \
    # --sfm_path ${sfm_path} \
    # --track_lenth ${track_lenth} --reproj_error ${reproj_error} --voxel_size ${voxel_size} \
    # --bbx_name eval_bbx \
    # --mesh \
    # 2>&1|tee log/${jobname}_colmap.log


    # # Evaluation on mvs result
    # python utils/eval_mesh.py \
    # --file_pred ${vismvs_path}/${vismvs_mesh_name} \
    # --file_trgt ${eval_target} \
    # --scene_config_path ${IMC_DATA_PATH}/phototourism/training_set/${scene_name}/config.yaml \
    # --sfm_path ${sfm_path} \
    # --track_lenth ${track_lenth} --reproj_error ${reproj_error} --voxel_size ${voxel_size} \
    # --threshold ${thresholds} \
    # --save_name $1_${vismvs_mesh_name} \
    # --bbx_name eval_bbx


    # # Evaluation on nerfw result
    # python utils/eval_mesh.py \
    # --file_pred ${nerfw_path}/${nerfw_mesh_name} \
    # --file_trgt ${eval_target} \
    # --scene_config_path ${IMC_DATA_PATH}/phototourism/training_set/${scene_name}/config.yaml \
    # --mesh \
    # --sfm_path ${sfm_path} \
    # --track_lenth ${track_lenth} --reproj_error ${reproj_error} --voxel_size ${voxel_size} \
    # --threshold ${thresholds} \
    # --save_name $1_${nerfw_mesh_name} \
    # --bbx_name eval_bbx

    # # visualize colmap and ours
    # python utils/vis_metrics.py \
    # --max_num 80 \
    # --ours_path ${pred_path}/eval_$1_${ours_mesh_name} \
    # --colmap_path ${colmap_path}/eval_$1_${colmap_mesh_name} \
    # --save_name ${scene_abv}_$1_ous_${ours_mesh_name}_colmap_${colmap_mesh_name}


    # # visualize nerfw and ours
    # python utils/vis_metrics.py \
    # --max_num 80 \
    # --ours_path ${nerfw_path}/eval_$1_${nerfw_mesh_name} \
    # --colmap_path ${nerfw_path}/eval_$1_${nerfw_mesh_name} \
    # --save_name ${scene_abv}_$1_ous_${ours_mesh_name}_nerfw_${colmap_mesh_name}


    python utils/vis_metrics.py \
    --max_num 80 \
    --ours_path ${pred_path}/eval_$1_${ours_mesh_name} \
    --colmap_path ${vismvs_path}/eval_$1_${vismvs_mesh_name} \
    --save_name ${scene_abv}_$1_ous_${ours_mesh_name}_vismvs_${vismvs_mesh_name}


done