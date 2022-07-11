#!/bin/bash
set -x
set -u
set -e

now=$(date +"%Y%m%d_%H%M%S")
echo "working directory is $(pwd)"
jobname="data-generation-$now"

################################################################
# Replace following variables
# colmap workspace should be in ${SRC_PATH}/${SCENE_NAME}
# output is in ${DEST_PATH}/${SCENE_NAME}
SRC_PATH='/data/phototourism/training_set' 
SCENE_NAME='brandenburg_gate'
DEST_PATH='/data/recon_data'

# colmap sparse folder and image folder 
COLMAP_PATH='dense/sparse'
IMG_PATH='dense/images'

# If you have multiple sub images folder, change SPLIT_TYPE to "plain" to reconstruct them seperatly, 
# or group them as "folder1#folder2, folder3", folder1 and folder2 will results in one output, folder3 in another
SPLIT_TYPE="none"
###############################################################

python tools/pre_process.py --src ${SRC_PATH}/${SCENE_NAME} \
--dest ${DEST_PATH} \
--split ${SPLIT_TYPE} \
--img_dir ${IMG_PATH} \
--colmap_dir ${COLMAP_PATH}

# clear temp folder
rm -rf ${DEST_PATH}/${SCENE_NAME}/undistort
export CUDA_VISIBLE_DEVICES=0
for scene_name in ${DEST_PATH}/${SCENE_NAME}/* ; do
    root_dir=${scene_name}
    echo processing scene ${root_dir}
    python \
    tools/prepare_data/prepare_semantic_maps.py \
    --root_dir ${root_dir} 2>&1|tee log/${jobname}_preprocess.log

    min_observation=-1

    python tools/prepare_data/prepare_data_split.py \
    --root_dir ${root_dir}\
    --num_test 10 \
    --min_observation $min_observation --roi_threshold 0 --static_threshold 0

    dataset_name="phototourism"
    cache_dir="cache_sgs"

    python tools/prepare_data/prepare_data_cache.py \
    --root_dir ${root_dir} \
    --dataset_name $dataset_name --cache_dir $cache_dir \
    --img_downscale 1 \
    --semantic_map_path semantic_maps --split_to_chunks 64 \
    2>&1|tee log/${jobname}.log
done