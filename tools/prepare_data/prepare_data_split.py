import sys
sys.path.insert(1, '.')
import pandas as pd
import numpy as np
import os
import argparse
from dataset_filter_utils import NIMA_filter, view_selection, filter_image_based_on_transient_percent
import yaml

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--num_test', type=int, default=10,
                        help='size of test set')
    parser.add_argument('--min_observation', type=int, default=-1,
                        help='min_observation for view selection')
    parser.add_argument('--roi_threshold', type=float, default=0.5,
                        help='roi_threshold to filter images')
    parser.add_argument('--static_threshold', type=float, default=0.6,)  
    parser.add_argument('--nima_ckpt_path', type=str, default='weights/nima_epoch-34.pth',
                        help='ckpt path for nima model')
    parser.add_argument('--semantic_map_path', type=str, default='semantic_maps')
    return parser.parse_args()

args = get_opts()

scene_config_path = os.path.join(args.root_dir, 'config.yaml')
with open(scene_config_path, "r") as yamlfile:
    scene_config = yaml.load(yamlfile, Loader=yaml.FullLoader)

image_paths = view_selection(
    root_dir=args.root_dir,
    scene_origin=scene_config['origin'],
    scene_radius=scene_config['radius'],
    min_observation=args.min_observation,
    roi_threshold=args.roi_threshold,
)
image_paths = np.random.permutation(image_paths)[:, np.newaxis]
# image_paths = NIMA_filter(ckpt_path=args.nima_ckpt_path, root_dir=args.root_dir, image_names=image_paths[:,0])[:, np.newaxis]

# transient_objects = ['person', 'car', 'bicycle', 'minibike', 'tree', 'sky']
transient_objects = ['person', 'car', 'bicycle', 'minibike', 'tree']
# transient_objects = ['car', 'bicycle', 'minibike', 'tree', 'sky']
image_paths = filter_image_based_on_transient_percent(
    root_dir=args.root_dir,
    semantic_map_path=args.semantic_map_path,
    image_paths=image_paths,
    transient_objects=transient_objects,
    static_threshold=args.static_threshold,
)

num_img = image_paths.shape[0]
pseudo_image_ids = np.arange(num_img)[:, np.newaxis]
img_label_test = np.repeat('test', args.num_test)
img_label_train = np.repeat('train', num_img - args.num_test)
img_label = np.concatenate((img_label_test, img_label_train), 0)[:, np.newaxis]
img_dataset = np.repeat(args.root_dir.split('/')[-1], num_img)[:, np.newaxis]
content = np.concatenate((image_paths, pseudo_image_ids, img_label, img_dataset), 1)
df = pd.DataFrame(content, columns=['filename', 'id', 'split', 'dataset'])
df.to_csv(os.path.join(args.root_dir, f'{args.root_dir.split("/")[-1]}.tsv'), sep = '\t', index=False)