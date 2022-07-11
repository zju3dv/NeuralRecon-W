import sys

sys.path.insert(1, '.')
import argparse
import os
import glob
import pandas as pd
from utils.colmap_utils import read_images_binary
from mmseg.apis import inference_segmentor, init_segmentor
from tqdm import tqdm
import numpy as np
from PIL import Image

def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which gpu to run')

    return parser.parse_args()

# deeplabv3 config file path
config_file = 'config/deeplabv3_config/deeplabv3_r101-d8_512x512_160k_ade20k.py'
checkpoint_file = 'weights/deeplabv3_r101-d8_512x512_160k_ade20k_20200615_105816-b1f72b3b.pth'

if __name__ == '__main__':
    args = get_opts()

    os.makedirs(os.path.join(args.root_dir, f'semantic_maps'), exist_ok=True)
    os.makedirs(os.path.join(args.root_dir, f'segmentation_vis'), exist_ok=True)

    print(f'Preparing semantic maps for {args.root_dir.split("/")[-1]} set...')

    image_paths = os.listdir(os.path.join(args.root_dir, 'dense/images'))

    # build the DeepLabv3 model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device=f'cuda:{args.gpu}')

    for img_path in tqdm(image_paths):
        img = Image.open(os.path.join(args.root_dir, 'dense/images', img_path))
        image_name = img_path.split('.')[0]
        img_w, img_h = img.size
        img = np.array(img)
        result = inference_segmentor(model, img)[0] # (H, W)
        model.show_result(img, [result], out_file=os.path.join(args.root_dir, f'segmentation_vis/{image_name}.png'), opacity=0.5)
        np.savez_compressed(os.path.join(
            args.root_dir, f'semantic_maps/{image_name}.npz'), result)
        
