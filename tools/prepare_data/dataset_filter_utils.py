import os
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from utils.colmap_utils import \
    read_cameras_binary, read_images_binary
from datasets.ray_utils import *
from datasets.mask_utils import get_label_id_mapping
import shutil

"""
Copyright (C) Yunxiao Shi 2017 - 2021
NIMA is released under the MIT license. See LICENSE for the fill license text.
"""
class NIMA(nn.Module):

    """Neural IMage Assessment model by Google"""
    def __init__(self, base_model, num_classes=10):
        super(NIMA, self).__init__()
        self.features = base_model.features
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.75),
            nn.Linear(in_features=25088, out_features=num_classes),
            nn.Softmax())

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

def NIMA_filter(ckpt_path, root_dir, image_names, ):
    print('filter based on NIMA score...')
    image_dir = os.path.join(root_dir, 'dense/images')

    base_model = models.vgg16(pretrained=True)
    model = NIMA(base_model)

    try:
        model.load_state_dict(torch.load(ckpt_path))
        print('successfully loaded model')
    except Exception as e:
        e.args = "NIMA model not found, did you run scripts/download_weights.sh? :\n%s" % e.args

    seed = 42
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    model.eval()

    test_transform = transforms.Compose([
        transforms.Scale(256), 
        transforms.RandomCrop(224), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        ])
    
    test_imgs = image_names
    pbar = tqdm(total=len(test_imgs))

    mean, std = 0.0, 0.0

    filtered_images = []
    for i, img in enumerate(test_imgs):
        im = Image.open(os.path.join(image_dir, str(img)))
        ori_img = img
        
        im = im.convert('RGB')
        imt = test_transform(im)
        imt = imt.unsqueeze(dim=0)
        imt = imt.to(device)
        with torch.no_grad():
            out = model(imt)
        out = out.view(10, 1)
        for j, e in enumerate(out, 1):
            mean += j * e
        for k, e in enumerate(out, 1):
            std += e * (k - mean) ** 2
        std = std ** 0.5
        if mean >= 3:
            filtered_images.append(img)
        else:
            ori_img.save(os.path.join(root_dir, 'trash_images', f'NIMA_filter_{str(img)}'))
        mean, std = 0.0, 0.0
        pbar.update()
    print(f'filter {len(test_imgs)-len(filtered_images)} images in NIMA filtering process')
    
    return np.array(filtered_images)

def view_selection(root_dir, scene_origin, scene_radius, min_observation, roi_threshold=0.2):
    
    scene_origin = torch.tensor(scene_origin).cuda()
    imdata = read_images_binary(os.path.join(root_dir, 'dense/sparse/images.bin'))
    img_path_to_id = {}
    image_paths = {}
    img_ids = []
    for v in imdata.values():
        img_ids += [v.id]
        img_path_to_id[v.name] = v.id
        image_paths[v.id] = v.name

    Ks = {} # {id: K}
    camdata = read_cameras_binary(os.path.join(root_dir, 'dense/sparse/cameras.bin'))
    for id_ in img_ids:
        K = np.zeros((3, 3), dtype=np.float32)
        cam = camdata[imdata[id_].camera_id]
        K[0, 0] = cam.params[0]
        K[1, 1] = cam.params[1]
        K[0, 2] = cam.params[2]
        K[1, 2] = cam.params[3]
        K[2, 2] = 1
        Ks[id_] = K

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
    for id_ in img_ids:
        im = imdata[id_]
        R = im.qvec2rotmat()
        t = im.tvec.reshape(3, 1)
        w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
    w2c_mats = np.stack(w2c_mats, 0) # (N_images, 4, 4)
    poses = np.linalg.inv(w2c_mats)[:, :3] # (N_images, 3, 4)
    # Original poses has rotation in form "right down front", change to "right up back"
    poses[..., 1:3] *= -1
    poses_dict = {id_: poses[i] for i, id_ in enumerate(img_ids)}
   
    shutil.rmtree(os.path.join(root_dir, 'trash_images'), ignore_errors=True)
    os.makedirs(os.path.join(root_dir, 'trash_images'))
    if min_observation > 0:
        print('view_selection_based_on observations...')
        sparse_filtered_path = os.path.join(root_dir, f'dense/sparse_filtered_{min_observation}')
        imgs_filtered_data = read_images_binary(os.path.join(sparse_filtered_path, 'images.bin'))
        img_id_filtered = []
        for v in imgs_filtered_data.values():
            img_id_filtered += [v.id]
        img_less_covis = set(img_ids) - set(img_id_filtered)
        for img_id in img_less_covis:
            cam = camdata[imdata[img_id].camera_id]
            if (cam.params[0]>2000) and (cam.params[1]>2000):
                img_id_filtered += [img_id]
        trash_imgs_id = set(img_ids) - set(img_id_filtered)
        for v_id in trash_imgs_id:
            img = Image.open(os.path.join(root_dir, 'dense/images', image_paths[v_id])).convert('RGB')
            img.save(os.path.join(root_dir, 'trash_images', f'less_covis_{image_paths[v_id]}'))
        print('view_selection_based_on observations Done')
        print(f'filter {len(img_ids)-len(img_id_filtered)} images in view_selection_based_on_observations')
        img_ids = img_id_filtered

    print('view_selection_based_on_ROI...')
    transform = transforms.ToTensor()
    filtered_images = []
    for id_ in tqdm(img_ids):
        c2w = torch.FloatTensor(poses_dict[id_]).cuda()
        img = Image.open(os.path.join(root_dir, 'dense/images',
                                        image_paths[id_])).convert('RGB')
        ori_img = img
        img_w, img_h = img.size
        img = transform(img) # (3, h, w)
        img = img.view(3, -1).permute(1, 0) # (h*w, 3) RGB
        directions = get_ray_directions(img_h, img_w, Ks[id_]).cuda()
        rays_o, rays_d = get_rays(directions, c2w)  # (H*W, 3)

        camera2origin = scene_origin - rays_o # (H*W, 3)
        dot_product = torch.sum((scene_origin - rays_o) * rays_d, -1, keepdim=True) # (H*W, 1)
        camera2origin_project = dot_product * rays_d # (H*W, 3)
        dist_ray_origin = torch.linalg.norm((camera2origin - camera2origin_project), ord=2, dim=-1) # (H*W)
        dist_camera2origin = torch.linalg.norm(camera2origin, ord=2, dim=-1) # (H*W)
        if_ROI = torch.logical_and(
            torch.logical_or((scene_radius > dist_camera2origin), dot_product.squeeze(-1)>0), (dist_ray_origin < scene_radius))
        ROI_percent = torch.count_nonzero(if_ROI) / if_ROI.shape[0]
        if ROI_percent < roi_threshold:
            ori_img.save(os.path.join(root_dir, 'trash_images', f'less_ROI_{image_paths[id_]}'))
        else:
            filtered_images += [image_paths[id_]]
    print(f'filter {len(img_ids)-len(filtered_images)} images in view_selection_based_on_ROI')
    return np.array(filtered_images)

def filter_image_based_on_transient_percent(root_dir, semantic_map_path, image_paths, transient_objects, static_threshold=0.3):
    print('filter_image_based_on_transient_percent...')

    filtered_images = []
    for image_name in tqdm(image_paths[:,0]):
        semantic_map = np.load(os.path.join(root_dir, f'{semantic_map_path}/{image_name.split(".")[0]}.npz'))['arr_0']
        static_mask = np.ones_like(semantic_map)
        for label_name in transient_objects:
            static_mask[get_label_id_mapping()[label_name] == semantic_map] = False
        static_percent = np.count_nonzero(static_mask) / (static_mask.shape[0]*static_mask.shape[1])
        if static_percent > static_threshold:
            filtered_images.append(image_name)
        else:
            ori_img = Image.open(os.path.join(root_dir, 'dense/images',
                                image_name)).convert('RGB')
            ori_img.save(os.path.join(root_dir, 'trash_images', f'transient_much_{image_name}'))
            
    filtered_images = np.array(filtered_images)[:, np.newaxis]
    print(f'filter {image_paths.shape[0]-filtered_images.shape[0]} images in transient filtering process')
    return filtered_images
