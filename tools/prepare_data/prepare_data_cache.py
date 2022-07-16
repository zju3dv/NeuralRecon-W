import sys

sys.path.insert(1, ".")
import argparse
from datasets import dataset_dict
import numpy as np
import os
import json
from config.defaults import get_cfg_defaults
import h5py
import time
import torch
from tqdm import tqdm


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root_dir", type=str, required=True, help="root directory of dataset"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="phototourism",
        choices=["phototourism"],
        help="which dataset to generate cache",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="cache",
        help="used as output directory of cache",
    )
    parser.add_argument(
        "--cache_type",
        type=str,
        default="h5",
        choices=["h5", "npz"],
        help="which type of cache to save",
    )
    parser.add_argument(
        "--img_downscale",
        type=int,
        default=1,
        help="how much to downscale the images for phototourism dataset",
    )
    # it will generate (args.split_to_chunks+1) chunks if the number of rays % chunk_length != 0
    parser.add_argument(
        "--split_to_chunks",
        type=int,
        default=-1,
        help="split large cache files to small chunks",
    )

    parser.add_argument(
        "--cfg_path", type=str, help="config path (only for human recon now)"
    )
    parser.add_argument(
        "--semantic_map_path",
        type=str,
        default=None,
        help="set corresponding semantic_map_path flag in phototourism dataloader",
    )

    return parser.parse_args()


def query_list_of_tensors(list_of_tensors, index):
    pointer = 0
    for i, t in enumerate(list_of_tensors):
        if pointer + t.shape[0] > index:
            position = index - pointer
            return i, position, t[position]
        pointer += t.shape[0]


def split_to_chunks(
    rgbs, all_lengths, chunk_length, split_path, args, padding_index, arr_type="rgbs"
):
    if len(padding_index) == 0:
        padding_tensor = torch.zeros((0, rgbs[0].size()[1]))
    else:
        padding_tensor = torch.cat(
            [query_list_of_tensors(rgbs, i)[2].unsqueeze(0) for i in padding_index]
        )

    padding_size = padding_tensor.shape[0]

    if padding_size > 0:
        rgbs += [padding_tensor]

    meta_info = {
        "data_length": all_lengths + padding_size,
        "chunk_length": chunk_length,
        "n_trunks": args.split_to_chunks,
    }

    current_tensor = 0
    current_index = 0
    for i, c in enumerate(tqdm(range(0, all_lengths + padding_size, chunk_length))):
        print(f"Processing {i}th chunk, from {c} to {c + chunk_length}..")
        current_chunk = []
        current_chunk_length = 0
        for tensor in rgbs[current_tensor:]:
            if current_chunk_length + tensor.shape[0] - current_index >= chunk_length:
                current_chunk += [
                    tensor[
                        current_index : current_index
                        + chunk_length
                        - current_chunk_length
                    ]
                ]
                if (
                    current_chunk_length + tensor.shape[0] - current_index
                    == chunk_length
                ):
                    current_tensor += 1
                    current_index = 0
                else:
                    current_index = current_index + chunk_length - current_chunk_length
                break
            else:
                current_chunk += [tensor[current_index:]]
                current_chunk_length += tensor[current_index:].shape[0]
                current_tensor += 1
                current_index = 0
        os.makedirs(os.path.join(split_path, f"split_{i}"), exist_ok=True)
        chunk_file = os.path.join(
            split_path,
            f"split_{i}",
            f"{arr_type}{args.img_downscale}.{args.cache_type}",
        )
        if args.cache_type == "h5":
            with h5py.File(chunk_file, "a") as f:
                is_start = True
                for index, t in enumerate(current_chunk):
                    if t.shape[0] == 0:
                        continue
                    if is_start:
                        dset = f.create_dataset(
                            arr_type,
                            (t.shape[0], t.shape[1]),
                            maxshape=(None, t.shape[1]),
                            chunks=True,
                        )
                        dset[:] = t
                        is_start = False
                    else:
                        dset.resize(dset.shape[0] + t.shape[0], axis=0)
                        dset[-t.shape[0] :] = t
        else:
            current_chunk = torch.cat(current_chunk).numpy()
            np.savez_compressed(chunk_file, current_chunk)

    with open(
        os.path.join(split_path, f"{arr_type}{args.img_downscale}_meta_info.json"), "w"
    ) as outfile:
        json.dump(meta_info, outfile)


if __name__ == "__main__":
    args = get_opts()
    os.makedirs(os.path.join(args.root_dir, args.cache_dir), exist_ok=True)
    print(f"Preparing cache for scale {args.img_downscale}...")

    with_semantic = not (args.semantic_map_path is None)
    if args.dataset_name == "phototourism":
        kwargs = {
            "root_dir": args.root_dir,
            "split": "train",
            "img_downscale": args.img_downscale,
            "with_semantics": with_semantic,
            "semantic_map_path": args.semantic_map_path,
        }
    dataset = dataset_dict[args.dataset_name](**kwargs)

    if args.split_to_chunks > 0:
        rgbs = dataset.all_rgbs
        tensor_length = []
        for r in rgbs:
            tensor_length += [r.shape[0]]
        all_lengths = sum(tensor_length)
        split_path = os.path.join(args.root_dir, args.cache_dir, "splits")
        os.makedirs(split_path, exist_ok=True)
        print(f"Loading rgb files...")
        n_trunks = args.split_to_chunks
        print(f"Splitting cache to {args.split_to_chunks} chunks...")
        padding_size = args.split_to_chunks - all_lengths % args.split_to_chunks

        if padding_size < args.split_to_chunks:
            padding_index = np.random.choice(all_lengths, padding_size, replace=False)
        elif padding_size == args.split_to_chunks:
            # no need to padding
            padding_index = np.array([]).astype(int)
            padding_size = 0

        chunk_length = (all_lengths + padding_size) // args.split_to_chunks
        print(
            f"Total length: {all_lengths}, Padding size: {padding_size}, Trunk length: {chunk_length}"
        )
        print(f"Splitting rgb...")
        split_to_chunks(
            rgbs, all_lengths, chunk_length, split_path, args, padding_index, "rgbs"
        )
        rays = dataset.all_rays
        print(f"Splitting ray...")
        split_to_chunks(
            rays, all_lengths, chunk_length, split_path, args, padding_index, "rays"
        )
    else:
        start_time = time.time()
        if args.cache_type == "h5":
            rays_file = os.path.join(
                args.root_dir, f"{args.cache_dir}/rays{args.img_downscale}.h5"
            )
            with h5py.File(rays_file, "w") as f:
                f.create_dataset("rays", data=torch.cat(dataset.all_rays).numpy(), chunks=True)
            rgbs_file = os.path.join(
                args.root_dir, f"{args.cache_dir}/rgbs{args.img_downscale}.h5"
            )
            with h5py.File(rgbs_file, "w") as f:
                f.create_dataset("rgbs", data=torch.cat(dataset.all_rgbs).numpy(), chunks=True)
        else:
            np.savez_compressed(
                os.path.join(
                    args.root_dir, f"{args.cache_dir}/rays{args.img_downscale}.npz"
                ),
                torch.cat(dataset.all_rays).numpy(),
            )
            np.savez_compressed(
                os.path.join(
                    args.root_dir, f"{args.cache_dir}/rgbs{args.img_downscale}.npz"
                ),
                torch.cat(dataset.all_rgbs).numpy(),
            )
        t = (time.time() - start_time) / 60
        print(f"Using {t} min to save!")
        print(f"Data cache saved to {os.path.join(args.root_dir, args.cache_dir)} !")
