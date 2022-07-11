import matplotlib.pyplot as plt
import numpy as np
import os
import json
from argparse import ArgumentParser

def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--max_num', type=int,
                        help='max number of data point to plot')
    parser.add_argument('--ours_path', type=str,
                        help='ours eval result folder') 
    parser.add_argument('--colmap_path', type=str,
                        help='colmap eval result folder')
    parser.add_argument('--save_name', type=str,
                        help='path to save plot result') 

    return parser.parse_args()   


def svae_plot(ind, data1, data2, name1, name2, save_path, name):
    plt.plot(ind, np.array(data1) * 100, "-b", label=name1)
    plt.plot(ind, np.array(data2) * 100, "-r", label=name2)
    plt.legend(loc="upper left")
    plt.title(name)
    plt.xlabel("thresholds(m)")
    plt.ylabel("score")
    plt.ylim(0, 1.0 * 100)
    plt.savefig(os.path.join(save_path, f"{name}.png"))
    plt.show()
    plt.clf()


def vis_results(ours_path, colmap_path, save_name, max_num):
    rslt_file = os.path.join(ours_path, 'metrics.json')
    ours_metrics = json.load(open(rslt_file, 'r'))

    rslt_file = os.path.join(colmap_path, 'metrics.json')
    colmap_metrics = json.load(open(rslt_file, 'r'))

    save_path = os.path.join("eval_results", save_name, save_name)

    thresholds = ours_metrics['thresholds'][:max_num]
    del ours_metrics['thresholds']

    save_path = os.path.join("eval_results", f"{save_name}")
    os.makedirs(save_path, exist_ok=True)
    for key in ours_metrics.keys():
        svae_plot(thresholds, ours_metrics[key][:max_num],  colmap_metrics[key][:max_num], \
                    "ours", "colmap", save_path, f"{key}")

if __name__ == "__main__":
    args = get_opts()
    vis_results(args.colmap_path, args.ours_path, args.save_name, args.max_num)