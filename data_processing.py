from my_utils import make_gps, construct_M1, construct_M2, make_hist_2d, load_latlon_range
import json
import pathlib
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from my_utils import get_datadir


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--n_bins', default=1, type=int)
    parser.add_argument('--save_name', default="test_1", type=str)
    parser.add_argument('--latlon_name', default="matt_geolife", type=str)
    args = parser.parse_args()
    
    data_path = get_datadir() / args.dataset / args.save_name
    with open(data_path / "params.json", "w") as f:
        json.dump(vars(args), f)
        
    training_data = pd.read_csv(data_path / "training_data.csv", header=None).values
    
    lat_range, lon_range = load_latlon_range(args.latlon_name)
    max_locs = (args.n_bins+2)**2

    if not (data_path/"gps.csv").exists():
        gps = make_gps(lat_range, lon_range, args.n_bins)
        gps.to_csv(data_path / f"gps.csv", header=None, index=None)
        print(gps)
    else:
        print("GPS exists")

    if not (data_path/"M1.npy").exists():
        M1 = construct_M1(training_data, max_locs)
        np.save(data_path/f'M1.npy',M1)
        print(M1)
    else:
        print("M1 exists")
        
    if not (data_path/"M2.npy").exists():
        gps = pd.read_csv(data_path/"gps.csv", header=None)
        M2 = construct_M2(training_data, max_locs, gps)
        np.save(data_path/f'M2.npy',M2)
        print(M2)
    else:
        print("M2 exists")


    args.traj_length = len(training_data[0])
    for hour in range(args.traj_length+1):
        if hour == args.traj_length:
            hour = "all"
            training_counts = np.bincount(training_data.reshape(-1), minlength=max_locs)
        else:
            training_counts = np.bincount(training_data[:, hour].reshape(-1), minlength=max_locs)
        training_hist2d = make_hist_2d(training_counts, args.n_bins)

        ax = sns.heatmap(training_hist2d, cmap=sns.color_palette("light:b", as_cmap=True), vmax=3)
        ax.invert_yaxis()

        ax.figure.savefig(data_path/f'{hour}.png')
        plt.clf()