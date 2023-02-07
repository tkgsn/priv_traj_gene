from my_utils import make_gps, construct_M1, construct_M2, make_hist_2d, load_latlon_range, latlon_to_state
import json
import pathlib
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from my_utils import get_datadir
import glob
import tqdm
from datetime import datetime
from bisect import bisect_left

peopleflow_raw_data_dir = "/data/peopleflow/tokyo2008/p-csv/0000/*.csv"
# peopleflow_raw_data_dir = "/data/peopleflow/tokyo2008/p-csv/test/*.csv"
format = '%H:%M:%S'
basic_time = datetime.strptime("00:00:00", format)

def str_to_minute(time_str):
    format = '%H:%M:%S'
    return int((datetime.strptime(time_str, format) - basic_time).seconds / 60)
    
def load(n_bins, lat_range, lon_range):

    files = glob.glob(peopleflow_raw_data_dir)

    lat_index = 5
    lon_index = 4
    dataset = []
    times = []

    for file in tqdm.tqdm(files):
        trajectory = []
        time_trajectory = []
        df = pd.read_csv(file, header=None)
        
        previous_time = -1
        for record in df.iterrows():
            record = record[1]
            
            time = str_to_minute(record[3].split(" ")[1])
            if previous_time > time:
                break
            else:
                previous_time = time
            
            state = latlon_to_state(record[lat_index], record[lon_index], lat_range, lon_range, n_bins)
            trajectory.append(state)
            time_trajectory.append(time)
        dataset.append(trajectory)
        times.append(time_trajectory)

    return dataset, times

def split(time, seq_len, start_hour, end_hour):
    start_time = start_hour * 60
    end_time = end_hour * 60
    
    time_range = (end_time - start_time) / seq_len
    target_times = [i*time_range for i in range(seq_len)]
    
    split_indices = []
    for target_time in target_times:
        split_indices.append(bisect_left(time, target_time))
        
    return split_indices
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default="test_1", type=str)
    args = parser.parse_args()
    
    with open(pathlib.Path("./") / "dataset_configs" / args.config_name, "r") as f:
        configs = json.load(f)
    
    data_path = get_datadir() / configs["dataset"] / configs["save_name"]
    data_path.mkdir(exist_ok=True)
    with open(data_path / "params.json", "w") as f:
        json.dump(vars(args), f)
    
    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]
    start_hour = configs["start_hour"]
    end_hour = configs["end_hour"]
    n_bins = configs["n_bins"]
    seq_len = configs["seq_len"]
    
    max_locs = (n_bins+2)**2

    if not (data_path / "training_data.csv").exists():
        dataset = []
        
        locations, times = load(n_bins, lat_range, lon_range)
        for location, time in zip(locations, times):

            split_indices = split(time, seq_len, start_hour, end_hour)
            dataset.append(np.array(location)[split_indices])
            
        training_data = pd.DataFrame(dataset).to_csv(data_path / "training_data.csv", index=None)
    else:
        training_data = pd.read_csv(data_path / "training_data.csv", header=None).values
    
    
    if not (data_path/"gps.csv").exists():
        gps = make_gps(lat_range, lon_range, n_bins)
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