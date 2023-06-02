from my_utils import make_gps, construct_M1, construct_M2, make_hist_2d, load_latlon_range, latlon_to_state
import json
import pathlib
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from my_utils import get_datadir
from grid import Grid
import glob
import tqdm
from datetime import datetime
from bisect import bisect_left
from evaluation import plot_hist2d
from geopy.distance import geodesic

# peopleflow_raw_data_dir = "/data/peopleflow/tokyo2008/p-csv/0000/*.csv"
format = '%H:%M:%S'
basic_time = datetime.strptime("00:00:00", format)

# check if the location is in the range
def in_range(lat_range, lon_range, lat, lon):
    return float(lat_range[0]) <= lat <= float(lat_range[1]) and float(lon_range[0]) <= lon <= float(lon_range[1])


def make_raw_data(dataset_name):
    data_path = get_datadir() / dataset_name
    if (dataset_name == "geolife"):

        save_path = data_path / "file_base_in_border_time.csv"
        if not save_path.exists():
            geolife_raw_path = pathlib.Path("/data/geolife/Data")
            print(f"make raw data from {geolife_raw_path}")
            trajectories = make_raw_data_geolife(geolife_raw_path)
            save_timelatlon_with_nan_padding(save_path, trajectories)
        else:
            print(f"exist raw data in {save_path}")
    
        save_path = data_path / "file_base_in_border.csv"
        if not save_path.exists():
            print("make raw data without time info of Geolife")
            save_latlon_with_nan_padding(save_path, trajectories)

    if dataset_name == "geolife_test":

        save_path = data_path / "file_base_in_border_time.csv"
        if not save_path.exists():
            geolife_raw_path = pathlib.Path("/data/geolife_test/Data")
            print(f"make raw data from {geolife_raw_path}")
            trajectories = make_raw_data_geolife(geolife_raw_path)
            save_timelatlon_with_nan_padding(save_path, trajectories)
        else:
            print(f"exist raw data in {save_path}")
    

        save_path = data_path / "file_base_in_border.csv"
        if not save_path.exists():
            print("make raw data without time info of Geolife")
            save_latlon_with_nan_padding(save_path, trajectories)

    if dataset_name == "peopleflow_test":
        peopleflow_raw_data_dir = "/data/peopleflow/tokyo2008/p-csv/test/*.csv"
        save_path = data_path / "file_base_in_border_time_0000.csv"
        if not save_path.exists():
            print("make raw data of Peopleflow")
            trajectories = make_raw_data_peopleflow(peopleflow_raw_data_dir)
            save_timelatlon_with_nan_padding(save_path, trajectories)
        else:
            print(f"exist raw data in {save_path}")
    
        save_path = data_path / "file_base_in_border_0000.csv"
        if not save_path.exists():
            print("make raw data without time info of Peopleflow")
            save_latlon_with_nan_padding(save_path, trajectories)

    if dataset_name == "peopleflow":
        peopleflow_raw_data_dir = "/data/peopleflow/tokyo2008/p-csv/0000/*.csv"
        save_path = data_path / "file_base_in_border_time_0000.csv"
        if not save_path.exists():
            print("make raw data of Peopleflow")
            trajectories = make_raw_data_peopleflow(peopleflow_raw_data_dir)
            save_timelatlon_with_nan_padding(save_path, trajectories)
        else:
            print(f"exist raw data in {save_path}")
    
        save_path = data_path / "file_base_in_border_0000.csv"
        if not save_path.exists():
            print("make raw data without time info of Peopleflow")
            save_latlon_with_nan_padding(save_path, trajectories)

    if dataset_name == "peopleflow_":
        # make raw data for each number (0000-0027)
        for i in range(28):
            peopleflow_raw_data_dir = f"/data/peopleflow/tokyo2008/p-csv/{i:04d}/*.csv"
            save_path = data_path / f"file_base_in_border_time_{i:04d}.csv"
            if not save_path.exists():
                print(f"make raw data of Peopleflow {save_path}")
                trajectories = make_raw_data_peopleflow(peopleflow_raw_data_dir)
                save_timelatlon_with_nan_padding(save_path, trajectories)
            else:
                print(f"exist raw data in {save_path}")
        
            save_path = data_path / f"file_base_in_border_{i:04d}.csv"
            if not save_path.exists():
                print("make raw data without time info of Peopleflow")
                save_latlon_with_nan_padding(save_path, trajectories)

        # peopleflow_raw_data_dir = "/data/peopleflow/tokyo2008/p-csv/0000/*.csv"
        # save_path = data_path / "file_base_in_border_time.csv"
        # if not save_path.exists():
        #     print("make raw data of Peopleflow")
        #     trajectories = make_raw_data_peopleflow(peopleflow_raw_data_dir)
        #     save_timelatlon_with_nan_padding(save_path, trajectories)
        # else:
        #     print(f"exist raw data in {save_path}")
    
        # save_path = data_path / "file_base_in_border.csv"
        # if not save_path.exists():
        #     print("make raw data without time info of Peopleflow")
        #     save_latlon_with_nan_padding(save_path, trajectories)

    if dataset_name == "taxi":

        save_path = pathlib.Path("/data/taxi/raw_data.csv")
        if save_path.exists():
            print(f"exist raw data in {save_path}")
        else:
            print(f"make raw data of taxi")
            make_raw_data_taxi(save_path)

    if dataset_name == "taxi_full":

        save_path = pathlib.Path("/data/taxi_full/file_base_in_border_time_0000.csv")
        if save_path.exists():
            print(f"exist raw data in {save_path}")
        else:
            print(f"make raw data of taxi")
            make_raw_data_taxi(save_path, full=True)

## Geolife
def make_raw_data_geolife(geolife_raw_path):

    print("load config of geolife from matt_geolife.json")
    with open("/root/movesim/dataset_configs/matt_geolife.json", "r") as f:
        configs = json.load(f)

    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]

    # get users from the directory
    users = [p.name for p in geolife_raw_path.iterdir() if p.is_dir()]

    trajectories = []
    for user in users:
        print(user)
        user_path = geolife_raw_path.joinpath(user)
        # get the dates from the directory
        dates = [p.name for p in user_path.iterdir() if p.is_dir()]
        for date in dates:
            print(date)
            date_path = user_path.joinpath(date)
            # get the files from the directory
            files = glob.glob(str(date_path.joinpath("*.plt")))
            for file in files:
                trajectory = []
                print(file)
                is_valid = True
                with open(file, "r") as f:
                    # skip 6 lines
                    for _ in range(6):
                        f.readline()
                    for line in f:
                        record = line.strip().split(",")
                        lat, lon = float(record[0]), float(record[1])
                        is_valid = is_valid and in_range(lat_range, lon_range, lat, lon)
                        trajectory.append(("-".join([record[5],record[6]]), lat, lon))
                if is_valid:
                    trajectories.append(trajectory)
        
    return trajectories


## Peopleflow
def make_raw_data_peopleflow(peopleflow_raw_data_dir):
    
    print("load config of peopleflow from peopleflow_test.json")
    with open("/root/movesim/dataset_configs/peopleflow_test.json", "r") as f:
        configs = json.load(f)

    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]

    files = glob.glob(peopleflow_raw_data_dir)

    lat_index = 5
    lon_index = 4
    time_index = 3
    trajectories = []

    for file in tqdm.tqdm(files):
        trajectory = []
        df = pd.read_csv(file, header=None)
        is_valid = True
        time = 0
        for record in df.iterrows():
            record = record[1]
            lat, lon = float(record[lat_index]), float(record[lon_index])
            if time > str_to_minute(record[time_index].split(" ")[1]):
                break
            else:
                time = str_to_minute(record[time_index].split(" ")[1])
            is_valid = is_valid and in_range(lat_range, lon_range, lat, lon)
            trajectory.append((time, lat, lon))
        if is_valid:
            trajectories.append(trajectory)

    return trajectories

def make_raw_data_taxi(save_path, full=False):

    raw_data_path = pathlib.Path('/data/taxi/raw_data.csv')
    if not raw_data_path.exists():
            
        # read header of csv of /data/taxi/raw/train.csv
        header = pd.read_csv('/data/taxi/raw/train.csv', nrows=0)

        # count the number of lines of csv of /data/taxi/raw/train.csv
        with open('/data/taxi/raw/train.csv') as f:
            for i, l in enumerate(f):
                pass
        count = i + 1


        # read csv of /data/taxi/raw/train.csv by max_data_length
        max_data_length = 10000

        dfs = []
        for i in range(count // max_data_length + 1):
            if i == 0:
                df = pd.read_csv('/data/taxi/raw/train.csv', nrows=max_data_length, header=None)
            else:
                df = pd.read_csv('/data/taxi/raw/train.csv', skiprows=max_data_length * i, nrows=max_data_length, header=None)
            df.columns = header.columns
            dfs.append(df)

        # convert the list to the format
        # [[lon,lat],...] -> [[time,lat,lon],...]
        # the time starts from 0 and the unit is minute

        def convert_to_list_of_points(polyline):
            if polyline == []:
                return []
            else:
                return [[i*15/60,point[1],point[0]] for i,point in enumerate(polyline)]

        for df in dfs:

            # remove the record with missing data
            df = df[df['MISSING_DATA'] == False]

            # convert the trajectory data to a list of points
            # the trajectory data is string of the form "[[x1,y1],[x2,y2],...,[xn,yn]]"
            # the list of points is a list of tuples (x,y)
            df["POLYLINE"] = df["POLYLINE"].apply(lambda x: eval(x))

            for i in df.index:
                trajectories.append(convert_to_list_of_points(df["POLYLINE"][i]))

        save_timelatlon_with_nan_padding(save_path, trajectories)

    # count the number of records in /data/taxi/raw_data.csv
    with open('/data/taxi/raw_data.csv', 'r') as f:
        n_data = len(f.readlines())

    n_records_for_each = 10000

    # read each data form the csv of /data/taxi/raw_data.csv with n_records_for_each rows 
    for i in tqdm.tqdm(range(0, n_data, n_records_for_each)):
        df = pd.read_csv('/data/taxi/raw_data.csv', header=None, skiprows=i, nrows=n_records_for_each)
        if full:
            save_path = pathlib.Path(f'/data/taxi_full/file_base_in_border_time_{int(i/n_records_for_each):04d}.csv')
        else:
            save_path = pathlib.Path(f'/data/taxi/file_base_in_border_time_{int(i/n_records_for_each):04d}.csv')

        trajs = []
        for record in df.values.tolist():
            if type(record[0]) != str:
                continue

            def value_to_time_lat_lon(value):
                time, lat, lon = list(map(float, value.split(" ")))
                return time, lat, lon

            if full:
                # remove nan
                record = [v for v in record if type(v) == str]
                traj = [value_to_time_lat_lon(v) for v in record]
            else:
                start = record[0]
                # find the index of end where the value is not nan
                end = [v for v in record if type(v) == str][-1]

                start_time, start_lat, start_lon = value_to_time_lat_lon(start)
                end_time, end_lat, end_lon = value_to_time_lat_lon(end)
                traj = [(start_time, start_lat, start_lon), (end_time, end_lat, end_lon)]
                
            trajs.append(traj)
        
        save_timelatlon_with_nan_padding(save_path, trajs)



# def make_raw_data_peopleflow(n_bins, lat_range, lon_range):
        # previous_time = -1
        # for record in df.iterrows():
        #     record = record[1]
            
        #     time = str_to_minute(record[3].split(" ")[1])
        #     if previous_time > time:
        #         break
        #     else:
        #         previous_time = time
            
        #     state = latlon_to_state(record[lat_index], record[lon_index], lat_range, lon_range, n_bins)
        #     trajectory.append(state)
        #     time_trajectory.append(time)
        # dataset.append(trajectory)
        # times.append(time_trajectory)


def make_stay_trajectory(trajectories, time_threshold, location_threshold):

    print(f"make stay trajectory with threshold {location_threshold}m and {time_threshold}min")

    stay_trajectories = []
    time_trajectories = []
    for trajectory in tqdm.tqdm(trajectories):

        stay_trajectory = []
        # remove nan
        trajectory = [v for v in trajectory if type(v) is str]
        time_trajectory = []

        start_index = 0
        start_time = 0
        i = 0
        while True:
            # find the length of the stay
            start_location = trajectory[start_index].split(" ")
            start_location = (float(start_location[1]), float(start_location[2]))

            if i == len(trajectory)-1:
                time_trajectory.append((start_time, time))
                stay_trajectory.append(target_location)
                # print("finish", start_time, time, start_location)
                break

            for i in range(start_index+1, len(trajectory)):

                target_location = trajectory[i].split(" ")
                time = float(target_location[0])
                target_location = (float(target_location[1]), float(target_location[2]))
                distance = geodesic(start_location, target_location).meters
                if distance > location_threshold:
                    # print(f"move {distance}m", start_time, time, trajectory[i])
                    if time - start_time >= time_threshold:
                        # print("stay", start_time, time, start_location)
                        stay_trajectory.append(start_location)
                        time_trajectory.append((start_time, time))

                    start_time = time
                    # print(trajectory[i])
                    start_index = i
                    # print("start", start_time, start_index, len(trajectory))
                    # print(time, i)

                    break
        
        stay_trajectories.append(stay_trajectory)
        time_trajectories.append(time_trajectory)
    return time_trajectories, stay_trajectories


def make_complessed_dataset(time_trajectories, trajectories, grid):
    dataset = []
    times = []
    for trajectory, time_trajectory in tqdm.tqdm(zip(trajectories, time_trajectories)):
        state_trajectory = []
        for lat, lon in trajectory:
        # for record in trajectory:
            # skip if record is not valid
            # if type(record) is not str:
            #     continue
            # record = record.split(" ")
            # lat, lon = float(record[0]), float(record[1])

            # convert lat, lon to state
            state = grid.latlon_to_state(lat, lon)
            state_trajectory.append(state)

        # compless time trajectory according to state trajectory
        complessed_time_trajectory = []
        j = 0
        for i, time in enumerate(time_trajectory):
            if i != j:
                continue   
            target_state = state_trajectory[i]
            # find the max length of the same states
            for j in range(i+1, len(state_trajectory)+1):
                if j == len(state_trajectory):
                    break
                if (state_trajectory[j] != target_state):
                    break
            complessed_time_trajectory.append((time[0],time_trajectory[j-1][1]))

        # remove consecutive same states
        state_trajectory = [state_trajectory[0]] + [state_trajectory[i] for i in range(1, len(state_trajectory)) if state_trajectory[i] != state_trajectory[i-1]]
        dataset.append(state_trajectory)
        times.append(complessed_time_trajectory)

        assert len(state_trajectory) == len(complessed_time_trajectory), f"state trajectory length {len(state_trajectory)} != time trajectory length {len(complessed_time_trajectory)}"
        # times.append([time for time, _, _ in trajectory])
    return dataset, times
    # return dataset, times



def str_to_minute(time_str):
    format = '%H:%M:%S'
    return int((datetime.strptime(time_str, format) - basic_time).seconds / 60)
    

def split(time, seq_len, start_hour, end_hour):
    start_time = start_hour * 60
    end_time = end_hour * 60
    
    time_range = (end_time - start_time) / seq_len
    target_times = [i*time_range for i in range(seq_len)]
    
    split_indices = []
    for target_time in target_times:
        split_indices.append(bisect_left(time, target_time))
        
    return split_indices

def save_with_nan_padding(save_path, trajectories, formater):
    # compute the max length in trajectories
    max_len = max([len(trajectory) for trajectory in trajectories])

    print(f"save to {save_path}")
    with open(save_path, "w") as f:
        for trajectory in trajectories:
            for record in trajectory:
                f.write(formater(record))
            # padding with nan
            for _ in range(max_len - len(trajectory)):
                f.write(",")
            f.write("\n")

def save_timelatlon_with_nan_padding(save_path, trajectories):
    def formater(record):
        return f"{record[0]} {record[1]} {record[2]},"
    
    save_with_nan_padding(save_path, trajectories, formater)
    # compute the max length in trajectories
    # max_len = max([len(trajectory) for trajectory in trajectories])

    # print(f"save to {save_path}")
    # with open(save_path, "w") as f:
    #     for trajectory in trajectories:
    #         for record in trajectory:
    #             f.write(f"{record[0]} {record[1]} {record[2]},")
    #         # padding with nan
    #         for _ in range(max_len - len(trajectory)):
    #             f.write(",")
    #         f.write("\n")

def save_latlon_with_nan_padding(save_path, trajectories):
    def formater(record):
        return f"{record[1]} {record[2]},"
    
    save_with_nan_padding(save_path, trajectories, formater)

    # # compute the max length in trajectories
    # max_len = max([len(trajectory) for trajectory in trajectories])

    # print(f"save to {save_path}")
    # with open(save_path, "w") as f:
    #     for trajectory in trajectories:
    #         for record in trajectory:
    #             f.write(f"{record[1]} {record[2]},")
    #         # padding with nan
    #         for _ in range(max_len - len(trajectory)):
    #             f.write(",")
    #         f.write("\n")

def save_state_with_nan_padding(save_path, trajectories):
    def formater(record):
        return f"{record},"
    
    save_with_nan_padding(save_path, trajectories, formater)

    # # compute the max length in trajectories
    # max_len = max([len(trajectory) for trajectory in trajectories])

    # print(f"save to {save_path}")
    # with open(save_path, "w") as f:
    #     for trajectory in trajectories:
    #         for state in trajectory:
    #             f.write(f"{state},")
    #         # padding with nan
    #         for _ in range(max_len - len(trajectory)):
    #             f.write(",")
    #         f.write("\n")



def save_time_with_nan_padding(save_path, trajectories, max_time):
    def formater(record):
        return f"{record[0]},"
    
    for trajectory in trajectories:
        trajectory.append([max_time])
    
    save_with_nan_padding(save_path, trajectories, formater)

# def save_time_with_nan_padding(save_path, trajectories):
#     def formater(record):
#         return f"{record[0]}_{record[1]},"
    
#     save_with_nan_padding(save_path, trajectories, formater)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default="test_1", type=str)
    args = parser.parse_args()
    
    with open(pathlib.Path("./") / "dataset_configs" / args.config_name, "r") as f:
        configs = json.load(f)
    
    data_path = get_datadir() / configs["dataset"] / configs["save_name"]
    data_path.mkdir(exist_ok=True, parents=True)

    make_raw_data(configs["dataset"])

    with open(data_path / "params.json", "w") as f:
        json.dump(configs, f)
        
    lat_range = configs["lat_range"]
    lon_range = configs["lon_range"]
    # start_hour = configs["start_hour"]
    # end_hour = configs["end_hour"]
    n_bins = configs["n_bins"]
    # seq_len = configs["seq_len"]
    time_threshold = configs["time_threshold"]
    location_threshold = configs["location_threshold"]

    max_locs = (n_bins+2)**2

    print("make grid", lat_range, lon_range, n_bins)
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid = Grid(ranges)

    # list the files that start file_base_in_border_time_ in the directory data_path.parent
    file_names = glob.glob(str(data_path.parent) + "/file_base_in_border_time_*.csv")

    for i in range(len(file_names)):

        # if not (data_path / f"training_data_{i:04d}.csv").exists():
        if True:

            # if configs["dataset"] == "geolife" or configs["dataset"] == "geolife_test":

            print(f"load raw data from {data_path.parent}/file_base_in_border_{i:04d}.csv")
            trajectories = pd.read_csv(data_path.parent / f"file_base_in_border_time_{i:04d}.csv", header=None).values[:100]
            
            print("make stay trajectory")
            time_trajectories, trajectories = make_stay_trajectory(trajectories, time_threshold, location_threshold)

            print("make complessed dataset")
            dataset, times = make_complessed_dataset(time_trajectories, trajectories, grid)
            save_path = data_path / f"training_data_{i:04d}.csv"
            print(f"save complessed dataset to {save_path}")
            save_state_with_nan_padding(save_path, dataset)
            
            time_save_path = data_path / f"training_data_time_{i:04d}.csv"
            save_time_with_nan_padding(time_save_path, times)

            training_data = pd.DataFrame(dataset).values
            
            # else:
            #     dataset = []
                
            #     locations, times = load(n_bins, lat_range, lon_range)
            #     for location, time in zip(locations, times):

            #         split_indices = split(time, seq_len, start_hour, end_hour)
            #         dataset.append(np.array(location)[split_indices])
                    
            #     training_data = pd.DataFrame(dataset).to_csv(data_path / "training_data.csv", header=None, index=None)

        else:
            print("training data exists")
            training_data = pd.read_csv(data_path / f"training_data_{i:04d}.csv", header=None).values
        
    
    if not (data_path/"gps.csv").exists():
        gps = make_gps(lat_range, lon_range, n_bins)
        gps.to_csv(data_path / f"gps.csv", header=None, index=None)
        print(gps)
    else:
        print("GPS exists")

    # if not (data_path/"M1.npy").exists():
    #     M1 = construct_M1(training_data, max_locs)
    #     np.save(data_path/f'M1.npy',M1)
    #     print(M1)
    # else:
    #     print("M1 exists")
        
    # if not (data_path/"M2.npy").exists():
    #     gps = pd.read_csv(data_path/"gps.csv", header=None)
    #     M2 = construct_M2(training_data, max_locs, gps)
    #     np.save(data_path/f'M2.npy',M2)
    #     print(M2)
    # else:
    #     print("M2 exists")


    training_data[np.isnan(training_data)] = max_locs
    training_data = training_data.astype(int)

    plot_hist2d(training_data.reshape(-1), n_bins, max_locs, data_path / f"pr_training.png")
    for hour in range(len(training_data[0])):
        plot_hist2d(training_data[:,hour].reshape(-1), n_bins, max_locs, data_path / f"pr_training_{hour}.png")