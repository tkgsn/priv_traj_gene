import argparse
import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns
import sys
import json
import scipy

from my_utils import get_datadir, get_gps, latlon_to_state
import pandas as pd
import matplotlib.pyplot as plt
from grid import Grid

def p_r(trajectories, vocab_size):
    data = trajectories.astype(int)
    data = data.reshape(-1)
    p_r = np.bincount(data, minlength=vocab_size)[:vocab_size]
    p_r = p_r / p_r.sum()
    return p_r

def construct_prob_from_r(target, trajectories, vocab_size, traj_length, threshold=100):
    trajectories = trajectories[:,:traj_length]
    ys, xs = np.where(trajectories == target)
    
    if len(xs) <= threshold:
        return None
    
    probs = np.zeros(vocab_size)
    for x, y in zip(xs, ys):
        if x == traj_length-1:
            continue
        if trajectories[y, x+1] < vocab_size:
            probs[trajectories[y, x+1]] += 1
        
    probs[target] = 0
    
    if probs.sum() == 0:
        return None
    
    probs = probs/probs.sum()
    return list(probs)

def js_divergence_p_r(data1, data2, vocab_size):
    data1_loc_dist = p_r(data1, vocab_size)
    data2_loc_dist = p_r(data2, vocab_size)
    return get_js_divergence(data1_loc_dist, data2_loc_dist)

def get_js_divergence(p1, p2):

    p1 = p1 / (np.sum(p1)+1e-14)
    p2 = p2 / (np.sum(p2)+1e-14)
    m = (p1 + p2) / 2
    js = 0.5 * scipy.stats.entropy(p1, m) + \
        0.5 * scipy.stats.entropy(p2, m)
    return js
    
def arr_to_distribution(arr, min, max, bins):
    if max - min != 0:
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, float(
                    max - min) / bins))
    else:
        distribution, base = np.histogram(
            arr, np.arange(
                min, max, bins))
    return distribution, base[:-1]

def get_distances_from_0(trajs, grid):
    distances = []
    for traj in trajs:
        distances.append(get_distance_from_0(traj, grid))
    distances = np.array(distances, dtype=float)
    return distances

def get_distance_from_0(traj, grid):
    distances = []
    for i in range(len(traj)):
        # if (traj[0] >= len(X)) or (traj[i] >= len(X)):
        #     distances.append(0)
        # else:
        #     dx = X[traj[0]] - X[traj[i]]
        #     dy = Y[traj[0]] - Y[traj[i]]
        #     distances.append(np.sqrt(dx**2 + dy**2))
        distances.append(get_distance(traj[0], traj[i], grid))
    return distances

def get_distance(state_0, state_1, grid):
    if (state_0 >= len(grid.grids)) or (state_1 >= len(grid.grids)):
        # print("FIND OUTSIDE LOCATION")
        return -1
    
    lat_0, lon_0 = grid.state_to_center_latlon(state_0)
    lat_1, lon_1 = grid.state_to_center_latlon(state_1)

    dx = lon_0 - lon_1
    dy = lat_0 - lat_1
    return np.sqrt(dx**2 + dy**2)

def get_distances(trajs, grid):
    distances = []
    for traj in trajs:
        for i in range(len(traj) - 1):
            # dx = X[traj[i]] - X[traj[i + 1]]
            # dy = Y[traj[i]] - Y[traj[i + 1]]
            # distances.append(np.sqrt(dx**2 + dy**2))
            distances.append(get_distance(traj[i], traj[i+1], grid))
    distances = np.array(distances, dtype=float)
    return distances

def get_durations(trajs):
    seq_len = len(trajs[0])
    d = []
    for traj in trajs:
        num = 1
        for i, lc in enumerate(traj[1:]):
            if lc == traj[i]:
                num += 1
            else:
                d.append(num)
                num = 1
    return np.array(d)/seq_len

def get_gradius(trajs, grid):
    gradius = []
    seq_len = len(trajs[0])
    for traj in trajs:
        xys = np.array([grid.state_to_center_latlon(t) for t in traj if t < len(grid.grids)])
        if len(xys) == 0:
            continue
        xs, ys = xys[:, 1], xys[:, 0]
        xcenter, ycenter = np.mean(xs), np.mean(ys)
        dxs, dys = xs - xcenter, ys - ycenter
        rad = [dxs[i]**2 + dys[i]**2 for i in range(len(xys))]
        rad = np.mean(np.array(rad, dtype=float))
        gradius.append(rad)
    gradius = np.array(gradius, dtype=float)
    return gradius

def plot_distance_from_0(training_data, syn_data, grid):  

    training_result = get_distances_from_0(training_data, grid)
    syn_result = get_distances_from_0(syn_data, grid)

    training_result[training_result == -1] = np.nan
    syn_result[syn_result == -1] = np.nan

    # compute mean of axis=0 without nan
    training_y = np.nanmean(training_result, axis=0)
    syn_y = np.nanmean(syn_result, axis=0)

    # training_y = training_result.mean(axis=0)
    # syn_y = syn_result.mean(axis=0)
    if len(syn_y) < len(training_y):
        syn_y = np.pad(syn_y, [0,len(training_y)-len(syn_y)], 'constant')
    else:
        syn_y = syn_y[:len(training_y)]
    x = range(0,len(training_y))
    
    # plt.plot(x, training_y, label="real")
    # plt.plot(x, syn_y, label="syn")
    # plt.legend()
    # plt.xlabel("hour")
    # plt.ylabel("distance")
    # plt.savefig(save_path)
    # plt.clf()
    
    return list(training_y), list(syn_y)


def plot_distance(training_data, syn_data, grid):
    
    seq_len = syn_data.shape[1]
    
    training_result = get_distances(training_data, grid)
    syn_result = get_distances(syn_data, grid)
    
    print(training_result)
    g1_dist, _ = arr_to_distribution(training_result, 0, grid.max_distance**2, 10000)
    g2_dist, _ = arr_to_distribution(syn_result, 0, grid.max_distance**2, 10000)
    
    g_jsd = get_js_divergence(g1_dist, g2_dist)
    
    return g_jsd

def plot_duration(training_data, syn_data):
    

    seq_len = syn_data.shape[1]
    training_result = get_durations(training_data)
    syn_result = get_durations(syn_data)
    
    du1_dist, _ = arr_to_distribution(training_result, 0, 1, seq_len)
    du2_dist, _ = arr_to_distribution(syn_result, 0, 1, seq_len)
    
    du_jsd = get_js_divergence(du1_dist, du2_dist)
    
    return du_jsd


def plot_gradius(training_data, syn_data, grid):
    
    seq_len = syn_data.shape[1]
    
    training_result = get_gradius(training_data, grid)
    syn_result = get_gradius(syn_data, grid)
    
    g1_dist, _ = arr_to_distribution(training_result, 0, grid.max_distance**2, 10000)
    g2_dist, _ = arr_to_distribution(syn_result, 0, grid.max_distance**2, 10000)
    
    g_jsd = get_js_divergence(g1_dist, g2_dist)
        
    return g_jsd
    
    
def compute_p_r_r(vocab_size, syn_data, training_data, threshold=100):
    results = {}
    traj_length = min([syn_data.shape[1], training_data.shape[1]])
    for j in range(vocab_size):
        syn_probs = construct_prob_from_r(j, syn_data, vocab_size, traj_length, threshold=threshold)
        if syn_probs is None:
            continue
        
        training_probs = construct_prob_from_r(j, training_data, vocab_size, traj_length, threshold=threshold)
        if training_probs is None:
            continue
            
        p_r_r = get_js_divergence(training_probs, syn_probs)
        
        results[j] = p_r_r
    
    return results

def make_hist_2d(counts, n_bins):
    hist2d = [[0 for i in range(n_bins+2)] for j in range(n_bins+2)]
    for state in range((n_bins+2)**2):
        x,y = state_to_xy(state, n_bins)
        hist2d[x][y] = counts[state]
    return np.array(hist2d)

def state_to_xy(state, n_bins):
    n_x = n_bins+2
    n_y = n_bins+2

    x = (state) % n_x
    y = int((state) / n_y)
    return x, y

def plot_hist2d(data, n_bins, vocab_size, save_path):
    counts = np.bincount(data, minlength=vocab_size)
    hist2d = make_hist_2d(counts, n_bins)

    ax = sns.heatmap(hist2d.T/hist2d.sum(), cmap=sns.color_palette("light:b", as_cmap=True), vmax=0.01)
    ax.invert_yaxis()
    plt.savefig(save_path)
    plt.clf()

def discretize(data, grid):
    state_trajectories = []
    n_locations = len(grid.grids)
    for trajectory in data:
        state_trajectory = []

        for latlon in trajectory:
            if type(latlon) is not str:
                break
            else:
                lat, lon = latlon.split(" ")
                lat = float(lat)
                lon = float(lon)
                
                # state_trajectory.append(grid.latlon_to_state(lat, lon))
                state_trajectory.append(latlon_to_state(lat, lon, grid.lat_range, grid.lon_range, grid.n_bins))
        
        # remove consecutive same states
        # if len(state_trajectory) = 0, then state_trajectory[0] will raise error
        if len(state_trajectory) > 0:
            state_trajectory = [state_trajectory[0]] + [state_trajectory[i] for i in range(1, len(state_trajectory)) if state_trajectory[i] != state_trajectory[i-1]]
        state_trajectories.append(state_trajectory)
    # add padding so that all trajectories have the same length
    max_seq_len = max([len(trajectory) for trajectory in state_trajectories])
    for state_trajectory in state_trajectories:
        state_trajectory += [n_locations] * (max_seq_len - len(state_trajectory))
    return np.array(state_trajectories)


def discretized_evaluation(discretized_original_data, discretized_syn_data, grid):

    print(discretized_syn_data)
    results = {}

    seq_len = min([discretized_original_data.shape[1], discretized_syn_data.shape[1]])

    results["p_r"] = js_divergence_p_r(discretized_original_data, discretized_syn_data, grid.vocab_size)
    print("p_r", results["p_r"])

    results["p_r_t"] = {}
    for hour in range(seq_len):
        results["p_r_t"][hour] = js_divergence_p_r(discretized_original_data[:,hour], discretized_syn_data[:,hour], grid.vocab_size)
        print("p_r_t", hour, results["p_r_t"][hour])
    
    results["p_r_r"] = compute_p_r_r(grid.vocab_size, discretized_syn_data, discretized_original_data)
    print("p_r_r", results["p_r_r"])

    results["durations"] = plot_duration(discretized_original_data, discretized_syn_data)
    results["gradius"] = plot_gradius(discretized_original_data, discretized_syn_data, grid)
    results["distance_from_0"] = plot_distance_from_0(discretized_original_data, discretized_syn_data, grid)
    results["distances"] = plot_distance(discretized_original_data, discretized_syn_data, grid)

    return results

def evaluation(original_data_path, syn_data_path, grid):
    print("load training data from", original_data_path)
    original_data = pd.read_csv(original_data_path, header=None)
    print("load syn data from", syn_data_path)
    syn_data = pd.read_csv(syn_data_path, header=None)    

    # discretize on the grid
    discretized_original_data = discretize(original_data.values, grid)
    discretized_syn_data = discretize(syn_data.values, grid)

    return discretized_evaluation(discretized_original_data, discretized_syn_data, grid)         
    # with open(save_path / f"results.json", "w") as f:
    #     json.dump(results, f)


    
def evaluation_(dataset_name, save_name, syn_data_name, name):
    results = {}
    save_path = (get_datadir() / "results" / dataset_name).parent / save_name
    data_path = get_datadir() / dataset_name 
    print("save to", save_path)
    print("load from", data_path)

    with open(data_path / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    vocab_size = (n_bins+2)**2
    

    training_data_path = data_path / "training_data.csv"
    training_data = pd.read_csv(training_data_path, header=None)
    training_data = training_data.fillna(vocab_size).astype(int)
    print("original", training_data_path)
    
    print(training_data)

    syn_data_path = save_path / syn_data_name
    syn_data = pd.read_csv(syn_data_path, header=None)
    print("syn", syn_data_path)
    print(syn_data)
    
    seq_len = min([syn_data.values.shape[1], training_data.values.shape[1]])
    print("seq_len", seq_len)
    
    plot_hist2d(training_data.values.reshape(-1), n_bins, vocab_size, save_path / f"{name}_pr_training.png")
    plot_hist2d(syn_data.values.reshape(-1), n_bins, vocab_size, save_path / f"{name}_pr_syn.png")

    results["distance_from_0"] = plot_distance_from_0(training_data, syn_data, dataset_name, vocab_size, save_path / f"{name}_distance_from_0.png")
    print("distance_from_0", results["distance_from_0"][0])
    print("distance_from_0", results["distance_from_0"][1])
    
    results["distances"] = plot_distance(training_data, syn_data, dataset_name, vocab_size, save_path / f"{name}_distance.png")
    
    results["durations"] = plot_duration(training_data, syn_data, dataset_name, vocab_size, save_path / f"{name}_distances.png")
    results["gradius"] = plot_duration(training_data, syn_data, dataset_name, vocab_size, save_path / f"{name}_gradius.png")
    
    results["p_r_t"] = {}
    
    for hour in range(seq_len):
        results["p_r_t"][hour] = js_divergence_p_r(training_data.loc[:,hour], syn_data.loc[:,hour], vocab_size)
        print("p_r_t", hour, results["p_r_t"][hour])
    
    results["p_r"] = js_divergence_p_r(training_data, syn_data, vocab_size)
    print("p_r", results["p_r"])
    
    results["p_r_r"] = compute_p_r_r(vocab_size, syn_data, training_data)
    print("p_r_r", results["p_r_r"])
                                     
    with open(save_path / f"{name}_results.json", "w") as f:
        json.dump(results, f)
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='test', type=str)
    parser.add_argument('--data_name', default='test_1', type=str)
    parser.add_argument('--save_name', default='test_1', type=str)
    parser.add_argument('--syn_data_name', default='gene_epoch_0.csv', type=str)
    args = parser.parse_args()

    dataset = args.dataset
    data_name = args.data_name
    syn_data_name = args.syn_data_name
    save_name = args.save_name

    grid = Grid()
    ranges = Grid.make_ranges_from_latlon_range_and_nbins(lat_range, lon_range, n_bins)
    grid.make_grid_from_ranges(ranges)

    evaluation(training_data_path, syn_data_path, grid)

    # evaluation(f"{dataset}/{data_name}", args.save_name, syn_data_name, syn_data_name)