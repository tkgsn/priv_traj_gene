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

from my_utils import get_datadir, get_gps, get_maxdistance
import pandas as pd
import matplotlib.pyplot as plt


def p_r(trajectories, vocab_size):
    data = trajectories.astype(int)
    data = data.values.reshape(-1)
    p_r = np.bincount(data, minlength=vocab_size)
    p_r = p_r / p_r.sum()
    return p_r

def construct_prob_from_r(target, trajectories, vocab_size, traj_length, threshold=100):
    trajectories = trajectories.loc[:,:traj_length-1]
    ys, xs = np.where((trajectories == target).values)
    
    if len(xs) <= threshold:
        return None
    
    probs = np.zeros(vocab_size)
    for x, y in zip(xs, ys):
        if x == traj_length-1:
            continue
        probs[trajectories.loc[y, x+1]] += 1
        
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

    p1 = p1 / (p1.sum()+1e-14)
    p2 = p2 / (p2.sum()+1e-14)
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

def get_distances_from_0(trajs, dataset_name):
    distances = []
    for traj in trajs:
        distances.append(get_distance_from_0(traj, dataset_name))
    distances = np.array(distances, dtype=float)
    return distances

def get_distance_from_0(traj, dataset_name):
    X, Y = get_gps(dataset_name)
    seq_len = len(traj)
    distances = []
    for i in range(seq_len):
        dx = X[traj[0]] - X[traj[i]]
        dy = Y[traj[0]] - Y[traj[i]]
        distances.append(np.sqrt(dx**2 + dy**2))
    return distances
    
def get_distances(trajs, dataset_name):
    X, Y = get_gps(dataset_name)
    distances = []
    for traj in trajs:
        for i in range(len(traj) - 1):
            dx = X[traj[i]] - X[traj[i + 1]]
            dy = Y[traj[i]] - Y[traj[i + 1]]
            distances.append(np.sqrt(dx**2 + dy**2))
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

def get_gradius(trajs):
    gradius = []
    seq_len = len(trajs[0])
    for traj in trajs:
        xs = np.array([self.X[t] for t in traj])
        ys = np.array([self.Y[t] for t in traj])
        xcenter, ycenter = np.mean(xs), np.mean(ys)
        dxs = xs - xcenter
        dys = ys - ycenter
        rad = [dxs[i]**2 + dys[i]**2 for i in range(seq_len)]
        rad = np.mean(np.array(rad, dtype=float))
        gradius.append(rad)
    gradius = np.array(gradius, dtype=float)
    return gradius

def plot_distance_from_0(training_data, syn_data, dataset_name, vocab_size, save_path):  

    training_result = get_distances_from_0(training_data.values, dataset_name)
    syn_result = get_distances_from_0(syn_data.values, dataset_name)
    
    training_y = training_result.mean(axis=0)
    syn_y = syn_result.mean(axis=0)
    x = range(0,len(training_y))
    
    plt.plot(x, training_y, label="real")
    plt.plot(x, syn_y, label="syn")
    plt.legend()
    plt.xlabel("hour")
    plt.ylabel("distance")
    plt.savefig(save_path)
    plt.clf()
    
    return list(training_y), list(syn_y)


def plot_distance(training_data, syn_data, dataset_name, vocab_size, save_path):
    
    seq_len = syn_data.values.shape[1]
    
    training_result = get_distances(training_data.values, dataset_name)
    syn_result = get_distances(syn_data.values, dataset_name)
    
    print(training_result)
    g1_dist, _ = arr_to_distribution(training_result, 0, get_maxdistance(dataset_name)**2, 10000)
    g2_dist, _ = arr_to_distribution(syn_result, 0, get_maxdistance(dataset_name)**2, 10000)
    
    g_jsd = get_js_divergence(g1_dist, g2_dist)
    
    return g_jsd

def plot_duration(training_data, syn_data, dataset_name, vocab_size, save_path):
    

    seq_len = syn_data.values.shape[1]
    training_result = get_durations(training_data.values)
    syn_result = get_durations(syn_data.values)
    
    du1_dist, _ = arr_to_distribution(training_result, 0, 1, seq_len)
    du2_dist, _ = arr_to_distribution(syn_result, 0, 1, seq_len)
    
    du_jsd = get_js_divergence(du1_dist, du2_dist)
    
    return du_jsd


def plot_gradius(training_data, syn_data, dataset_name, vocab_size, save_path):
    
    seq_len = syn_data.values.shape[1]
    
    individualEval = IndividualEval(dataset_name, data_name, seq_len, vocab_size)
    training_result = get_gradius(training_data.values)
    syn_result = get_gradius(syn_data.values)
    
    g1_dist, _ = arr_to_distribution(training_result, 0, get_maxdistance(dataset_name)**2, 10000)
    g2_dist, _ = arr_to_distribution(syn_result, 0, get_maxdistance(dataset_name)**2, 10000)
    
    g_jsd = get_js_divergence(g1_dist, g2_dist)
        
    return g_jsd
    
    
def compute_p_r_r(vocab_size, syn_data, training_data):
    results = {}
    traj_length = syn_data.values.shape[1]
    for j in range(vocab_size):
        syn_probs = construct_prob_from_r(j, syn_data, vocab_size, traj_length, threshold=100)
        if syn_probs is None:
            continue
        
        training_probs = construct_prob_from_r(j, training_data, vocab_size, traj_length, threshold=100)
        if training_probs is None:
            continue
            
        p_r_r = compute_JS_divergence(training_probs, syn_probs)
        
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
    counts = np.bincount(data.values.reshape(-1), minlength=vocab_size)
    hist2d = make_hist_2d(counts, n_bins)

    ax = sns.heatmap(hist2d.T, cmap=sns.color_palette("light:b", as_cmap=True), vmax=30)
    ax.invert_yaxis()
    plt.savefig(save_path)
    plt.clf()

    
def evaluation(dataset_name, save_name, syn_data_name, name):
    results = {}
    save_path = (get_datadir() / "results" / dataset_name).parent / save_name
    data_path = get_datadir() / dataset_name 
    print("save to", save_path)

    with open(data_path / "params.json", "r") as f:
        param = json.load(f)
    n_bins = param["n_bins"]
    vocab_size = (n_bins+2)**2
    

    training_data_path = data_path / "training_data.csv"
    training_data = pd.read_csv(training_data_path, header=None)
    print("original", training_data_path)

    syn_data_path = save_path / syn_data_name
    syn_data = pd.read_csv(syn_data_path, header=None)
    print("syn", syn_data_path)
    
    seq_len = syn_data.values.shape[1]
    
    plot_hist2d(training_data, n_bins, vocab_size, save_path / f"{name}_pr_training.png")
    plot_hist2d(syn_data, n_bins, vocab_size, save_path / f"{name}_pr_syn.png")

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

    dataset_name = "peopleflow"
    data_name = "peopleflow_dnum2000"
    syn_data_name = "gene_epoch_0.csv"
    save_name = "0"
    save_name = data_name
    
    evaluation(dataset_name, data_name, save_name, syn_data_name, save_name)