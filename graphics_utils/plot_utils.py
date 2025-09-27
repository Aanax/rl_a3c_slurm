import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import datetime
import os
import shutil

def get_log_df(logname):
    """
    get df from logfile
    """
    log_file = open(logname)
    lines = log_file.readlines()
    #print("Got ",len(lines)," lines total")
    log_file.close()
    
    #train_start_line = 19 + 2
    for l in range(len(lines)):
        if "Time" in lines[l]:
            train_start_line = l
            break
    else:
        train_start_line = len(lines)
    
    columns = ['Date','Server_time','runtime_h','runtime_m','runtime_s','episode_reward','episode_length','reward_mean','frames_total']
    df = pd.DataFrame(columns = columns)
    #print(df)
    for line_num in range(train_start_line,len(lines)):
        if len(lines[line_num].strip()):
            vals = np.array(lines[line_num].replace(',','').strip().split(' '))
            # print("---")
            # print(vals)
            df =df.append(pd.DataFrame([vals[[0,1,4,5,6,9,12,15,17]]], columns=columns))
        
    df['episode_reward'] = df['episode_reward'].astype(float)
    df['episode_length'] = df['episode_length'].astype(int)
    df['reward_mean'] = df['reward_mean'].astype(float)
    df['frames_total'] = df['frames_total'].astype(int)
    df['server_time'] = df['Date']+' '+df['Server_time'].str.slice(start=0, stop=-3)
    df['server_time'] = pd.to_datetime(df['server_time'])
    df.drop(['Date','Server_time'],axis=1,inplace=True)
    df['total_run_time'] = df['runtime_h'].str.slice(stop=-1)+':'+df['runtime_m'].str.slice(stop=-1)+':'+df['runtime_s'].str.slice(stop=-1)
    df['total_run_time'] = pd.to_timedelta(df['total_run_time'])
    df.drop(['runtime_h','runtime_m','runtime_s'],axis=1,inplace=True)
    df["reward_mean_10"] = df["episode_reward"].rolling(10).mean()
    
    return df


def get_log_len(logname):
    log_file = open(logname)
    lenfile = sum(1 for line in log_file)
    #print("Got ",len(lines)," lines total")
    log_file.close()
    return lenfile
def get_log_offset(logname):
    """
    how many lines to skip from start. containing exp info.
    can vary in different veersions
    """
    log_file = open(logname)
    i=0
    line = ''
    while not "Time" in line:
        line = log_file.readline()
        i+=1
    log_file.close()
    return i


def get_mean_stats_df(logs_folder):
#     logs_folder = "./logs/trunc/"
    log_dfs = []
    min_len = 10000000
    for logname in os.listdir(logs_folder):
        if not logname.startswith("."):
            log_len = get_log_len(logs_folder+logname)
            print(log_len)
            if log_len<min_len:
                min_len = log_len

    offset = get_log_offset(logs_folder+logname)
    min_len -= offset
    dfs = []
    for logname in os.listdir(logs_folder):
        if not logname.startswith("."):
            log_df = get_log_df(logs_folder+logname)[:min_len].drop(["server_time"],axis=1)
            print(len(log_df))
            dfs.append(log_df)
            log_dfs.append(log_df.values)
    columns = log_df.columns
    arr = np.array(log_dfs)
    new_values = arr.mean(axis = 0)
    df = pd.DataFrame(new_values, columns = columns)
    return df, dfs


# from scipy.interpolate import interp1d
# def get_mean_interpolated_stats_df(logs_folder, target_value = "reward_mean_10", num_points=500, default_r=-21):
# #     logs_folder = "./logs/trunc/"
#     interp_funcs = []
#     min_len = 10000000
#     for logname in os.listdir(logs_folder):
#         if not logname.startswith("."):
#             log_len = get_log_len(logs_folder+logname)
#             print("got log of len", log_len)
#             if log_len<min_len:
#                 min_len = log_len
                
#     print("Minimal log len",min_len)

#     offset = get_log_offset(logs_folder+logname)
#     print("And offset is ", offset)
#     min_len -= offset
#     print("So final min_len is", min_len)
#     dfs = []
#     max_num_frames = 0
#     min_num_frames = np.inf
#     for logname in os.listdir(logs_folder):
#         if not logname.startswith("."):
#             log_df = get_log_df(logs_folder+logname)[:min_len].drop(["server_time"],axis=1)
#             print("got df of len(log_df)= ", len(log_df))
#             dfs.append(log_df)
#             list_frames = list(log_df["frames_total"])
#             last_num_frames = list_frames[-1]
#             first_num_frames = list_frames[0]
#             if last_num_frames>max_num_frames:
#                 max_num_frames = last_num_frames
#             if first_num_frames<min_num_frames:
#                 min_num_frames = first_num_frames
#             interp_funcs.append(interp1d(list_frames, list(log_df[target_value].fillna(default_r)),fill_value="extrapolate"))
#     #print(interp_funcs[0]._check_bounds(0))
#     print(min_num_frames," to ", max_num_frames)
#     new_x = np.linspace(min_num_frames, max_num_frames, num_points)
#     new_y = []
#     for x in new_x:
#         new_y.append(np.mean([f(x) for f in interp_funcs]))
        
#     return new_x,new_y, dfs, list_frames

def draw_mean_and_runs(mean, runs, n_tests = 500, col = "red", alpha=0.25, ax = None, label="unknown"):
    #plt.figure(figsize=(20,20))
    n_frames = n_tests
    if ax is None:
        ax = mean[:n_frames].plot(x="frames_total", y="reward_mean_10", figsize=(20,10), lw=4,legend=False,color=col, fontsize=20, label=label)
    else:
        mean[:n_frames].plot(x="frames_total", y="reward_mean_10", ax=ax, figsize=(20,10), lw=4,legend=False,color=col, fontsize=20, label=label)
    for run in runs:
        run[:n_frames].plot(x="frames_total", y="reward_mean_10", ax=ax, lw=4, legend=False, alpha=alpha,color=col, fontsize=20, label='_nolegend_')

    #plt.legend(labels = ["base","star","starTA","Starv1"],loc="lower_right", prop={'size': 26})
    plt.xlabel('frames_total', fontsize=22)
    plt.ylabel('mean_reward_10', fontsize=22)
    return ax


def remove_short_logs(logs_folder, min_len=50):
    listdir = [i for i in os.listdir(logs_folder) if (not os.path.isdir(logs_folder+i))]
    
    for logname in listdir:
        if not logname.startswith("."):
            log_len = get_log_len(logs_folder+logname)
            print(log_len)
            if log_len<min_len:
                print("removing ",logs_folder+logname)
                os.remove(logs_folder+logname)

from scipy.interpolate import interp1d
def get_mean_interpolated_stats_df(logs_folder, target_value = "reward_mean_10", num_points=500, default_r=-21, verbose=2,
                                  min_frames_in_log_to_use_it=900000):
    """
    list all fines in logs_dir
    get length of minimal log (in lines!)
    get offset (in log startinig lines there are hyperparams description)
    get df for each logfile
    interpolate continous line MEAN10REWARD(num_frames) from min_start_frames to max_end_frames
    get mean from all runs' interpolated funcions
    """
    
#     logs_folder = "./logs/trunc/"
    interp_funcs = []
    min_len = 10000000
    if verbose==2:
        print("verbose ", verbose)
    
    listdir = [i for i in os.listdir(logs_folder) if (not os.path.isdir(logs_folder+i))]
    
    for logname in listdir:
        if logname.endswith(".log"):
            log_len = get_log_len(logs_folder+logname)
            if verbose==2:
                print("log_len ",log_len)
            if log_len<min_len:
                min_len = log_len
    if verbose==2:
        print("Minimal log len",min_len)
    offset = get_log_offset(logs_folder+logname)
    min_len -= offset
    if verbose==2:
        print("So final min_len is", min_len)
    dfs = []
    max_num_frames = np.inf
    min_num_frames = np.inf
    for logname in listdir:
        if logname.endswith(".log") or logname.endswith("log"):
            log_df = get_log_df(logs_folder+logname).drop(["server_time"],axis=1) #[:min_len]
            if verbose==2:
                print("Got log df len ",len(log_df))
            
            list_frames = list(log_df["frames_total"])
            last_num_frames = list_frames[-1]
            
            if last_num_frames >=min_frames_in_log_to_use_it:
                dfs.append(log_df)
            else:
                continue
            if verbose==2:
                print("Got last framenum ", last_num_frames)
            first_num_frames = list_frames[0]
            if verbose==2:
                print("Got first framenum ", first_num_frames)
            if last_num_frames<max_num_frames:
                max_num_frames = last_num_frames
            if first_num_frames<min_num_frames:
                min_num_frames = first_num_frames
            interp_funcs.append(interp1d(list_frames, list(log_df[target_value].fillna(default_r)),fill_value="extrapolate"))
    #print(interp_funcs[0]._check_bounds(0))
    
#     #get original stds ### THEY DONT MATCH BY X! ASYNC. DIFFERENT TIMES
#     stds = np.std(np.array([df[target_value].fillna(default_r) for df in dfs]), axis=0)
#     print("Len orig stds ", len(stds))
#     #build interpolated stds
    
    
    
    
    if verbose>0:
        print("X from ", min_num_frames," to ", max_num_frames)
    new_x = np.linspace(min_num_frames, max_num_frames, num_points)
    new_y = []
    for x in new_x:
        new_y.append(np.mean([f(x) for f in interp_funcs]))
        
    runs = []
    for f in interp_funcs:
        runs.append([f(x) for x in new_x])
    
    return new_x, new_y, runs, interp_funcs


def draw_mean_with_std(mean, runs, new_x, n_tests = 500, col = "red", alpha=0.25, ax = None, label="unknown", rus=False, linestyle='solid'):
    """
    draw mean with std zone
    """
    
    n_frames = n_tests
    if ax is None:
        plt.figure(figsize=(20,10))
        ax=plt.gca()
        ax.plot(new_x, mean, lw=3, color=col, label=label, linestyle=linestyle)
    else:
        ax.plot(new_x,mean, lw=3, color=col,label=label, linestyle=linestyle)
        alpha=0.2
#     stds=[]
#     for dotnum in range(len(new_x)):
#         stds.append(np.std([run[dotmun] for run in runs]))
    
    stds = np.std(runs, axis=0)
    
    #plot stds
    ax.fill_between(new_x, mean-stds, mean+stds, color=col, alpha=alpha)
    
    if rus:
        plt.xlabel('Общее число кадров', fontsize=24)
        plt.ylabel('Средняя награда за последние 10 тестов', fontsize=24)
    else:
        plt.xlabel('frames_total', fontsize=24)
        plt.ylabel('mean_reward_10', fontsize=24)
        
    
    return ax



def draw_meanIntep_and_runs(mean, runs, new_x, n_tests = 500, col = "red", alpha=0.25, ax = None, label="unknown", linestyle='solid'):
    """
    Main draw function.
    create new axis if not already
    plot mean at new_x with bold color
    and runs at new_x with non zero opacity
    """
    
    #plt.figure(figsize=(20,20))
    n_frames = n_tests
    if ax is None:
        plt.figure(figsize=(200,100))
        ax=plt.gca()
        ax.plot(new_x, mean, lw=30, color=col, label=label, linestyle=linestyle)#, ax=ax)
        #mean[:n_frames].plot(x="frames_total", y="reward_mean_10", figsize=(20,10), lw=4,legend=False,color=col, fontsize=20, label=label)
    else:
        ax.plot(new_x,mean, lw=30, color=col,label=label, linestyle=linestyle)#,ax=ax)
        #mean[:n_frames].p, lot(x="frames_total", y="reward_mean_10", ax=ax, figsize=(20,10), lw=4,legend=False,color=col, fontsize=20, label=label)
    for run in runs:
        ax.plot(new_x, run, lw=30,
                alpha=alpha,color=col, label='_nolegend_')
        #run[:n_frames].plot(x="frames_total", y="reward_mean_10", ax=ax, lw=4, legend=False, alpha=alpha,color=col, fontsize=20, label='_nolegend_')

    #plt.legend(labels = ["base","star","starTA","Starv1"],loc="lower_right", prop={'size': 26})
    plt.xlabel('frames_total', fontsize=22)
    plt.ylabel('mean_reward_10', fontsize=22)
    return ax
