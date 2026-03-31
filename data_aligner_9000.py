import torch
import test
import os
import numpy as np
import time

# Take the data and time dictionaries for a single sample or experiment, and resample to align with the lowest sampled data (HR)
def align_features(sample_dict:dict, time_dict:dict):
    features = ['EDA', 'BVP', 'TEMP', 'ACC']  # Features to align, without HR
    time_array = time_dict['HR']
    length = len(time_array)    # Scale everything to the length of the HR array
    result = np.zeros((length, len(features) + 3))      # Create empty feature matrix, with one column for each feature
    result[:,0:1] = sample_dict['HR']
    sample_dict['ACCB'] = (abs(sample_dict['ACC'][:,0]) + abs(sample_dict['ACC'][:,1]) + abs(sample_dict['ACC'][:,2])) / 3
    time_dict['ACCB'] = time_dict['ACC']
    # Main loop creating the feature matrix. Yes the code is ugly.
    start = time.time()
    label = 2
    print(sample_dict["tags"])
    for i in range(length):
        result[i][1] = find_avg(time_array[i], np.ravel(sample_dict['EDA']), np.ravel(time_dict['EDA']), 4)
        result[i][2] = find_std(time_array[i], np.ravel(sample_dict['EDA']), np.ravel(time_dict['EDA']), 4)
        result[i][3] = find_nearest_point(time_array[i], np.ravel(sample_dict['BVP']), np.ravel(time_dict['BVP']))
        result[i][4] = find_avg(time_array[i], np.ravel(sample_dict['TEMP']), np.ravel(time_dict['TEMP']), 4)

        result[i][5] = find_avg(time_array[i], np.ravel(sample_dict['ACCB']), np.ravel(time_dict['ACCB']), 64)
        result[i][6] = find_std(time_array[i], np.ravel(sample_dict['ACCB']), np.ravel(time_dict['ACCB']), 64)
        

    # print(time.time() - start)
    # print(result[:,4])
    # print(result[:,4].shape)
    # print(result[:,5])
    return result

def stress_F_Labels(sample_dict:dict, time_dict:dict):
    length = len(time_dict["HR"])
    result = np.zeros(length)
    set_number = 0
    next_tag = 2
    #label = 0
    sample_dict["tags"] += [0]
    for i in range(length):
        if i == sample_dict["tags"][next_tag]:
            set_number = (next_tag) * ((next_tag+1) % 2) #Multiplied by 0 if inbetween two sets
            next_tag += 1
        result[i] = int(set_number/2)
    return result

def stress_S_Labels(sample_dict:dict, time_dict:dict):
    length = len(time_dict["HR"])
    result = np.zeros(length)
    set_number = 0
    next_tag = 3
    #label = 0
    sample_dict["tags"][13] = 0
    for i in range(length):
        if i == sample_dict["tags"][next_tag]:
            set_number = (next_tag-2) * ((next_tag) % 2)
            next_tag += 1
        result[i] = int((set_number+1)/2)#int(set_number/2)
    return result

# Find the value of sample data b that is nearest to the time of sample a
def find_nearest_point(a_time, b_sample, b_times):
    idx = (np.abs(b_times - a_time)).argmin()
    return b_sample[idx]

def find_avg(a_time, b_sample, b_times, b_fs):
    idx = (np.abs(b_times - a_time)).argmin()
    return np.average(b_sample[idx:idx+b_fs])

def find_std(a_time, b_sample, b_times, b_fs):
    idx = (np.abs(b_times - a_time)).argmin()
    return np.std(b_sample[idx:idx+b_fs])

def get_features():
    dataset_path='data/Wearable_Dataset/'#replace the folder path

    strees_level_v1_path='data/Stress_Level_v1.csv'#replace the file path
    strees_level_v2_path='data/Stress_Level_v2.csv'#replace the file path
    states = os.listdir(dataset_path) #['AEROBIC', 'ANAEROBIC', 'STRESS']

    signal_data={}
    time_data={}
    fs_dict={}
    participants={}

    for state in states:
        folder_path = f'{dataset_path}/{state}'
        participants[state] = os.listdir(folder_path)
        signal_data[state], time_data[state], fs_dict[state] = test.read_signals(folder_path)

    return signal_data, time_data, fs_dict, participants

signal_data, time_data, fs_dict, participants = get_features()
fs_dict = fs_dict['AEROBIC']['f01'] # Sampling frequencies are the same for every experiment so we simplify the dict
features = align_features(signal_data['STRESS']['S01'], time_data['STRESS']['S01'])
labels = stress_S_Labels(signal_data['STRESS']['S01'], time_data['STRESS']['S01'])
with np.printoptions(edgeitems=4000):
    print(labels)

# print(features.shape)
# print(features)


