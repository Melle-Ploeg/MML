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
    # Main loop creating the feature matrix. Yes the code is ugly.
    start = time.time()
    for i in range(length):
        result[i][1] = find_nearest_point(time_array[i], np.ravel(sample_dict['EDA']), np.ravel(time_dict['EDA']))
        result[i][2] = find_nearest_point(time_array[i], np.ravel(sample_dict['BVP']), np.ravel(time_dict['BVP']))
        result[i][3] = find_nearest_point(time_array[i], np.ravel(sample_dict['TEMP']), np.ravel(time_dict['TEMP']))
        result[i][4] = find_nearest_point(time_array[i], sample_dict['ACC'][:,0], time_dict['ACC'])
        result[i][5] = find_nearest_point(time_array[i], sample_dict['ACC'][:,1], time_dict['ACC'])
        result[i][6] = find_nearest_point(time_array[i], sample_dict['ACC'][:,2], time_dict['ACC'])
    print(time.time() - start)
    return result

# Find the value of sample data b that is nearest to the time of sample a
def find_nearest_point(a_time, b_sample, b_times):
    idx = (np.abs(b_times - a_time)).argmin()
    return b_sample[idx]

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
features = align_features(signal_data['STRESS']['f01'], time_data['STRESS']['f01'])
print(features.shape)
print(features)


