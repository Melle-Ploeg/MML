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
    result = np.zeros((length, len(features) + 2))      # Create empty feature matrix, with one column for each feature
    result[:,0:1] = sample_dict['HR']
    sample_dict['ACCB'] = (abs(sample_dict['ACC'][:,0]) + abs(sample_dict['ACC'][:,1]) + abs(sample_dict['ACC'][:,2])) / 3
    time_dict['ACCB'] = time_dict['ACC']
    # Main loop creating the feature matrix. Yes the code is ugly.
    start = time.time()
    for i in range(length):
        result[i][1] = find_nearest_point(time_array[i], np.ravel(sample_dict['EDA']), np.ravel(time_dict['EDA']))
        result[i][2] = find_nearest_point(time_array[i], np.ravel(sample_dict['BVP']), np.ravel(time_dict['BVP']))
        result[i][3] = find_nearest_point(time_array[i], np.ravel(sample_dict['TEMP']), np.ravel(time_dict['TEMP']))

        result[i][4] = find_avg(time_array[i], np.ravel(sample_dict['ACCB']), np.ravel(time_dict['ACCB']), 64)
        result[i][5] = find_std(time_array[i], np.ravel(sample_dict['ACCB']), np.ravel(time_dict['ACCB']), 64)

    print(time.time() - start)
    # print(result[:,4])
    # print(result[:,4].shape)
    # print(result[:,5])
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


def label_aerobic(aligned_data, sample_dict):
    result = {}
    bad_keys = ['S03', 'S07', 'S11_a', 'S11_b', 'S12']
    for subject in aligned_data.keys():
        if not subject in bad_keys:
            labels = np.zeros(len(aligned_data[subject][:,0]))
            tags = sample_dict[subject]['tags']
            labels[tags[1]:tags[-2]] = 1
            result[subject] = labels
    return result


def label_anaerobic(aligned_data, sample_dict):
    result = {}
    bad_keys = ['S06', 'S16_a', 'S16_b']
    for subject in aligned_data.keys():
        if not subject in bad_keys:
            print(subject)
            labels = np.zeros(len(aligned_data[subject][:, 0]))
            tags = sample_dict[subject]['tags']
            if 'S' in subject:
                # Three tests
                for i in range(0, 6, 2):
                    start, end = tags[i], tags[i + 1]
                    labels[start:end] = 1
                    result[subject] = labels
            else:
                # Four tests
                for i in range(2, 10, 2):
                    start, end = tags[i], tags[i + 1]
                    labels[start:end] = 1
                    result[subject] = labels
    return result

def align_all(sample_dict, time_dict):
    result = {}
    for subject in sample_dict.keys():
        result[subject] = align_features(sample_dict[subject], time_dict[subject])
    return result


signal_data, time_data, fs_dict, participants = get_features()
print(signal_data['ANAEROBIC']['f01']['tags'])
fs_dict = fs_dict['ANAEROBIC']['f01'] # Sampling frequencies are the same for every experiment so we simplify the dict
features = align_features(signal_data['ANAEROBIC']['f01'], time_data['ANAEROBIC']['f01'])
# print(features.shape)
# print(features)
np.set_printoptions(threshold=np.inf)
test_dict = {'f01':features}
print(label_anaerobic(test_dict, signal_data['ANAEROBIC'])['f01'])

for s in signal_data['ANAEROBIC'].keys():
    print(s)
    print(len(signal_data['ANAEROBIC'][s]['tags']))


