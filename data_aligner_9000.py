#import torch
import test
import os
import numpy as np
import time

# Take the data and time dictionaries for a single sample or experiment, and resample to align with the lowest sampled data (HR)
def align_features(sample_dict:dict, time_dict:dict):
    features = ['HR', 'EDA', 'EDA_std', 'BVP', 'TEMP', 'ACC']#, 'ACC_std']  # Features to align
    time_array = time_dict['HR']
    length = len(time_array)    # Scale everything to the length of the HR array
    result = np.zeros((length, len(features)))      # Create empty feature matrix, with one column for each feature
    result[:,0:1] = sample_dict['HR']
    sample_dict['ACCB'] = (abs(sample_dict['ACC'][:,0]) + abs(sample_dict['ACC'][:,1]) + abs(sample_dict['ACC'][:,2])) / 3
    time_dict['ACCB'] = time_dict['ACC']
    # Main loop creating the feature matrix. Yes the code is ugly.
    start = time.time()
    for i in range(length):
        result[i][1] = find_avg(time_array[i], np.ravel(sample_dict['EDA']), np.ravel(time_dict['EDA']), 4)
        result[i][2] = find_std(time_array[i], np.ravel(sample_dict['EDA']), np.ravel(time_dict['EDA']), 4)
        result[i][3] = find_nearest_point(time_array[i], np.ravel(sample_dict['BVP']), np.ravel(time_dict['BVP']))
        result[i][4] = find_avg(time_array[i], np.ravel(sample_dict['TEMP']), np.ravel(time_dict['TEMP']), 4)

        result[i][5] = find_avg(time_array[i], np.ravel(sample_dict['ACCB']), np.ravel(time_dict['ACCB']), 64)
        #result[i][6] = find_std(time_array[i], np.ravel(sample_dict['ACCB']), np.ravel(time_dict['ACCB']), 64)


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

# def stress_S_Labels(sample_dict:dict, time_dict:dict):
#     length = len(time_dict["HR"])
#     result = np.zeros(length)
#     set_number = 0
#     next_tag = 3
#     #label = 0
#     sample_dict["tags"][13] = 0
#     for i in range(length):
#         if i == sample_dict["tags"][next_tag]:
#             set_number = (next_tag-2) * ((next_tag) % 2)
#             next_tag += 1
#         result[i] = int((set_number+1)/2)#int(set_number/2)
#     return result


def label_stress(sample_dict:dict, time_dict:dict, bad_keys=['f07', "f14_a", "f14_b"]):
    result = {}
    for subject in sample_dict.keys():
        if subject not in bad_keys:
            current_dict = sample_dict[subject]
            length = len(time_dict[subject]["HR"])
            new_array = np.zeros(length)
            set_number = 0
            if 'S' in subject:
                next_tag = 3
                #label = 0
                #current_dict["tags"][13] = 0
                for i in range(length):
                    if i == current_dict["tags"][next_tag]:
                        #set_number = (next_tag-2) * ((next_tag) % 2)
                        set_number = (next_tag) % 2 #1 if during test, otherwise 0
                        if i != current_dict["tags"][13]: next_tag += 1
                    new_array[i] = int(set_number)#int((set_number+1)/2)#int(set_number/2)
            else:
                next_tag = 2
                #label = 0
                current_dict["tags"] += [0]
                for i in range(length):
                    if i == current_dict["tags"][next_tag]:
                        # set_number = (next_tag) * ((next_tag+1) % 2) #Multiplied by 0 if inbetween two sets
                        set_number = (next_tag+1) % 2 #1 if during test, otherwise 0
                        next_tag += 1
                    new_array[i] = int(set_number)#int(set_number/2)
            result[subject] = new_array
    return result

def label_aerobic(sample_dict:dict, time_dict:dict, bad_keys=['S03', 'S07', 'S11_a', 'S11_b', 'S12']):
    result = {}
    for subject in sample_dict.keys():
        if not subject in bad_keys:
            #print(aligned_data[subject])
            labels = np.zeros(len(time_dict[subject]["HR"]))
            tags = sample_dict[subject]['tags']
            labels[tags[1]:tags[-1]] = 2
            result[subject] = labels
    return result


def label_anaerobic(sample_dict, time_dict, bad_keys=['S06', 'S16_a', 'S16_b']):
    result = {}
    for subject in sample_dict.keys():
        if not subject in bad_keys:
            labels = np.zeros(len(time_dict[subject]["HR"]))
            tags = sample_dict[subject]['tags']
            if 'S' in subject:
                # Three tests
                for i in range(1, 6, 2):
                    start, end = tags[i], tags[i + 1]
                    labels[start:end] = 3
                    # Label the periods in between and after aerobic bursts as anaerobic, as there is slow easy peddling
                    start, end = tags[i + 1], tags[i + 2]
                    labels[start:end] = 2
                    result[subject] = labels

            else:
                # Four tests
                start, end = tags[1], tags[2]
                labels[start:end] = 2
                for i in range(2, 10, 2):
                    start, end = tags[i], tags[i + 1]
                    labels[start:end] = 3
                    # Label the periods in between and after aerobic bursts as anaerobic, as there is slow easy peddling
                    start, end = tags[i + 1], tags[i + 2]
                    labels[start:end] = 2
                    result[subject] = labels
    return result

# def label_aerobic(aligned_data, sample_dict, bad_keys=['S03', 'S07', 'S11_a', 'S11_b', 'S12']):
#     result = {}
#     for subject in aligned_data.keys():
#         if not subject in bad_keys:
#             #print(aligned_data[subject])
#             labels = np.zeros(len(aligned_data[subject][:,0]))
#             tags = sample_dict[subject]['tags']
#             labels[tags[1]:tags[-2]] = 2
#             result[subject] = labels
#     return result


# def label_anaerobic(aligned_data, sample_dict, bad_keys=['S06', 'S16_a', 'S16_b']):
#     result = {}
#     for subject in aligned_data.keys():
#         if not subject in bad_keys:
#             labels = np.zeros(len(aligned_data[subject][:, 0]))
#             tags = sample_dict[subject]['tags']
#             if 'S' in subject:
#                 # Three tests
#                 for i in range(0, 6, 2):
#                     start, end = tags[i], tags[i + 1]
#                     labels[start:end] = 3
#                     # Label the periods in between and after aerobic bursts as anaerobic, as there is slow easy peddling
#                     start, end = tags[i + 1], tags[i + 2]
#                     labels[start:end] = 2
#                     result[subject] = labels

#             else:
#                 # Four tests
#                 for i in range(2, 10, 2):
#                     start, end = tags[i], tags[i + 1]
#                     labels[start:end] = 3
#                     # Label the periods in between and after aerobic bursts as anaerobic, as there is slow easy peddling
#                     start, end = tags[i + 1], tags[i + 2]
#                     labels[start:end] = 2
#                     result[subject] = labels
#     return result

def align_all(sample_dict, time_dict):
    result = {}
    for subject in sample_dict.keys():
        result[subject] = align_features(sample_dict[subject], time_dict[subject])
    return result


# signal_data, time_data, fs_dict, participants = get_features()
# fs_dict = fs_dict['AEROBIC']['f01'] # Sampling frequencies are the same for every experiment so we simplify the dict
#
# print(signal_data['AEROBIC']['f02']['tags'])
# print(len(signal_data['AEROBIC']['f02']['HR']))
# print(signal_data['AEROBIC']['f07']['tags'])
# print(len(signal_data['AEROBIC']['f07']['HR']))
# print(signal_data['AEROBIC']['f11']['tags'])
# print(len(signal_data['AEROBIC']['f11']['HR']))
# print(signal_data['AEROBIC']['f03']['tags'])
# print(len(signal_data['AEROBIC']['f03']['HR']))
#TODO TEMPRARY CODE VERY TEMPORATY

# subject = "S01"

# features = align_features(signal_data['STRESS'][subject], time_data['STRESS'][subject])
# labels = label_stress(signal_data['STRESS'], time_data['STRESS'])[subject]


# print(features.shape)
# print(labels.shape)
# with np.printoptions(edgeitems=4000):
#     print(labels)

# print(features.shape)
# print(features)

#np.set_printoptions(threshold=np.inf)
#test_dict = {'f01':features}
# print(label_anaerobic(test_dict, signal_data['ANAEROBIC'])['f01'])

# for s in signal_data['ANAEROBIC'].keys():
#     print(s)
#     print(signal_data['ANAEROBIC'][s]['tags'])
