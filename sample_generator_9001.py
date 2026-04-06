import test
import os
import numpy as np
import time
import random

from data_aligner_9000 import align_features, align_all, label_stress, label_aerobic, label_anaerobic, get_features

# Returns array of tuples: each tuple is a block, with first element matrix of features and second a vector of labels
def generate_samples():
    signal_data, time_data, fs_dict, participants = get_features()
    fs_dict = fs_dict['AEROBIC']['f01'] # Sampling frequencies are the same for every experiment so we simplify the dict

    samples = []
    for method in ["STRESS", "AEROBIC", "ANAEROBIC"]:
        print(method)
        if   method == "STRESS":
            bad_keys = ['f07', "f14_a", "f14_b"]
            labels = label_stress(signal_data['STRESS'], time_data['STRESS'], bad_keys)

        elif method == "AEROBIC": 
            bad_keys = ['S03', 'S07', 'S11_a', 'S11_b', 'S12']
            labels = label_aerobic(signal_data['AEROBIC'], time_data['AEROBIC'], bad_keys)
        else:
            bad_keys = ['S06', 'S16_a', 'S16_b']
            labels = label_anaerobic(signal_data['ANAEROBIC'], time_data['ANAEROBIC'], bad_keys)

        for subject in signal_data[method].keys():
            if subject in bad_keys: continue
            indices = []
            features = align_features(signal_data[method][subject], time_data[method][subject])
            print(subject)
            if   method == "STRESS":  
                start_point = signal_data[method][subject]["tags"][1]
                if "S" in subject:
                    end_point = signal_data[method][subject]["tags"][13]
                    block_count = 2
                else:
                    end_point = signal_data[method][subject]["tags"][9]
                    block_count = 3
            elif method == "AEROBIC":
                end_point = signal_data[method][subject]["tags"][-1]
                if "S" in subject :
                    block_count = 2
                    start_point = signal_data[method][subject]["tags"][1] #The first tag, plus some bonus time where the subject is not exercising yet.
                else: 
                    block_count = 1
                    if subject == 'f02':
                        start_point = signal_data[method][subject]["tags"][1]   #This one without the moment of idleness, because f02 started the test more quickly
                    else:
                        start_point = signal_data[method][subject]["tags"][1] - 200 #The first tag, plus some bonus time where the subject is not exercising yet.
                    if subject in ['f02', 'f07', 'f11']:
                        end_point -= 10     #These tests have tags after the end of the sample, smh
            else:           
                end_point = signal_data[method][subject]["tags"][-1]
                if "S" in subject : 
                    block_count = 1
                    start_point = signal_data[method][subject]["tags"][1] - 100
                else: 
                    block_count = 2
                    start_point = signal_data[method][subject]["tags"][1]
            end_point -= 500

            indices = [start_point, end_point]
            indices.extend(np.random.uniform(start_point, end_point, block_count))

            for i in indices:
                i_int = int(i)
                sample_features = features[i_int:i_int+500]
                if sample_features.shape[0] != 500:
                    print(method, subject, i)     #For when something goes wrong
                sample_labels = labels[subject][i_int:i_int+500]
                samples.append((sample_features, sample_labels))
    return samples

def store_samples(samples, sample_length, features):
    print('Writing ', len(samples), ' samples')

    feature_matrix = np.empty((len(samples), sample_length, features))
    label_matrix = np.empty((len(samples), sample_length))

    for i in range(len(samples)):
        feature_matrix[i] = samples[i][0]
        label_matrix[i] = samples[i][1]

    np.save("processed_data/features", feature_matrix, allow_pickle=False)
    np.save("processed_data/labels", feature_matrix, allow_pickle=False)

# signal_data, time_data, fs_dict, participants = get_features()
#
#
# print("stress")
# bad_keys = ['f07', "f14_a", "f14_b"]
# labels = label_stress(signal_data['STRESS'], time_data['STRESS'], bad_keys)
#
# print("aerosbic")
# bad_keys = ['S03', 'S07', 'S11_a', 'S11_b', 'S12']
# labels = label_aerobic(signal_data['AEROBIC'], time_data['AEROBIC'], bad_keys)
#
# print("anareoic")
# bad_keys = ['S06', 'S16_a', 'S16_b']
# labels = label_anaerobic(signal_data['ANAEROBIC'], time_data['ANAEROBIC'], bad_keys)



samples = generate_samples()
store_samples(samples, 500, 7)

# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])


