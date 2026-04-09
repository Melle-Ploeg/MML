import test
import os
import numpy as np
import time
import random

from data_aligner_9000 import align_features, align_all, label_stress, label_aerobic, label_anaerobic, get_features

# Returns array of tuples: each tuple is a block, with first element matrix of features and second a vector of labels
def generate_samples(seq_length, block_multiplier=1):
    random.seed(234567)
    signal_data, time_data, fs_dict, participants = get_features()
    fs_dict = fs_dict['AEROBIC']['f01'] # Sampling frequencies are the same for every experiment so we simplify the dict

    train_samples = []
    test_samples = []
    val_samples = []
    for method in ["STRESS", "AEROBIC", "ANAEROBIC"]:
        print(method)
        keys = signal_data[method].keys()
        if   method == "STRESS":
            bad_keys = ['f07', "f14_a", "f14_b"]
            labels = label_stress(signal_data['STRESS'], time_data['STRESS'], bad_keys)

        elif method == "AEROBIC": 
            bad_keys = ['S03', 'S07', 'S11_a', 'S11_b', 'S12']
            labels = label_aerobic(signal_data['AEROBIC'], time_data['AEROBIC'], bad_keys)
        else:
            bad_keys = ['S06', 'S16_a', 'S16_b']
            labels = label_anaerobic(signal_data['ANAEROBIC'], time_data['ANAEROBIC'], bad_keys)
        
        keys = list(filter(lambda k: k not in bad_keys, keys))

        f_keys = list(filter(lambda k: "S" not in k, keys))
        S_keys = list(filter(lambda k: "S" in k, keys))

        test_val_f = random.sample(f_keys, 4)
        test_val_S = random.sample(S_keys, 4)
        test_keys = [test_val_f[0], test_val_f[1], test_val_S[0], test_val_S[1]]
        val_keys = [test_val_f[2], test_val_f[3], test_val_S[2], test_val_S[3]]

        for subject in keys:
            if subject in bad_keys: continue
            features = align_features(signal_data[method][subject], time_data[method][subject])
            print(subject)
            if   method == "STRESS":  
                start_point = signal_data[method][subject]["tags"][1]
                if "S" in subject:
                    end_point = signal_data[method][subject]["tags"][13]
                    block_count = 5*block_multiplier #2
                else:
                    end_point = signal_data[method][subject]["tags"][9]
                    block_count = 6*block_multiplier #3
            elif method == "AEROBIC":
                end_point = signal_data[method][subject]["tags"][-1]
                if "S" in subject :
                    block_count = 2*block_multiplier
                    start_point = signal_data[method][subject]["tags"][1] #The first tag, plus some bonus time where the subject is not exercising yet.
                else: 
                    block_count = 1*block_multiplier
                    if subject == 'f02':
                        start_point = signal_data[method][subject]["tags"][1]   #This one without the moment of idleness, because f02 started the test more quickly
                    else:
                        start_point = signal_data[method][subject]["tags"][1] - 200 #The first tag, plus some bonus time where the subject is not exercising yet.
                    if subject in ['f02', 'f07', 'f11']:
                        end_point -= 10     #These tests have tags after the end of the sample, smh
            else:           
                end_point = signal_data[method][subject]["tags"][-1]
                if "S" in subject : 
                    block_count = 1*block_multiplier
                    start_point = signal_data[method][subject]["tags"][1] - 100
                else: 
                    block_count = 2*block_multiplier
                    start_point = signal_data[method][subject]["tags"][1]
            end_point -= seq_length

            indices = [start_point, end_point]
            indices.extend(np.random.uniform(start_point, end_point, block_count))

            for i in indices:
                i_int = int(i)
                sample_features = features[i_int:i_int+seq_length]
                if sample_features.shape[0] != seq_length:
                    print(method, subject, i)     #For when something goes wrong
                
                sample_labels = np.zeros((seq_length, 4))
                for j in range(seq_length):
                    #print(labels[subject][i_int + j])
                    sample_labels[j][int(labels[subject][i_int + j])] = 1

                #sample_labels = labels[subject][i_int:i_int+500]

                if subject in test_keys:
                    test_samples.append((sample_features, sample_labels))
                elif subject in val_keys:
                    val_samples.append((sample_features, sample_labels))
                else:
                    train_samples.append((sample_features, sample_labels))
    return train_samples, test_samples, val_samples

def store_samples(samples, sample_length, features, classes, addition=""):
    print('Writing ', len(samples), ' samples')

    feature_matrix = np.empty((len(samples), sample_length, features))
    label_matrix = np.empty((len(samples), sample_length, classes))

    for i in range(len(samples)):
        feature_matrix[i] = samples[i][0]
        label_matrix[i] = samples[i][1]

    np.save("processed_data/features" + addition, feature_matrix, allow_pickle=False)
    np.save("processed_data/labels" + addition, label_matrix, allow_pickle=False)

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


feature_count=6
seq_length = 500

train_samples, test_samples, val_samples = generate_samples(seq_length)#, block_multiplier=10)
store_samples(train_samples, seq_length, feature_count, 4, "_train")
store_samples(test_samples, seq_length, feature_count, 4, "_test")
store_samples(val_samples, seq_length, feature_count, 4, "_val")

# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])


