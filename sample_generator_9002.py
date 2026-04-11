import test
import os
import numpy as np
import time
import random

from data_aligner_9000 import align_features, align_all, label_stress, label_aerobic, label_anaerobic, get_features

# Returns array of tuples: each tuple is a block, with first element matrix of features and second a vector of labels
def generate_samples(seq_length, stressFree = 5, stressFull = 5, anaerobic = 5, anaerobicChill = 5):
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
            sample_label = 1
            block_count_cont = stressFree
            block_count_test = stressFull
            #labels = label_stress(signal_data['STRESS'], time_data['STRESS'], bad_keys)
        elif method == "AEROBIC":
            continue 
        else:
            bad_keys = ['S06', 'S16_a', 'S16_b']
            sample_label = 2
            block_count_cont = anaerobicChill
            block_count_test = anaerobic
            #labels = label_anaerobic(signal_data['ANAEROBIC'], time_data['ANAEROBIC'], bad_keys)
        
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
                if "S" in subject:
                    start_point = signal_data[method][subject]["tags"][4]
                    start_point_test = signal_data[method][subject]["tags"][5]
                    end_point_test = signal_data[method][subject]["tags"][6]
                    end_point = signal_data[method][subject]["tags"][13]
                else:
                    start_point = signal_data[method][subject]["tags"][1]
                    start_point_test = signal_data[method][subject]["tags"][2]
                    end_point_test = signal_data[method][subject]["tags"][3]
                    end_point = signal_data[method][subject]["tags"][4]
                indices_control = list(range(start_point, start_point_test-seq_length)) + list(range(end_point_test, end_point-seq_length))
                indices_test = list(range(start_point_test, end_point_test-seq_length))
            else:
                end_point = signal_data[method][subject]["tags"][-1]
                if "S" in subject: 
                    #start_point = signal_data[method][subject]["tags"][1]
                    signal_data[method][subject]["tags"][7]
                    tagCount = 3
                    padding = 30
                else: 
                    #start_point = signal_data[method][subject]["tags"][1]
                    end_point = signal_data[method][subject]["tags"][9]
                    tagCount = 4
                    padding = 15
                
                indices_control = []
                indices_test = []
                for i in range(tagCount):
                    indices_control += list(range(max(0, signal_data[method][subject]["tags"][i*2]), max(0, signal_data[method][subject]["tags"][i*2+1]-60)))
                    indices_test += list(range(max(0, signal_data[method][subject]["tags"][i*2+1]-padding), max(0, signal_data[method][subject]["tags"][i*2+2]-60+padding))) #indices test has padding before and after the test
                indices_control += list(range(signal_data[method][subject]["tags"][tagCount*2-1], max(0, signal_data[method][subject]["tags"][tagCount*2]-60)))
            end_point -= seq_length

            for j in [0,1]:
                if j:
                    indices = indices_test
                    block_count = block_count_test
                else: 
                    indices = indices_control
                    block_count = block_count_cont

                for i in np.random.uniform(0, len(indices)-1, block_count):
                    i_int = int(i)
                    start_index = indices[i_int]

                    sample_features = features[start_index:start_index+seq_length]
                    if sample_features.shape[0] != seq_length:
                        print("SOMETHING WENT WRONG")
                        print(method, subject, start_index, j)     #For when something goes wrong

                    #sample_labels = labels[subject][i_int:i_int+500]

                    if subject in test_keys:
                        test_samples.append((sample_features, sample_label*j))
                    elif subject in val_keys:
                        val_samples.append((sample_features, sample_label*j))
                    else:
                        train_samples.append((sample_features, sample_label*j))
    print(train_samples[0][1])
    # for i in list(np.random.uniform(0, len(train_samples), 6)):
    #     print(train_samples[int(i)][1])
    return train_samples, test_samples, val_samples

def store_samples(samples, sample_length, features, addition=""):
    print('Writing ', len(samples), ' samples')

    feature_matrix = np.empty((len(samples), sample_length, features))
    label_vector = np.empty(len(samples))
    for i in range(len(samples)):
        feature_matrix[i] = samples[i][0]
        label_vector[i] = samples[i][1]

    print(label_vector)
    np.save("processed_data/features" + addition, feature_matrix, allow_pickle=False)
    np.save("processed_data/labels" + addition, label_vector, allow_pickle=False)

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
seq_length = 60

stressFull = 40
stressFree = 40
anaerobic = 40
anaerobicChill = 0#20

train_samples, test_samples, val_samples = generate_samples(seq_length, stressFree=stressFree, stressFull=stressFull, anaerobic=anaerobic, anaerobicChill=anaerobicChill)
store_samples(train_samples, seq_length, feature_count, "_train-v2_onlyStrest")
store_samples(test_samples, seq_length, feature_count, "_test-v2_onlyStrest")
store_samples(val_samples, seq_length, feature_count, "_val-v2_onlyStrest")

# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])
# print(samples[int(np.random.uniform(0, len(samples)))])