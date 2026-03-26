import os
import numpy as np
import pandas as pd
import datetime

# create a vector from the data frame (signal imported by pandas)
def create_df_array(dataframe):
    matrix_df=dataframe.values
    # returns 2-d matrix
    matrix = np.array(matrix_df)
    array_df = matrix.flatten()# Convert matrix into an array
    return array_df

# convert UTC arrays to arrays in seconds relative to 0 (record beginning)
def time_abs_(UTC_array):
    new_array=[]
    for utc in UTC_array:
        time=(datetime.datetime.strptime(utc,'%Y-%m-%d %H:%M:%S')-datetime.datetime.strptime(UTC_array[0], '%Y-%m-%d %H:%M:%S')).total_seconds()
        new_array.append(int(time))
    return new_array

def read_signals(main_folder):
    signal_dict = {}
    time_dict = {}
    fs_dict = {}

    # Get a list of subfolders in the main folder
    subfolders = next(os.walk(main_folder))[1]

    utc_start_dict = {}
    for folder_name in subfolders:
        csv_path = f'{main_folder}/{folder_name}/EDA.csv'
        df = pd.read_csv(csv_path)
        utc_start_dict[folder_name] = df.columns.tolist()

    # Iterate over the subfolders
    for folder_name in subfolders:
        folder_path = os.path.join(main_folder, folder_name)
        # Get a list of files in the subfolder
        files = os.listdir(folder_path)

        # Initialize a dictionary for the signals in the current subfolder
        signals = {}
        time_line = {}
        fs_signal = {}

        # Define the list of desired file names
        desired_files = ['EDA.csv', 'BVP.csv', 'HR.csv', 'TEMP.csv', 'tags.csv', 'ACC.csv']

        # Iterate over the files in the subfolder
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)

            # Check if it's a CSV file and if it is in the desired files list
            if file_name.endswith('.csv') and file_name in desired_files:
                # Read the CSV file and store the signal data

                if file_name == 'tags.csv':
                    try:
                        df = pd.read_csv(file_path, header=None)
                        tags_vector = create_df_array(df)
                        tags_UTC_vector = np.insert(tags_vector, 0, utc_start_dict[folder_name])
                        signal_array = time_abs_(tags_UTC_vector)
                    except pd.errors.EmptyDataError:
                        signal_array = []

                else:
                    df = pd.read_csv(file_path)
                    fs = df.loc[0]
                    fs = int(fs.iloc[0])  # Get sampling frequency
                    df.drop([0], axis=0, inplace=True)
                    signal_array = df.values
                    time_array = np.linspace(0, len(signal_array) / fs, len(signal_array))

                signal_name = file_name.split('.')[0]
                signals[signal_name] = signal_array
                time_line[signal_name] = time_array
                fs_signal[signal_name] = fs

        # Store the signals of the current subfolder in the main dictionary
        signal_dict[folder_name] = signals
        time_dict[folder_name] = time_line
        fs_dict[folder_name] = fs_signal

    return signal_dict, time_dict, fs_dict


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
    participants[state]=os.listdir(folder_path)
    signal_data[state], time_data[state], fs_dict[state] = read_signals(folder_path) # Returns three dictionaries with subjects info: raw signals (signal_data), temporal data ready to graph (time_data) and sample frequency for escha signal(fs_dict).


# print(signal_data['AEROBIC']['f01']['ACC'].shape)
# print(signal_data['AEROBIC']['f01']['HR'].shape)
# print(fs_dict['AEROBIC']['f01']['ACC'])
# print(fs_dict['AEROBIC']['f01']['HR'])

# print(signal_data['AEROBIC']['f01']['EDA'].shape)
# print(signal_data['AEROBIC']['f01']['TEMP'].shape)
# print(signal_data['AEROBIC']['f01']['BVP'].shape)
#
# print('Time data')
# print(time_data['AEROBIC']['f01']['ACC'].shape)
# print(time_data['AEROBIC']['f01']['HR'].shape)
# print(time_data['AEROBIC']['f01']['HR'])
#
# print(time_data['AEROBIC']['f02']['HR'].shape)
# print(fs_dict['AEROBIC']['f02']['BVP'])

