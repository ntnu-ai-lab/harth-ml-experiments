import glob
import os
import pickle
import pandas as pd
import src.config
import src.featurizer
import src.models
import src.utils

import argparse

def train(config_path, dataset_path=None):
    # Read config 
    config = src.config.Config(config_path+'config.yml')
    if dataset_path is None:
        dataset_path = config.TRAIN_DATA
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'Dataset {dataset_path} does not exist')

    #### Check existing arguments ####
    seconds = config.SEQUENCE_LENGTH//50
    scale = config.SCALE_DATA
    train_data_path = dataset_path.split('/')[-1]
    cmat_path = (config_path + 'cmats_' + '_scale_' + str(scale) +
                 '_' + str(seconds) + 'sec_' + str(config.OVERLAP) +
                 'overlap_' + train_data_path+'/')
    if config.SKIP_FINISHED_ARGS:
        existing_arguments = []
        if os.path.exists(cmat_path):
            for cmat_file in os.listdir(cmat_path):
                if 'best_idx' not in cmat_file:
                    with open(cmat_path+cmat_file, 'rb') as f:
                        existing_arguments.append(pickle.load(f)[0])
    ##################################

    # Read train data from csv files in configured direrctory
    print(f'Reading train data from {dataset_path}')
    subjects = {}
    for path in glob.glob(os.path.join(dataset_path, '*.csv')):
        subjects[os.path.basename(path)] = pd.read_csv(path)

    # Train one model for each sensor
    for sensor, columns, sensor_args in config.sensors:
        print(f'Training model for sensor {sensor}')
        # Grab data corresponding to column and compute feature
        data = {}
        for subject, subject_data in subjects.items():
            print(f'Preprocessing: {subject}')
            x = subject_data[columns]
            y = subject_data[config.LABEL_COLUMN]
            # Replace classes with majority
            y = src.utils.replace_classes(y, config.replace_classes)
            # Sample windows
            x, y = src.utils.sliding_window(x, y,
                                            config.SEQUENCE_LENGTH,
                                            overlapping=config.OVERLAP)
            # Generate features
            x = src.featurizer.Featurizer.get(config.FEATURES,
                                              x, columns,
                                              sample_rate=config.SAMPLE_RATE)
            # Tranform np array to series
            y = pd.Series(y)
            # Add to set of preprocessed data
            data[subject] = (x, y)

        # # Get model class
        Model = src.models.get_model(config.ALGORITHM)
        # # Search for best args using GS and CV
        best_score, best_args = src.utils.find_best_args(
                                     Model, data,
                                     config,
                                     sensor_args,
                                     intermediate_save_path=cmat_path,
                                     scale=scale,
                                     existing_arguments=existing_arguments,
                                     loo=False
                                     )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start ML training.')
    parser.add_argument('-p', '--params_path', required=True, type=str,
                        help='params path with config.yml file')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    args = parser.parse_args()
    config_path = args.params_path+'/'
    ds_path = args.dataset_path
    train(config_path, ds_path)
