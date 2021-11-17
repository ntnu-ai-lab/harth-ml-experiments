import glob
import os
import pickle
import pandas as pd
import src.config
import src.featurizer
import src.models
import src.utils

import argparse

def loso_cv(config_path, dataset_path=None):
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
    cmat_path = (config_path + 'loo_cmats_' + '_scale_' + str(scale) +
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
        if config.SKIP_FINISHED_ARGS and sensor_args in existing_arguments:
                print(f'Skipping existing arguments: {sensor_args}',
                      flush=True)
                continue
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
            x = src.featurizer.Featurizer.get(config.FEATURES, x, columns,
                                              sample_rate=config.SAMPLE_RATE) 
            # Tranform np array to series
            y = pd.Series(y)
            # Add to set of preprocessed data
            data[subject] = (x, y)

        # Get model class
        Model = src.models.get_model(config.ALGORITHM)
        # Search for best args using GS and CV
        best_score, best_args = src.utils.find_best_args(Model, data,
                                                         config, sensor_args,
                                                         cmat_path, scale=scale,
                                                         loo=True)
        if config.TRAIN_ON_FULL_DATASET:
            print(f'Training on full dataset')
            # Concatenate all training data
            train_x = pd.concat([data[s][0] for s in data],
                                axis=0, ignore_index=True)
            train_y = pd.concat([data[s][1] for s in data],
                                axis=0, ignore_index=True)
            # Train classifier and save it
            model = Model.create(train_x, train_y, **best_args)
            model_path = (config_path+sensor+'_loo_scale_'+str(scale) +
                          '_' + str(seconds) + 'sec_' + str(config.OVERLAP) +
                          'overlap_' + train_data_path + 'model.pkl')
            model.save(model_path)
            print(f'Model saved to {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start LOSO CV.')
    parser.add_argument('-p', '--params_path', required=True, type=str,
                        help='params path with config.yml file')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    args = parser.parse_args()
    config_path = args.params_path+'/'
    ds_path = args.dataset_path
    loso_cv(config_path, ds_path)
