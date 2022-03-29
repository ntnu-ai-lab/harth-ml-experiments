import glob
import os
import pickle
import numpy as np
import pandas as pd
import src.config
import src.featurizer
import src.models
import src.utils

import argparse
import cmat

def train(config, dataset_path=None, **kwargs):
    if dataset_path is None:
        dataset_path = config.TRAIN_DATA
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'Dataset {dataset_path} does not exist')

    #### Check existing arguments ####
    seconds = config.SEQUENCE_LENGTH//50
    scale = config.SCALE_DATA
    train_data_path = dataset_path.split('/')[-1]
    cmat_path = (f'{config.CONFIG_PATH}/cmats' +
                 f'_fold{config.FOLDS}' +
                 f'_scale{scale}' +
                 f'_{seconds}sec' +
                 f'_{config.FRAME_SHIFT}frame_shift' +
                 f'_{train_data_path}/')
    existing_arguments = []
    if config.SKIP_FINISHED_ARGS and os.path.exists(cmat_path):
        for cmat_file in os.listdir(cmat_path):
            if 'best_idx' not in cmat_file and cmat_file.endswith('.pkl'):
                with open(cmat_path+cmat_file, 'rb') as f:
                    existing_arguments.append(pickle.load(f)[0])
    if not os.path.exists(cmat_path):
        os.makedirs(cmat_path)
    ##################################

    # Read train data from csv files in configured directory
    data, ground_truths = src.utils.load_dataset(dataset_path, config)
    # # Get model class
    Model = src.models.get_model(config.ALGORITHM)

    # Training
    ###################################################
    grid_cms = []
    grid_args = []
    for i, args in enumerate(src.utils.grid_search(config.sensor_args)):
        print(f'Evaluating arguments: {args}', flush=True)
        if args in existing_arguments:
                print(f'Skipping existing arguments: {args}',
                      flush=True)
                continue
        cv_cms = []
        # Loop over CV folds
        s_groups = config.subject_groups
        num_to_test = config.GS_NUM_TEST
        randomize = config.CV_RANDOM
        if s_groups is None:  # Do standard cross validation
            data_split = src.utils.cv_split(data, config.FOLDS,
                                            randomize=randomize)
            print('Do Grid Search with',str(config.FOLDS),'-fold CV')
        else:  # Do split according to defined subsets
            data_split = src.utils.get_multiset_cv_split(
                                               data,
                                               s_groups,
                                               num_to_test,
                                               randomize=randomize)
            print('Do Grid Search with CV on subject subgroups')

        for j, train, valid in data_split:
            print(f'Fold {j}: valid={valid}')
            # Separate train/valid data
            train_x = pd.concat([data[s][0] for s in train],
                                axis=0, ignore_index=True)
            train_y = pd.concat([data[s][1] for s in train],
                                axis=0, ignore_index=True)
            valid_x = pd.concat([data[s][0] for s in valid],
                                axis=0, ignore_index=True)
            valid_y = np.concatenate([ground_truths[s] for s in valid],
                                 axis=0)
            # Create and train classifier
            model = Model.create(train_x, train_y, scale=scale, **args)
            # Evaluate it
            valid_y_hat = model.predict(valid_x)[0]

            valid_y_hat = src.utils.unfold_windows(
                valid_y_hat.reshape(-1,1),
                config.SEQUENCE_LENGTH,
                config.FRAME_SHIFT
            ).reshape(-1).astype(int)

            # Collect statistics and store result for given metric
            cm = cmat.create(valid_y, valid_y_hat,
                             config.class_labels,
                             config.class_names)
            cv_cms.append(cm)

        # Save cmat object and args in pickle file:
        src.utils.save_intermediate_cmat(
            cmat_path,
            'args_'+str(i).zfill(6)+'.pkl',
            args, cv_cms
        )
        # Store args used so we can find the best
        print(pd.concat([r.report.rename(f'fold_{i}')
                         for i, r in enumerate(cv_cms)], axis=1))
        grid_args.append(args)
        grid_cms.append(cv_cms)
        print()
    # Find best average score and corresponding arguments.
    if len(grid_cms) != 0:
        grid_scores = pd.concat([
            pd.concat([cm.report.rename(f'fold_{i}')
                       for i, cm in enumerate(cms)],
                      axis=1).loc[[config.CV_METRIC]]
            for cms in grid_cms
        ], ignore_index=True)
        averages = grid_scores.mean(axis=1)
        best_idx = averages.idxmax()
        best_args = grid_args[best_idx]
        # Save best cmat object and args in pickle file:
        src.utils.save_intermediate_cmat(
            cmat_path,
            'best_args.pkl',
            best_args, grid_cms[best_idx]
        )
        print(f'Best avg {config.CV_METRIC}: idx={best_idx} score={averages[best_idx]: .3f} args={best_args}')
        print(f'{config.CV_METRIC} for all grid iterations and folds:')
        print(grid_scores)
        print('CMATS saved at: ', cmat_path)

        if config.TRAIN_ON_FULL_DATASET:
            print(f'Training on full dataset')
            # Concatenate all training data
            train_x = pd.concat([data[s][0] for s in data],
                                axis=0, ignore_index=True)
            train_y = pd.concat([data[s][1] for s in data],
                                axis=0, ignore_index=True)
            # Train classifier and save it
            model = Model.create(train_x, train_y, **best_args)
            model_path = os.path.join(config.CONFIG_PATH, 'model.pkl')
            model.save(model_path)
            print(f'Model saved to {model_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start ML training.')
    parser.add_argument('-p', '--params_path', required=True, type=str,
                        help='params path with config.yml file')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    train(config, ds_path)
