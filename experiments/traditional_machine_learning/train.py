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
    if type(dataset_path)==str:
        dataset_path = [dataset_path]
    for ds_p in dataset_path:
        if not os.path.exists(ds_p):
            raise FileNotFoundError(f'Dataset {ds_p} does not exist')
    seq_len = config.SEQUENCE_LENGTH if type(config.SEQUENCE_LENGTH)==int \
            else config.SEQUENCE_LENGTH[0]
    #### Check existing arguments ####
    scale = config.SCALE_DATA
    cmat_path = (f'{config.CONFIG_PATH}/cmats' +
                 f'_folds{config.FOLDS}/')
    if config.FORCE_NEW_CMAT_PATH:
        cmat_path_iter = 0
        while os.path.exists(cmat_path):
            cmat_path = (f'{config.CONFIG_PATH}/cmats' +
                         f'_folds{config.FOLDS}_rep_{cmat_path_iter}/')
            cmat_path_iter += 1
        print(f'cmat_path: {cmat_path}')
    existing_arguments = []
    existing_cms = []
    existing_subjects = []
    if config.SKIP_FINISHED_ARGS and os.path.exists(cmat_path):
        for cmat_file in os.listdir(cmat_path):
            if 'best' not in cmat_file and cmat_file.endswith('.pkl'):
                with open(cmat_path+cmat_file, 'rb') as f:
                    fpkl = pickle.load(f)
                    existing_arguments.append(fpkl[0])
                    existing_cms.append(fpkl[1])
                    existing_subjects.append(fpkl[2])
    if not os.path.exists(cmat_path) and config.STORE_CMATS:
        os.makedirs(cmat_path)
    ##################################
    # Read train data from csv files in configured directory
    data, ground_truths, valid_x_lasts = src.utils.load_dataset(
        dataset_path,
        config
    )
    # # Get model class
    Model = src.models.get_model(config.ALGORITHM)
    # Training
    ###################################################
    grid_cms = []
    grid_args = []
    for i, args in enumerate(src.utils.grid_search(config.sensor_args)):
        print(f'Evaluating arguments: {args}', flush=True)
        if args not in existing_arguments:
            cv_cms = []
            cv_valid_subjects = []
            if config.WANDB:
                ds_name = '_'.join([os.path.realpath(x).split('/')[-1] for x in dataset_path])
                proj_name = 'harth_plus_'+ds_name
                wandb_config = args.copy()
                wandb_config.update(vars(config))
                src.utils.wandb_init(
                    run_name=config.CONFIG_NAME+'_'+config.ALGORITHM,
                    wandb_config=wandb_config,
                    entity='hunt4-har',
                    proj_name=proj_name,
                )
                all_valid_y_hat = []
                all_valid_y = []
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
                fold_valid_y = []
                fold_valid_y_hat = []
                # Separate train/valid data
                train_x = pd.concat([data[s][0] for s in train],
                                    axis=0, ignore_index=True)
                train_y = pd.concat([data[s][1] for s in train],
                                    axis=0, ignore_index=True)
                # Create and train classifier
                model = Model.create(train_x, train_y,
                                     scale=scale,
                                     sampler=config.SAMPLER,
                                     dim_reduction=config.DIM_REDUCTION,
                                     **args)
                # Evaluate it for each valid subj
                for _s in valid:
                    valid_x = data[_s][0]
                    valid_y = ground_truths[_s]
                    valid_x_last = valid_x_lasts[_s]
                    valid_y_hat = model.predict(valid_x)[0]
                    # Unfolding subject prediction
                    valid_y_hat = src.utils.unfold_windows(
                        valid_y_hat.reshape(-1,1),
                        seq_len,
                        config.FRAME_SHIFT,
                        overlap_kind='last'
                    ).reshape(-1).astype(int)
                    if len(valid_x_last) != 0:
                        valid_y_last_hat = model.predict(valid_x_last)[0]
                        # Append last "cutted" prediction to get predictions
                        # for the whole signal
                        num_to_append = len(valid_y)-len(valid_y_hat)
                        valid_y_last_hat = valid_y_last_hat.tolist()*num_to_append
                        valid_y_hat = list(valid_y_hat) + valid_y_last_hat
                    # Collect statistics and store result for given metric
                    fold_valid_y += list(valid_y)
                    fold_valid_y_hat += list(valid_y_hat)
                    if config.WANDB:
                        src.utils.log_wandb(
                            y_pred=valid_y_hat,
                            y_true=valid_y,
                            params_config=config,
                            log_name=str(_s)
                        )
                        all_valid_y_hat+=list(valid_y_hat)
                        all_valid_y+=list(valid_y)
                    if config.save_predictions:
                        pred_path = os.path.join(config.CONFIG_PATH, 'predictions')
                        src.utils.save_predictions(
                            folder_path=pred_path,
                            filename=str(_s),
                            predictions=valid_y_hat,
                            ground_truths=valid_y,
                            index=valid_y.index
                        )
                cm = cmat.create(fold_valid_y, fold_valid_y_hat,
                                 config.class_labels,
                                 config.class_names)
                cv_cms.append(cm)
                cv_valid_subjects.append(valid)
            if config.WANDB:
                src.utils.log_wandb_class_distribution(
                    all_valid_y,
                    label_name_dict=config.class_label_name_map
                )
                src.utils.log_wandb_class_sample_width(
                    all_valid_y,
                    label_name_dict=config.class_label_name_map
                )
                src.utils.log_wandb_cmat(
                    y_pred=all_valid_y_hat,
                    y_true=all_valid_y,
                    label_names=config.class_names,
                    log_name='all_subjects',
                    label_mapping=config.label_index
                )
        else:
            print(f'Skipping existing arguments: {args}',
                  flush=True)
            args_i = existing_arguments.index(args)
            cv_cms = existing_cms[args_i]
            cv_valid_subjects = existing_subjects[args_i]

        # Save cmat object and args in pickle file:
        if config.STORE_CMATS:
            src.utils.save_intermediate_cmat(
                cmat_path,
                'args_'+str(i).zfill(6)+'.pkl',
                args, cv_cms, cv_valid_subjects
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
        stds = grid_scores.std(axis=1)
        best_idx = averages.idxmax()
        best_args = grid_args[best_idx]
        # Save best cmat object and args in pickle file:
        if config.STORE_CMATS:
            src.utils.save_intermediate_cmat(
                cmat_path,
                'best_args.pkl',
                best_args, grid_cms[best_idx]
            )
            print('CMATS saved at: ', cmat_path)
        print(f'Best avg {config.CV_METRIC}: idx={best_idx} score={averages[best_idx]: .3f} args={best_args}')
        print(f'{config.CV_METRIC} for all grid iterations and folds:')
        print(grid_scores)

        if config.TRAIN_ON_FULL_DATASET:
            print(f'Training on full dataset')
            # Concatenate all training data
            train_x = pd.concat([data[s][0] for s in data],
                                axis=0, ignore_index=True)
            train_y = pd.concat([data[s][1] for s in data],
                                axis=0, ignore_index=True)
            # Train classifier and save it
            # model = Model.create(train_x, train_y, **best_args)
            model = Model.create(train_x, train_y, scale=scale,
                                 sampler=config.SAMPLER, **args)
            model_path = os.path.join(config.CONFIG_PATH, 'model.pkl')
            model.save(model_path)
            print(f'Model saved to {model_path}')
        return averages[best_idx], stds[best_idx], best_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start ML training.')
    parser.add_argument('-p', '--params_path', required=False, type=str,
                        help='params path with config.yml file',
                        default='/param/config.yml')
    parser.add_argument('-d', '--dataset_path', required=False, type=str,
                        help='path to dataset.', default=None)
    args = parser.parse_args()
    config_path = args.params_path
    # Read config
    config = src.config.Config(config_path)
    ds_path = args.dataset_path
    train(config, ds_path)
