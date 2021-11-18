import os
import cmat
import pickle
import math
import collections
from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection


def sliding_window(x, y=None, sequence_length=None, overlapping=0, padding_value=0):
    '''Creates a sliding window arrays of the given array

    If the input array is not divisible by the sequence_length,
    the first samples are removed to fit.

    Parameters
    ----------
    x : np.array
    y : np.array
    sequence_length : int
    overlapping : int
        Between 0 and 1, how strong is the overlap
    padding_value : int or NaN, optional
        To ensure same window size, padding is required
        (default is 0)

    Returns
    -------
    : np.array, np.array

    '''
    full_array = x.values
    new_array = None
    # Input array:
    for axis in range(full_array.shape[-1]):
        array = full_array[:,axis]
        array_ext = np.full(sequence_length-1, padding_value)
        array_ext = np.concatenate((array_ext, array))
        strided = np.lib.stride_tricks.as_strided
        windows = strided(array_ext,
                          shape=(array.shape[0], sequence_length),
                          strides=array_ext.strides*2)
        slider = math.floor(sequence_length*(1-overlapping))
        windows = windows[0::slider]
        windows = windows.reshape(list(windows.shape) + [1])
        if new_array is None:
            new_array = windows
        else:
            new_array = np.append(new_array, windows, axis=2)
    if y is not None:
        # Label array (padding value is -1):
        full_labels = y.values
        label_ext = np.full(sequence_length-1, -1)
        label_ext = np.concatenate((label_ext, full_labels))
        label_windows = strided(label_ext,
                                shape=(full_labels.shape[0],
                                       sequence_length),
                                strides=label_ext.strides*2)
        label_windows = label_windows[0::slider]
        # The padded window is removed if necessary
        if len(array)%sequence_length != 0:
            new_array = new_array[1:]
            label_windows = label_windows[1:]
        major_labels = []
        # Majority voting for each window (ignore -1)
        for label_window in label_windows:
            lw = list(label_window)
            major_label = Counter(list(lw)).most_common()[0][0]
            major_labels.append(major_label)
        return new_array, np.array(major_labels)
    return new_array


def pick_majority_class(y):
    """
    Get the majority value along second axis.
    """
    # This is somehow 10x faster than pd.DataFrame.mode
    return pd.Series([collections.Counter(seq).most_common(1)[0][0] for seq in y])


def replace_classes(y, replace_dict):
    if replace_dict:
        return y.replace(replace_dict)
    else:
        return y


def grid_search(args):
    """
    Wrapper around sklearn's parameter grid. Mends
    dict values that are not wrapped in list
    """
    args = args if isinstance(args, (list, tuple)) else [args]
    return sklearn.model_selection.ParameterGrid([
        {k: v if isinstance(v, (list, tuple)) else [v] for k, v in a.items()}
        for a in args
    ])


def cv_split(data, folds, randomize=0):
    """
    Do a cross validation split on subjects
    """
    if folds > len(data):
        raise ValueError(f'More folds than subjects provided {folds} > {len(data)}')
    # Do leave-one-out if fold is zero or a negative number
    if folds <= 0:
        folds = len(data)
    # Make a list of subjects and do a seeded shuffle if configured
    subjects = list(data)
    if randomize > 0:
        np.random.seed(randomize)
        np.random.shuffle(subjects)
    # Get step size and loop over folds
    step = int(np.ceil(len(data) / folds))
    for fold in range(folds):
        valid = subjects[fold * step:(fold + 1) * step]
        train = [s for s in subjects if not s in valid]
        yield fold, train, valid


def get_multiset_cv_split(data, subject_groups, num_test,
                          num_valid=0, randomize=0):
    data_groups = []
    # Split the data into the subgroups:
    for group in subject_groups:
        corresponding_subjects = {}
        for subject in data.keys():
            if subject in group:
                corresponding_subjects[subject] = data[subject]
        corresponding_subjects = list(corresponding_subjects)
        if randomize > 0:
            np.random.seed(randomize)
            np.random.shuffle(corresponding_subjects)
        data_groups.append(corresponding_subjects)
    # Create the train/test split:
    folds = min([len(g) for g in data_groups])
    folds = folds//((num_test+num_valid)//len(data_groups))
    for fold in range(folds):
        fold_val = []
        fold_tst = []
        fold_tr = []
        step_val = num_valid//len(data_groups)
        step_tst = num_test//len(data_groups)
        current_start = fold * (step_val + step_tst)
        for subjects in data_groups:
            start = current_start
            end = start + step_val
            val = subjects[start:end]
            start = end
            end = start + step_tst
            tst = subjects[start:end]
            tr = [s for s in subjects if s not in val and s not in tst]
            fold_val += val
            fold_tst += tst
            fold_tr += tr
        yield fold, fold_tr, fold_tst # , fold_val


def read_chunked_data(file, batch_size, sequence_length):
    chunksize = batch_size * sequence_length
    it = pd.read_csv(file,
                     index_col=0, parse_dates=[0],
                     chunksize=chunksize
                     )
    for chunk in it:
        if len(chunk) < chunksize:
            chunk = chunk[:len(chunk) - len(chunk) % sequence_length]
            if len(chunk):
                yield chunk
            break
        yield chunk


def find_best_args(Model, data, config, sensor_args,
                   intermediate_save_path=None, scale=False,
                   existing_arguments=[], loo=False):
    '''Loop over parameters in grid manually and use CV

    Returns
    -------
    float: average(across folds) f1-score of best model
    dict: best hyperparameters
    scale: bool
        Whether to normalize the data before training
    loo: bool
        Whether to use loo or grid search+cross validation

    '''
    grid_cms = []
    grid_args = []
    for i, args in enumerate(grid_search(sensor_args)):
        print(f'Evaluating arguments: {args}', flush=True)
        if config.SKIP_FINISHED_ARGS and args in existing_arguments:
                print(f'Skipping existing arguments: {args}',
                      flush=True)
                continue
        cv_cms = []
        # Loop over CV folds
        # for j, train, valid in cv_split(data, config.CV_FOLDS, config.CV_RANDOM):
        s_groups = config.subject_groups
        num_to_test = config.GS_NUM_TEST
        randomize = config.CV_RANDOM
        if loo:
            data_split = cv_split(data,0,randomize=randomize)
            print('Do LOSO')
        else:
            data_split = get_multiset_cv_split(data,
                                               s_groups,
                                               num_to_test,
                                               randomize=randomize)
            print('Do Grid Search with CV')

        for j, train, valid in data_split:
            print(f'Fold {j}: valid={valid}')
            # Separate train/valid data
            train_x = pd.concat([data[s][0] for s in train],
                                axis=0, ignore_index=True)
            train_y = pd.concat([data[s][1] for s in train],
                                axis=0, ignore_index=True)
            valid_x = pd.concat([data[s][0] for s in valid],
                                axis=0, ignore_index=True)
            valid_y = pd.concat([data[s][1] for s in valid],
                                axis=0, ignore_index=True)
            # Create and train classifier
            model = Model.create(train_x, train_y, scale=scale, **args)
            # Evaluate it
            valid_y_hat = model.predict(valid_x)[0]
            # Collect statistics and store result for given metric
            cm = cmat.create(valid_y, valid_y_hat,
                             config.class_labels,
                             config.class_names)
            cv_cms.append(cm)
        if intermediate_save_path is not None:
            # Save cmat object and args in pickle file:
            save_intermediate_cmat(intermediate_save_path,
                                   'args_'+str(i).zfill(6)+'.pkl',
                                   args, cv_cms)
        # Store args used so we can find the best
        print(pd.concat([r.report.rename(f'fold_{i}') for i, r in enumerate(cv_cms)], axis=1))
        grid_args.append(args)
        grid_cms.append(cv_cms)
        print()
    # Find best average score and corresponding arguments.
    grid_scores = pd.concat([
        pd.concat([cm.report.rename(f'fold_{i}') for i, cm in enumerate(cms)], axis=1).loc[[config.CV_METRIC]]
        for cms in grid_cms
    ], ignore_index=True)
    averages = grid_scores.mean(axis=1)
    best_idx = averages.idxmax()
    best_args = grid_args[best_idx]
    # Save best cmat object and args in pickle file:
    if intermediate_save_path is not None:
        # Save cmat object and args in pickle file: 
        save_intermediate_cmat(intermediate_save_path,
                               'best_args.pkl',
                               best_args, grid_cms[best_idx])
    print(f'Best avg {config.CV_METRIC}: idx={best_idx} score={averages[best_idx]: .3f} args={best_args}')
    print(f'{config.CV_METRIC} for all grid iterations and folds:')
    print(grid_scores)
    print('CMATS saved at: ', intermediate_save_path)

    return averages[best_idx], best_args


def save_intermediate_cmat(path, filename, args, cmats):
    # Save cmat object and args in pickle file:
    args_cmats = [args, cmats]
    if not os.path.exists(path):
        os.makedirs(path)
    filehandler = open(path+filename, 'wb')
    pickle.dump(args_cmats, filehandler)
    filehandler.close()


def get_existing_features(path, label_column, fold_nr):
    '''Features and labels extracted from csv files'''
    folds = os.listdir(path)
    for fold in folds:
        folder_nr = int(fold.split('_')[-1])
        if folder_nr != fold_nr:
            continue
        print('Get from folder: ', fold)
        data = {}
        files = os.listdir(path+'/'+fold)
        for f in files:
            df = pd.read_csv(path+'/'+fold+'/'+f)
            y = df[label_column]
            feature_columns = list(df.columns.drop(label_column))
            feature_columns = [fc for fc in feature_columns if 'f' in fc]
            x = df[feature_columns]
            data[f] = (x,y)
        return data
