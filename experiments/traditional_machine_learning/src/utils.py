import os
import glob
import cmat
import pickle
import math
import collections
from collections import Counter
import numpy as np
import pandas as pd
import sklearn.model_selection
import src.featurizer


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


def save_intermediate_cmat(path, filename, args, cmats):
    # Save cmat object and args in pickle file:
    args_cmats = [args, cmats]
    if not os.path.exists(path):
        os.makedirs(path)
    filehandler = open(path+filename, 'wb')
    pickle.dump(args_cmats, filehandler)
    filehandler.close()


def windowed_labels(
    labels,
    num_labels,
    frame_length,
    frame_step=None,
    pad_end=False,
    kind='density',
):
    """Segmenting a label array

    With kind=None we are able to split the given labels
    array into batches.

    Parameters
    ----------
    labels : np.array
        Array of

    Returns
    -------
    : np.array
    """
    # Labels should be a single vector (int-likes) or kind has to be None
    labels = np.asarray(labels)
    if kind is not None and not labels.ndim == 1:
        raise ValueError('Labels must be a vector')
    if not (labels >= 0).all():
        raise ValueError('All labels must be >= 0')
    # Kind determines how labels in each window should be processed
    if not kind in {'counts', 'density', 'onehot', 'argmax', None}:
        raise ValueError('`kind` must be in {counts, density, onehot, argmax, None}')
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process labels with a sliding window.
    output = []
    for i in range(0, len(labels), frame_step):
        chunk = labels[i:i+frame_length]
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        # Just append the chunk if kind is None
        if kind == None:
            output.append(chunk)
            continue
        # Count the occurences of each label
        counts = np.bincount(chunk, minlength=max(labels))
        # Then process based on kind
        if kind == 'counts':
            output.append(counts)
        elif kind == 'density':
            output.append(counts / len(chunk))
        elif kind == 'onehot':
            one_hot = np.zeros(num_labels)
            one_hot[np.argmax(counts)] = 1
            output.append(one_hot)
        elif kind == 'argmax':
            output.append(np.argmax(counts))
    return np.array(output)


def windowed_signals(
    signals,
    frame_length,
    frame_step=None,
    pad_end=False
):
    """Generates signal segments of size frame_length"""
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process signals with a sliding window
    output = []
    for i in range(0, len(signals), frame_step):
        chunk = signals[i:i+frame_length]
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        output.append(chunk)
    return np.array(output)


def unfold_windows(arr, window_size, window_shift,
                   overlap_kind='mean'):
    '''

    Parameters
    ----------
    arr: np.array
        Either 2 or 3 dimensional
    window_size: int
    window_shift: int
    overlap_kind: str, optional
        What to do with possible overlapping areas. (default is 'sum')
        'sum' adds the values in the overlapping areas
        'mean' computes the mean of the overlapping areas

    Returns
    -------
    : np.arr
        2-dimensional array

    '''
    nseg = arr.shape[0]
    last_dim = arr.shape[-1]
    new_dim = (window_shift * nseg + window_size - window_shift, last_dim)
    buffer = np.zeros(new_dim)
    if overlap_kind == 'sum':
        for i in range(nseg):
            buffer[i*window_shift:i*window_shift+window_size] += arr[i]
        return buffer
    elif overlap_kind == 'mean':
        weights = np.zeros((new_dim[0],1))
        for i in range(nseg):
            buffer[i*window_shift:i*window_shift+window_size] += arr[i]
            weights[i*window_shift:i*window_shift+window_size] += 1.0
        return buffer/weights
    else:
        raise NotImplementedError(f'overlap_kind {overlap_kind}')


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


def load_dataset(dataset_paths, config):
    '''Loads the data and creates the features according to the config

    Parameters
    ----------
    dataset_paths: list of str
    config: src.config.Config

    Returns
    -------
    data: dict of (x,y) tuple
        x are the signal features
        y are the corresponding labels
    ground_truths: dict of np.array
        Due to feature creation, majority voting is applied, which reduces
        the number of labels in the training data. To allow proper testing,
        ground_truths contains the original labels for each sample without
        aggregation.

    '''
    ground_truths = {}
    data = {}
    # Read train data from csv files in configured directory
    for dataset_path in dataset_paths:
        subjects = {}
        print(f'Reading train data from {dataset_path}')
        for path in glob.glob(os.path.join(dataset_path, '*.csv')):
            subjects[os.path.basename(path)] = pd.read_csv(path)
        columns = config.SENSOR_COLUMNS
        # Grab data corresponding to column and compute features
        for subject, subject_data in subjects.items():
            print(f'Preprocessing: {subject}')
            x = subject_data[columns]
            y = subject_data[config.LABEL_COLUMN]
            # Replace classes with majority
            y = replace_classes(y, config.replace_classes)
            x = windowed_signals(
                x,
                config.SEQUENCE_LENGTH,
                config.FRAME_SHIFT
            )
            # Split original labels into subsets according to the frame_length
            # and frame_shift. This is later used for testing
            gt = windowed_labels(
                labels=y,
                num_labels=len(config.CLASSES),
                frame_length=config.SEQUENCE_LENGTH,
                frame_step=config.FRAME_SHIFT,
                pad_end=False,
                kind=None,
            ).reshape(-1)
            # Windowing and majority voting for training
            y = windowed_labels(
                labels=y,
                num_labels=len(config.CLASSES),
                frame_length=config.SEQUENCE_LENGTH,
                frame_step=config.FRAME_SHIFT,
                pad_end=False,
                kind='argmax',
            )
            # Generate features
            x = src.featurizer.Featurizer.get(config.FEATURES,
                                              x, columns,
                                              sample_rate=config.SAMPLE_RATE)
            # Tranform np array to series
            y = pd.Series(y)
            # Add to set of preprocessed data
            data[subject] = (x, y)
            ground_truths[subject] = gt
    return data, ground_truths
