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

import wandb
import tempfile
import zipfile


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
    subjects = sorted(list(data))
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


def save_intermediate_cmat(path, filename, args, cmats,
                           valid_subjects=None):
    # Save cmat object and args in pickle file:
    if valid_subjects is None:
        args_cmats = [args, cmats]
    else:
        # In case subject filenames are provided:
        args_cmats = [args, cmats, valid_subjects]
    if not os.path.exists(path):
        os.makedirs(path)
    filehandler = open(path+filename, 'wb')
    pickle.dump(args_cmats, filehandler)
    filehandler.close()


def save_predictions(
    folder_path,
    filename,
    predictions,
    ground_truths,
    index
):
    '''If predictions and ground_truths need to be saved on disk

    Parameters
    ----------
    folder_path: str
    filename: str
    predictions: array like
    ground_truths: array like
        Needs to be the same size as predictions
    index: array like
        The index to use, needs to be the same size as predictions

    '''
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    pd.DataFrame(
        zip(ground_truths,predictions),
        columns=['label', 'prediction'],
        index=index
    ).to_csv(os.path.join(folder_path, filename))
    print(f'Saved predictions: {os.path.join(folder_path, filename)}')


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
    if pad_end:
        return output
    else:
        return np.array(output)


def windowed_timestamps(
    signals,
    frame_length,
    frame_step=None,
    pad_end=False,
    kind=None
):
    """Generates timestamp segments of size frame_length

    With kind=None we are able to split the given timestamp
    array into batches.

    """
    # Kind determines how labels in each window should be processed
    if not kind in {'min', 'max', 'center', None}:
        raise ValueError('`kind` must be in {min, max, center, None}')
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process signals with a sliding window
    output = []
    for i in range(0, len(signals), frame_step):
        chunk = signals[i:i+frame_length]
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        if kind=='center':
            chunk = chunk[len(chunk)//2]
        elif kind=='min':
            chunk = min(chunk)
        elif kind=='max':
            chunk = max(chunk)
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
        'last' takes values of the last overlapping window

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
    elif overlap_kind == 'last':
        for i in range(nseg):
            buffer[i*window_shift:i*window_shift+window_size] = arr[i]
        return buffer

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



def read_dataframes(config, dataset_path):
    print(f'Reading train data from {dataset_path}')
    subjects = {}
    paths = glob.glob(os.path.join(dataset_path, '*.csv'))
    for path in paths:
        fname = os.path.basename(path)
        if config.LIMITED_USERS is not None and fname not in config.LIMITED_USERS:
            continue
        if config.INDEX_COLUMN is not None:
            _df = pd.read_csv(
                path,
                index_col=config.INDEX_COLUMN
            )
        else:
            _df = pd.read_csv(path)
        for drop_label in config.DROP_LABELS:
            _df = _df[_df[config.LABEL_COLUMN]!=drop_label]
        subjects[fname] = _df
    return subjects


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
    valid_x_lasts: dict of np.array or None
        If cutting through windowing happens, the last window of the
        signal is provided here

    '''
    ground_truths = {}
    data = {}
    valid_x_lasts = {}
    # Read train data from csv files in configured directory
    for dataset_path in dataset_paths:
        subjects = read_dataframes(config, dataset_path)
        columns = config.SENSOR_COLUMNS.copy()
        seq_lens = [config.SEQUENCE_LENGTH] if type(config.SEQUENCE_LENGTH)==int \
            else config.SEQUENCE_LENGTH
        assert len(seq_lens)==1 or (len(seq_lens)>1 and config.WINDOW_POSITION), \
            'When more than 1 SEQUENCE_LENGTH given, WINDOW_POSITION must be defined'
        assert len(seq_lens)==1 or seq_lens==sorted(seq_lens), \
            'SEQUENCE_LENGTH must be sorted with the smallest being the first'

        # Grab data corresponding to column and compute features
        for subject, subject_data in subjects.items():
            print(f'Preprocessing: {subject}')
            x = subject_data[columns]
            y = subject_data[config.LABEL_COLUMN]
            # Replace classes with majority
            y = replace_classes(y, config.replace_classes)
            #######################################
            xs = []
            for seq_len in seq_lens:
                pad_start = int(np.ceil((1-config.WINDOW_POSITION)*(seq_len-min(seq_lens))))
                pad_end = seq_len-min(seq_lens)-pad_start
                _x = windowed_signals(
                    np.concatenate([np.zeros([pad_start,x.shape[1]]),
                                    x.values,
                                    np.zeros([pad_end,x.shape[1]])]),
                    seq_len,
                    config.FRAME_SHIFT
                )
                # Generate features
                _x = src.featurizer.Featurizer.get(
                    _x,
                    config,
                    config.subject_features[subject] if subject in config.subject_features else {}
                )
                _x.columns = _x.columns+'_'+str(seq_len)
                xs.append(_x)
            # Timestamp related features
            if config.TIME_FEATURES:
                _ts = windowed_timestamps(
                    subject_data.index,
                    seq_lens[0],
                    config.FRAME_SHIFT,
                    kind='center'
                )
                _ts = src.featurizer.compute_time_features(
                    ts_arr=_ts,
                    kinds=config.TIME_FEATURES
                )
                xs.append(_ts)
            # Circadian features if needed:
            if config.CIRCADIAN_FEATURES:
                _cf = src.featurizer.compute_circ_features(
                    model_path=config.CIRCADIAN_FEATURES,
                    signal=subject_data,
                    subject=subject,
                    config=config
                )
                _cf = windowed_timestamps(
                    _cf.values,
                    seq_lens[0],
                    config.FRAME_SHIFT,
                    kind='center'
                )
                _cf = pd.DataFrame(_cf, columns=['cf'])
                xs.append(_cf)
            x = pd.concat(xs, axis=1)
            #######################################
            # Split original labels into subsets according to the frame_length
            # and frame_shift. This is later used for testing
            original_len = subject_data[columns].shape[0]
            new_len = x.shape[0]
            unfolded_new_len = config.FRAME_SHIFT*new_len + seq_lens[0] - config.FRAME_SHIFT
            num_padding_req = original_len - unfolded_new_len
            valid_x_last = pd.Series([])
            if config.CUT_GROUND_TRUTHS:
                # In case ground truths shall be cutted when windowing
                # does not fit exactly the signal length:
                gt = y.iloc[:unfolded_new_len]
            else:
                gt = y.copy()
                if num_padding_req != 0:
                    # generate last window to predict
                    #######################################
                    valid_xs_last = []
                    for seq_len in seq_lens:
                        pad_start = int(np.ceil((1-config.WINDOW_POSITION)*(seq_len-min(seq_lens))))
                        pad_end = seq_len-min(seq_lens)-pad_start
                        last_window = subject_data[columns].iloc[
                            -(seq_lens[0]+pad_start):
                        ]
                        _valid_x_last = windowed_signals(
                            np.concatenate([last_window.values,
                                            np.zeros([pad_end,last_window.shape[1]])]),
                            seq_len,
                            config.FRAME_SHIFT
                        )
                        _valid_x_last = src.featurizer.Featurizer.get(
                            _valid_x_last,
                            config,
                            config.subject_features[subject] if subject in config.subject_features else {}
                        )
                        _valid_x_last.columns = _valid_x_last.columns+'_'+str(seq_len)
                        valid_xs_last.append(_valid_x_last)
                    # Timestamp related features
                    if config.TIME_FEATURES:
                        _ts_last = windowed_timestamps(
                            subject_data[columns].iloc[-seq_lens[0]:].index,
                            seq_lens[0],
                            config.FRAME_SHIFT,
                            kind='center'
                        )
                        _ts_last = src.featurizer.compute_time_features(
                            ts_arr=_ts_last,
                            kinds=config.TIME_FEATURES
                        )
                        valid_xs_last.append(_ts_last)
                    # Circadian features if needed:
                    if config.CIRCADIAN_FEATURES:
                        _cf_last = src.featurizer.compute_circ_features(
                            model_path=config.CIRCADIAN_FEATURES,
                            signal=subject_data[columns].iloc[-seq_lens[0]:],
                            subject=subject,
                            config=config
                        )
                        _cf_last = windowed_timestamps(
                            _cf_last.values,
                            seq_lens[0],
                            config.FRAME_SHIFT,
                            kind='center'
                        )
                        _cf_last = pd.DataFrame(_cf_last, columns=['cf'])
                        valid_xs_last.append(_cf_last)
                    valid_x_last = pd.concat(valid_xs_last, axis=1)
                    if valid_x_last.shape[0]>1: breakpoint()
            # Windowing and majority voting for training
            y = windowed_labels(
                labels=y,
                num_labels=len(config.CLASSES),
                frame_length=seq_lens[0],
                frame_step=config.FRAME_SHIFT,
                pad_end=False,
                kind='argmax',
            )
            # Tranform np array to series
            y = pd.Series(y)
            # Add to set of preprocessed data
            data[subject] = (x, y)
            ground_truths[subject] = gt
            valid_x_lasts[subject] = valid_x_last
    return data, ground_truths, valid_x_lasts

def wandb_init(run_name, wandb_config, entity, proj_name):
    wandb.login(key='<WANDB_KEY>')
    wandb.init(name=run_name,
               project=proj_name,
               entity=entity,
               config=wandb_config)


def log_wandb(y_pred, y_true, params_config, log_name):
    '''Logging in weights and biases

    Parameters
    ----------
    y_pred : array like
    y_true : array like
    params_config : src.config.Config
    log_name : str
        Name shown in panel in wandb

    '''
    log_wandb_cmat(
        y_pred,
        y_true,
        label_names=params_config.class_names,
        log_name=log_name,
        label_mapping=params_config.label_index
    )
    log_metrics(
        y_pred=y_pred,
        y_true=y_true,
        labels=params_config.class_labels,
        names=params_config.class_names,
        log_name=log_name,
        metrics=['f1score',
                 'average_f1score',
                 'recall',
                 'average_recall',
                 'precision',
                 'average_precision',
                 'accuracy']
    )



def log_wandb_cmat(
    y_pred,
    y_true,
    label_names,
    log_name,
    label_mapping=None
):
    '''Logs confusion matrix in wandb

    Parameters
    ----------
    y_pred: array like
    y_true: array like
    label_names: list of str
        for each index a name
    log_name: str
    label_mapping: dict or list, optional
        If the given labels in y_true/y_pred have to be renamed

    '''
    if label_mapping is not None:
        y_pred = [label_mapping[y] for y in y_pred]
        y_true = [label_mapping[y] for y in y_true]
    cmat_name = 'cmat_' + log_name
    wandb.log({cmat_name: wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_true, preds=y_pred,
                    class_names=label_names,
                    title=cmat_name)})



def log_metrics(y_pred, y_true, labels, names,
                log_name, metrics=['average_f1score']):
    '''Logs different given metrics in wandb

    Parameters
    ----------
    y_pred: array like
    y_true: array like
    labels: list
        List of values that might occur in y_true/y_pred
    names: list
        List of names corresponding to labels
    log_name: str
    metrics: list of str, optional
        Which metrics to log
    label_mapping: dict or list, optional
        If the given labels in y_true/y_pred have to be renamed

    '''
    allowed_metrics = ['f1score',
                       'average_f1score',
                       'recall',
                       'average_recall',
                       'precision',
                       'average_precision',
                       'accuracy']
    assert set(metrics)<=set(allowed_metrics), print(f'{allowed_metrics}')
    cm = cmat.ConfusionMatrix.create(
      y_true = y_true,
      y_pred = y_pred,
      labels = labels,
      names = names
    )
    for metric in metrics:
        if metric in ['average_f1score',
                      'average_precision',
                      'average_recall',
                      'accuracy']:
            wandb.log({f'{metric}_{log_name}': getattr(cm, metric)})
        else:
            data = [(l,m) for l,m in getattr(cm, metric).items()]
            table = wandb.Table(data=data, columns=['label', metric])
            wandb.log({f'{metric}_{log_name}': wandb.plot.bar(
                table,
                'label',
                metric,
                title=f'bar_{metric}_{log_name}')})


def log_wandb_class_distribution(y_true, label_name_dict=None):
    '''Stores amount of samples for each activity in the dataset

    Parameters
    ----------
    y_true: array like
    label_name_dict: in case for renaming labels

    '''
    counts = pd.DataFrame(y_true).value_counts()
    if label_name_dict is None:
        data = [(a[0],c) for a,c in counts.items()]
    else:
        data = [(label_name_dict[a[0]],c) for a,c in counts.items()]
    table = wandb.Table(data=data,columns=['label','count'])
    wandb.log({f'class_distribution_{len(counts)}': wandb.plot.bar(
        table,
        'label',
        'count',
        title=f'Class Distribution for {len(counts)} classes')}
    )


import plotly.express as px
def log_wandb_class_sample_width(y_true, label_name_dict=None):
    '''Window size in samples for each class in the dataset

    Parameters
    ----------
    y_true: array like
    label_name_dict: in case for renaming labels

    '''
    counts = pd.DataFrame(y_true).value_counts()
    flip_indices = np.where(np.roll(y_true,1)!=y_true)[0]
    amount_values = np.diff(flip_indices.tolist() + [len(y_true)])
    if label_name_dict is None:
        flip_values = [y_true[x] for x in flip_indices]
    else:
        flip_values = [label_name_dict[y_true[x]] for x in flip_indices]
    df =  pd.DataFrame({'label': flip_values, 'window_size':amount_values})
    fig = px.box(df, x="label", y="window_size")
    wandb.log({f'window_size_per_label_{len(counts)}': wandb.data_types.Plotly(fig)})
