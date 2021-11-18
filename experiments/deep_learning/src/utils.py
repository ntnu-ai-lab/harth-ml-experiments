import numpy as np
import pandas as pd
import sklearn
import sklearn.model_selection
import os
import pickle


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

def save_intermediate_cmat(path, filename, args, cmats):
    # Save cmat object and args in pickle file:
    args_cmats = [args, cmats]
    if not os.path.exists(path):
        os.makedirs(path)
    filehandler = open(path+filename, 'wb')
    pickle.dump(args_cmats, filehandler)
    filehandler.close()
    print('Saved cmat: ', path+filename)


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
        yield fold, fold_tr, fold_tst, fold_val


def get_class_weights(data_paths, label_column='label', class_map=None):
    '''Calculates the class weights depending on the label distribution

    Returns
    -------
    Dict with index of class : weight of class
    '''
    all_dfs = pd.concat([pd.read_csv(sp) for sp in data_paths])
    all_classes = all_dfs.sort_values(label_column)[label_column].values
    # Replace not used classes:
    if class_map is not None:
        all_classes = np.array([class_map[x] for x in all_classes])
    all_classes.sort()
    unique_classes = np.unique(all_classes)
    # Compute weights
    weights = sklearn.utils.class_weight.compute_class_weight('balanced',
                                                classes=unique_classes,
                                                y=all_classes)
    return dict(zip(unique_classes, weights))


def save_df(df, path, name):
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(path+'/'+name,index=False)
