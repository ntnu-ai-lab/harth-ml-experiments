import src.data_generator as adg
import src.utils
import glob
import cmat
import pandas as pd
import pickle
import argparse
import os
import src.config

def train(config, dataset_path=None):
    # Load dataset path
    if dataset_path is None:
        dataset_path = config.TRAIN_DATA
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f'Dataset {dataset_path} does not exist')

    # For tensorflow GPU training (set device to use)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config.GPU)
    # After setting device, tensorflow can be loaded:
    from src.model import ANN

    # Read train data from csv files in configured direrctory
    print(f'Reading train data from {dataset_path}')
    subjects = {}
    for path in glob.glob(os.path.join(dataset_path, '*.csv')):
        subjects[os.path.basename(path)] = path

    columns = config.BACK_COLUMNS + config.THIGH_COLUMNS
    input_shape = (config.SEQUENCE_LENGTH, len(columns))
    subject_groups = config.subject_groups
    if config.CLASS_WEIGHT:
        # Create class weights to mitigate unbalanced label influence
        class_weights = src.utils.get_class_weights(subjects.values(),
                                                    config.LABEL_COLUMN,
                                                    config.class_map)
    else:
        class_weights = None


    #### Check existing arguments ####
    seconds = config.SEQUENCE_LENGTH//config.SAMPLE_RATE
    if config.SKIP_FINISHED_ARGS:
        cmat_path = config.CONFIG_PATH + 'cmats_' + str(seconds) + 'sec/'
        existing_arguments = []
        if os.path.exists(cmat_path):
            for cmat_file in os.listdir(cmat_path):
                if 'best_idx' not in cmat_file:
                    with open(cmat_path+cmat_file, 'rb') as f:
                        existing_arguments.append(pickle.load(f)[0])
    ##################################

    print('Perform a GridSearch and CrossValidation')
    grid_cms = []
    grid_args = []
    hyperparameters = config.ALGORITHM_ARGS
    for i, args in enumerate(src.utils.grid_search(hyperparameters)):
        if config.SKIP_FINISHED_ARGS and args in existing_arguments:
                print(f'Skipping existing arguments: {args}', flush=True)
                continue
        print(f'Evaluating arguments {i}: {args}', flush=True)
        cv_cms = []
        for j, train, test, valid in src.utils.get_multiset_cv_split(
                                                        subjects,
                                                        subject_groups,
                                                        config.GS_NUM_TEST,
                                                        config.GS_NUM_VALID,
                                                        config.CV_RANDOM):
            print(f'Fold {j}: valid={valid} test={test}')
            model = ANN(config.ALGORITHM,
                        input_shape=input_shape,
                        num_classes=config.num_outputs,
                        **args)
            train_paths = [dataset_path+'/'+x for x in train]
            # If no subjects in validation set, it is equal to testset
            if config.GS_NUM_VALID == 0:
                valid_paths = [dataset_path+'/'+x for x in test]
            else:
                valid_paths = [dataset_path+'/'+x for x in valid]
            test_paths = [dataset_path+'/'+x for x in test]
            # Creating the generator objects:
            # Train subset
            train_dg = adg.AccelerometerDataGenerator(
                                train_paths,
                                columns=columns,
                                label_column=config.LABEL_COLUMN,
                                sequence_length=config.SEQUENCE_LENGTH,
                                overlapping=config.OVERLAP,
                                padding_value=0,
                                batch_size=config.BATCH_SIZE,
                                class_map=config.class_map,
                                replace_class_map=config.replace_class_map,
                                inv_class_map=config.inv_class_map,
                                use_fft=False)
            # Test subset
            test_dg = adg.AccelerometerDataGenerator(
                                test_paths,
                                columns=columns,
                                label_column=config.LABEL_COLUMN,
                                sequence_length=config.SEQUENCE_LENGTH,
                                overlapping=config.OVERLAP,
                                padding_value=0,
                                batch_size=1,
                                class_map=config.class_map,
                                replace_class_map=config.replace_class_map,
                                inv_class_map=config.inv_class_map,
                                use_fft=False)

            # Validation subset
            valid_dg = adg.AccelerometerDataGenerator(
                                valid_paths,
                                columns=columns,
                                label_column=config.LABEL_COLUMN,
                                sequence_length=config.SEQUENCE_LENGTH,
                                overlapping=config.OVERLAP,
                                padding_value=0,
                                batch_size=config.BATCH_SIZE,
                                class_map=config.class_map,
                                replace_class_map=config.replace_class_map,
                                inv_class_map=config.inv_class_map,
                                use_fft=False)
            # Fitting the model
            # To save csv files containing the learning curves of the model
            log_name = 'arg'+str(i).zfill(4)+'_fold'+str(j).zfill(4)+'.csv'
            model.fit(train_dg, valid_dg, test_dg,
                      gpu=0,
                      class_weights=class_weights,
                      log_path=config.CONFIG_PATH+'training_curves_'+str(seconds)+'sec',
                      log_name=log_name,
                      **args)
            valid_y = test_dg.get_real_labels()
            valid_y_hat = model.predict(test_dg,
                                        config.inv_class_map,
                                        gpu=0,
                                        verbose=1)[0]
            # Collect statistics and store result for given metric
            cm = cmat.create(valid_y, valid_y_hat,
                             config.used_class_labels,
                             config.used_class_names)
            cv_cms.append(cm)
        # Save cmat object and args in pickle file:
        src.utils.save_intermediate_cmat(cmat_path,
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
    src.utils.save_intermediate_cmat(config.CONFIG_PATH+'cmats/',
                                     'best_args.pkl',
                                     best_args,
                                     grid_cms[best_idx])
    print(f'Best avg {config.CV_METRIC}: idx={best_idx} score={averages[best_idx]: .3f} args={best_args}')
    print(f'{config.CV_METRIC} for all grid iterations and folds:')
    print(grid_scores)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start DL training.')
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
