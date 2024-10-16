import argparse
import train
import src.config


def loso_cv(config, dataset_path=None):
    '''Starts a leave-one-out cross validation'''
    config.FOLDS = 0  # Enable loo split
    # In case sequence length and frame shift lead to cutting of
    # last part of the signal, shall the last part stay cutted or not
    # False is helpful if working with different window lengths.
    # Makes them comparable
    config.CUT_GROUND_TRUTHS = False
    # If logging in wandb required
    config.WANDB = False
    # For each subject, save predictions after training
    config.save_predictions = False
    train.train(config, dataset_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Start LOSO CV.')
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
    loso_cv(config, ds_path)
