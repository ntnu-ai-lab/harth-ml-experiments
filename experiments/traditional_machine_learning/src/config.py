import yaml
import os
import pandas as pd


class Config:

    def __init__(self, path):

        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        self.CLASSES = cfg['CLASSES']
        self.LABEL_COLUMN = cfg['LABEL_COLUMN']
        self.DROP_LABELS = cfg['DROP_LABELS'] if 'DROP_LABELS' in cfg else []
        self.INDEX_COLUMN = cfg['INDEX_COLUMN'] if 'INDEX_COLUMN' in cfg else None
        self.CUT_GROUND_TRUTHS = cfg['CUT_GROUND_TRUTHS'] if 'CUT_GROUND_TRUTHS' in cfg else True
        self.WANDB = cfg['WANDB'] if 'WANDB' in cfg else False
        self.SAMPLE_RATE = cfg['FREQUENCY']
        self.TRAIN_DATA = cfg['TRAIN_DATA']
        self.SEQUENCE_LENGTH = cfg['SEQUENCE_LENGTH']
        self.WINDOW_POSITION = cfg['WINDOW_POSITION'] if 'WINDOW_POSITION' in cfg else 0.5
        self.FRAME_SHIFT = cfg['FRAME_SHIFT']
        self.DEBUG = cfg['DEBUG']
        self.SUBJECT_GROUPS = cfg['SUBJECT_GROUPS']
        self.SCALE_DATA = cfg['SCALE_DATA']
        self.SAMPLER = cfg['SAMPLER'] if 'SAMPLER' in cfg else None
        self.CONFIG_PATH = os.path.dirname(os.path.realpath(path))  # Path of config file
        self.CONFIG_NAME = self.CONFIG_PATH.split('/')[-1]
        self.LIMITED_USERS = cfg['LIMITED_USERS'] if 'LIMITED_USERS' in cfg else None

        # Additional Features if given
        self.SUBJECT_FEATURES = cfg['SUBJECT_FEATURES'] if 'SUBJECT_FEATURES' in cfg else None
        self.TIME_FEATURES = cfg['TIME_FEATURES'] if 'TIME_FEATURES' in cfg else False
        self.TIME_FEATURES = ['hour', 'day', 'month'] if self.TIME_FEATURES==True else self.TIME_FEATURES
        self.CIRCADIAN_FEATURES = cfg['CIRCADIAN_FEATURES'] if 'CIRCADIAN_FEATURES' in cfg else None

        # Model
        self.ALGORITHM = cfg['ALGORITHM']
        self.ALGORITHM_ARGS = cfg['ALGORITHM_ARGS']
        self.DIM_REDUCTION = cfg['DIM_REDUCTION']  \
                if 'DIM_REDUCTION' in cfg else None

        # Cross validation
        self.CV_RANDOM = cfg.get('CV_RANDOM', 0)
        self.CV_METRIC = cfg['CV_METRIC']
        self.GS_NUM_TEST = cfg['GS_NUM_TEST']
        self.FOLDS = cfg['FOLDS']
        self.SKIP_FINISHED_ARGS = cfg['SKIP_FINISHED_ARGS']
        self.FORCE_NEW_CMAT_PATH = cfg['FORCE_NEW_CMAT_PATH'] if 'FORCE_NEW_CMAT_PATH' in cfg else False
        self.TRAIN_ON_FULL_DATASET = cfg['TRAIN_ON_FULL_DATASET']
        self.STORE_CMATS = cfg['STORE_CMATS'] if 'STORE_CMATS' in cfg else True

        # Inference
        self.INFERENCE_BATCH_SIZE = cfg['INFERENCE_BATCH_SIZE']
        self.PREDICTION_MODEL = cfg['PREDICTION_MODEL']
        # Whether to save model predictions after training (useful in LOO)
        self.save_predictions = cfg['SAVE_PREDICTIONS'] if 'SAVE_PREDICTIONS' in cfg else False

        # Which featurizer to use:
        if 'SENSORS' in cfg and 'FEATURES' not in cfg:
            # New featurizer
            self.OLD_FEATURIZER = False
            self.SENSORS = cfg['SENSORS']
            self.SENSOR_COLUMNS = [j for i in self.sensor_column_map.values() for j in i]
        else:
            # Old featurizer
            self.OLD_FEATURIZER = True
            self.FEATURES = cfg['FEATURES']
            self.SENSOR_COLUMNS = cfg['SENSOR_COLUMNS']

    @property
    def sensor_column_map(self):
        '''For each sensor, the corresponding columns

        e.g., {BackAcc:[back_x,back_y,back_z]}

        '''
        res = {}
        for sensor_type, sensor_setting in self.SENSORS.items():
            res.update(sensor_setting['COLUMNS'])
        return res

    @property
    def sensor_type_column_map(self):
        '''For each sensor type, the corresponding columns

        e.g., {'Acceleration': [back_x, back_y, back_z, thigh_x, thigh_y, thigh_z]}

        '''
        res = {}
        for sensor_type, sensor_setting in self.SENSORS.items():
            _cols = sensor_setting['COLUMNS'].values()
            res[sensor_type] = [j for i in _cols for j in i]
        return res

    @property
    def sensor_names(self):
        return [j for i in [x['COLUMNS'] for x in self.SENSORS.values()] for j in i]

    @property
    def num_sensors(self):
        '''Total amount of sensors'''
        return len(self.SENSOR_NAMES)

    @property
    def sensor_args(self):
        """
        Allow base args to be overrriden by sensor-specific args.
        """
        return {**self.ALGORITHM_ARGS}

    @property
    def all_columns(self):
        all_columns = set()
        for _, columns in self.sensors:
            all_columns |= set(columns)
        return sorted(all_columns)

    @property
    def all_classes(self):
        """
        Get all classes if present.
        """
        if self.CLASSES is None:
            raise Exception('Config file does not have "CLASSES" member')
        return self.CLASSES

    @property
    def classes(self):
        """
        Get all non-replaced classes.
        """
        return [c for c in self.all_classes if not 'replace' in c]

    @property
    def replace_classes(self):
        """
        Get replace dict for classes that have been dropped.
        """
        return {c['label']: c.get('replace', c['label']) for c in self.all_classes}

    @property
    def class_names(self):
        """
        Get all non-replaced class names.
        """
        return [c['name'] for c in self.classes]

    @property
    def class_labels(self):
        """
        Get all non-replaced class labels.
        """
        return [c['label'] for c in self.classes]

    @property
    def class_label_name_map(self):
        """
        Get all non-replaced classes as label to name dict
        """
        return {c['label']: c['name'] for c in self.classes}

    @property
    def class_name_label_map(self):
        """
        Get all non-replaced classes as name to label dict
        """
        return {c['name']: c['label'] for c in self.classes}

    @property
    def num_classes(self):
        """
        Total number of classes.
        """
        return len(self.classes)

    @property
    def subject_groups(self):
        '''If subjects to group in different subgroups'''
        if self.SUBJECT_GROUPS is None:
            return None
        return [x for x in self.SUBJECT_GROUPS.values()]

    @property
    def label_index(self):
        '''Index of each label defined in CLASSES'''
        return {c['label']: i for i, c in enumerate(self.classes)}

    @property
    def subject_features(self):
        '''Features of each subject if given as meta data csv file'''
        if (self.SUBJECT_FEATURES is None or
            self.SUBJECT_FEATURES['FEATURES'] is None):
            return {}
        subj_features = pd.read_csv(
            self.SUBJECT_FEATURES['PATH'],
            index_col=0
        )
        subj_features = subj_features[self.SUBJECT_FEATURES['FEATURES']]
        subj_features = subj_features.T.to_dict()
        return subj_features
