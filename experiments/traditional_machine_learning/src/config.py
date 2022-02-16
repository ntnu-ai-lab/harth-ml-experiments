import yaml
import os


class Config:

    def __init__(self, path):

        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)

        self.CLASSES = cfg['CLASSES']
        self.SENSOR_COLUMNS = cfg['SENSOR_COLUMNS']
        self.LABEL_COLUMN = cfg['LABEL_COLUMN']
        self.SAMPLE_RATE = cfg['FREQUENCY']
        self.TRAIN_DATA = cfg['TRAIN_DATA']
        self.SEQUENCE_LENGTH = cfg['SEQUENCE_LENGTH']
        self.FRAME_SHIFT = cfg['FRAME_SHIFT']
        self.FEATURES = cfg['FEATURES']
        self.DEBUG = cfg['DEBUG']
        self.SUBJECT_GROUPS = cfg['SUBJECT_GROUPS']
        self.SCALE_DATA = cfg['SCALE_DATA']
        self.CONFIG_PATH = os.path.dirname(os.path.realpath(path))  # Path of config file

        # Model
        self.ALGORITHM = cfg['ALGORITHM']
        self.ALGORITHM_ARGS = cfg['ALGORITHM_ARGS']

        # Cross validation
        self.CV_RANDOM = cfg.get('CV_RANDOM', 0)
        self.CV_METRIC = cfg['CV_METRIC']
        self.GS_NUM_TEST = cfg['GS_NUM_TEST']
        self.FOLDS = cfg['FOLDS']
        self.SKIP_FINISHED_ARGS = cfg['SKIP_FINISHED_ARGS']
        self.TRAIN_ON_FULL_DATASET = cfg['TRAIN_ON_FULL_DATASET']

        # Inference
        self.INFERENCE_BATCH_SIZE = cfg['INFERENCE_BATCH_SIZE']
        self.PREDICTION_MODEL = cfg['PREDICTION_MODEL']

    @property
    def sensor_args(self):
        """
        Allow base args to be overrriden by sensor-specific args.
        """
        return {**self.ALGORITHM_ARGS}

    def sensor_columns(self, sensor):
        """
        Get columns for sensor.
        """
        return self.SENSORS[sensor]['columns']

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
        return {c['label']: i for i, c in enumerate(self.CLASSES)}
