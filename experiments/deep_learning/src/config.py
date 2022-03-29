import os
import yaml


class Config():
    def __init__(self, path):
        with open(path, 'r') as f:
            cfg = yaml.load(f, Loader=yaml.Loader)
        self.CLASSES = cfg['CLASSES']
        self.LABEL_COLUMN = cfg['LABEL_COLUMN']
        self.SAMPLE_RATE = cfg['FREQUENCY']
        self.TRAIN_DATA = cfg['TRAIN_DATA']
        self.SEQUENCE_LENGTH = cfg['SEQUENCE_LENGTH']
        self.OVERLAP = cfg['OVERLAP']
        self.GPU = cfg['GPU']
        self.BACK_COLUMNS = cfg['BACK_COLUMNS']
        self.THIGH_COLUMNS = cfg['THIGH_COLUMNS']
        self.LABEL_COLUMN = cfg['LABEL_COLUMN']
        self.ALGORITHM = cfg['ALGORITHM']
        # Train parameter
        self.BATCH_SIZE = cfg['BATCH_SIZE']
        self.CLASS_WEIGHT = cfg['CLASS_WEIGHT']
        # Hyperopt and CV
        self.CV_RANDOM = cfg['CV_RANDOM']
        self.GS_NUM_VALID = cfg['GS_NUM_VALID']
        self.GS_NUM_TEST = cfg['GS_NUM_TEST']
        self.SUBJECT_GROUPS = cfg['SUBJECT_GROUPS']
        self.CV_METRIC = cfg['CV_METRIC']
        self.ALGORITHM_ARGS = cfg['ALGORITHM_ARGS']
        self.SKIP_FINISHED_ARGS = cfg['SKIP_FINISHED_ARGS']
        # Path of config file
        self.CONFIG_PATH = os.path.dirname(os.path.realpath(path))

    @property
    def non_replaced_classes(self):
        # Returns all classes that are not replaced
        return [c for c in self.CLASSES if not 'replace' in c]

    @property
    def replaced_classes(self):
        # Returns all replaced classes
        return [c for c in self.CLASSES if 'replace' in c]

    @property
    def num_outputs(self):
        # Returns the number of non-replaced classes
        return len(self.non_replaced_classes)

    @property
    def replace_class_map(self):
        '''To replace the given label'''
        return {c['label']: c.get('replace', c['label']) for c in self.CLASSES}


    @property
    def class_map(self):
        # Create one-hot encoding
        # Changing CLASSES will lead to bad results!
        class_map = {c['label']: i for i, c in enumerate(self.non_replaced_classes)}
        # Apply class replacement
        for c in self.replaced_classes:
            class_map[c['label']] = class_map[c['replace']]
        return class_map

    @property
    def inv_class_map(self):
        # Reverse one-hot encoding
        return {i: c['label'] for i, c in enumerate(self.non_replaced_classes)}

    @property
    def class_names(self):
        final_dict = {}
        for cc in self.non_replaced_classes: 
            final_dict[cc['label']] = cc['name']
        for cc in self.replaced_classes:
            final_dict[cc['label']] = final_dict[cc['replace']]
        return final_dict

    @property
    def used_class_names(self):
        '''List of class names actually used'''
        return [self.class_names[x] for x in self.inv_class_map.values()]

    @property
    def used_class_labels(self):
        '''List of class labels actually used'''
        return list(self.inv_class_map.values())

    @property
    def subject_groups(self):
        '''If subjects to group in different subgroups'''
        return [x for x in self.SUBJECT_GROUPS.values()]
    # ############################################################
