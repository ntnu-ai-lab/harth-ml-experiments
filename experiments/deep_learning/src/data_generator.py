import os
import math
import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import Counter
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence


class AccelerometerDataGenerator(Sequence):
    def __init__(self, subject_filepaths,
                 class_map,
                 replace_class_map,
                 inv_class_map=None,
                 columns=['back_x', 'back_y', 'back_z',
                          'thigh_x', 'thigh_y', 'thigh_z'],
                 label_column='label',
                 sequence_length=250,
                 overlapping=.5,
                 padding_value=0,
                 batch_size=128,
                 use_fft=False):
        self.subject_filepaths = subject_filepaths
        self.class_map = class_map
        self.replace_class_map = replace_class_map
        self.inv_class_map = inv_class_map
        self.columns = columns
        self.label_column = label_column
        self.sequence_length = sequence_length
        self.overlapping = overlapping
        self.padding_value = padding_value
        self.batch_size = batch_size
        self.total_length = 0
        self.train_x_batches = None
        self.train_y_batches = None
        self.create_batches(use_fft)

    def create_batches(self, use_fft):
        self.data_dict = {}
        self.label_dict = {}
        print('Loading csvs...')
        counts_under = []
        total_windows = []
        for subject_filepath in self.subject_filepaths:
            print(subject_filepath)
            df = pd.read_csv(subject_filepath)
            for i, column in enumerate(self.columns):
                raw_values = df[column].values
                labels = df[self.label_column].values
                labels = np.array([self.replace_class_map[l] for l in labels])
                # breakpoint()
                windowed = self.sliding_window(raw_values,
                                          labels,
                                          self.padding_value)
                windowed_values, windowed_labels = windowed
                if column not in self.data_dict:
                    self.data_dict[column] = windowed_values
                    self.label_dict[column] = windowed_labels
                else:
                    self.data_dict[column] += windowed_values
                    self.label_dict[column] += windowed_labels
                if i == 0:
                    self.total_length += len(windowed_values)
        all_columns_to_concat = []
        for cols, vals in self.data_dict.items():
            arr_vals = np.array(vals).reshape(-1,
                                              self.sequence_length,
                                              1)
            all_columns_to_concat.append(arr_vals)
        train_x_batches = np.concatenate(all_columns_to_concat,
                                         axis=2)
        train_y_batches = np.array(self.label_dict[self.columns[0]])
        train_y_batches = self.encode_labels(train_y_batches)
        cw = 0        
        self.train_x_batches = []
        self.train_y_batches = []
        for i in range(math.ceil(self.total_length/self.batch_size)):
            batch_x = train_x_batches[cw:cw+self.batch_size]
            batch_y = train_y_batches[cw:cw+self.batch_size]
            self.train_x_batches.append(batch_x)
            self.train_y_batches.append(batch_y)
            cw += self.batch_size


    def sliding_window(self, array, labels, padding_value=0):
        '''Creates a sliding window arrays of the given array

        Parameters
        ----------
        array : np.array
        labels : np.array
        padding_value : int or NaN, optional
            To ensure same window size, padding is required
            (default is 0)

        Returns
        -------
        : list, list

        '''
        # TODO: maybe remove the first if too much padding
        # Input array:
        array_ext = np.full(self.sequence_length-1, padding_value)
        array_ext = np.concatenate((array_ext, array))
        strided = np.lib.stride_tricks.as_strided
        windows = strided(array_ext,
                          shape=(array.shape[0], self.sequence_length),
                          strides=array_ext.strides*2)
        slider = math.floor(self.sequence_length*(1-self.overlapping))
        windows = windows[0::slider]
        # Label array (padding value is -1):
        label_ext = np.full(self.sequence_length-1, -1)
        label_ext = np.concatenate((label_ext, labels))
        label_windows = strided(label_ext,
                                shape=(labels.shape[0],
                                       self.sequence_length),
                                strides=label_ext.strides*2)
        label_windows = label_windows[0::slider]
        # The padded window is removed if necessary
        if len(array)%self.sequence_length != 0:
            windows = windows[1:]
            label_windows = label_windows[1:]
        major_labels = []
        # Majority voting for each window (ignore -1)
        for label_window in label_windows:
            lw = list(label_window)
            if -1 in lw:
                breakpoint()
                print('-1 still in labels!')
            # lw = [x for x in lw if x!=-1]
            major_label = Counter(list(lw)).most_common()[0][0]
            major_labels.append(major_label)
        return list(windows), major_labels

    def encode_labels(self, labels):
        """
        Encode as one-hot vectors
        """
        # Make sure that all labels are accounted for by the class map
        if not all([l in self.class_map for l in labels]):
            print('Unexpected labels found when encoding labels!')
        # Replace by one hot index using the provided class map
        labels = [self.class_map[l] for l in labels]
        # One hot encode
        one_hot = np.zeros((len(labels),
                            max(self.class_map.values()) + 1),
                           dtype=np.float32)
        one_hot[range(len(labels)), labels] = 1.0
        return one_hot

    def get_real_labels(self):
        '''Returns the real labels, specified in the config.yml'''
        return [self.inv_class_map[np.argmax(x)] for x in self.train_y_batches]

    def __len__(self) :
        '''Returning the number of created images

        Here all subjects are summed up and divided by
        the batch size afterward.
        '''
        return math.ceil(self.total_length/self.batch_size)

    def __getitem__(self, idx):
        train_x = self.train_x_batches[idx]
        train_y = self.train_y_batches[idx]
        return (train_x, train_y)
