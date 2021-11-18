import os
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D, Conv1D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import MaxPool2D, MaxPool1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import add
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import LearningRateScheduler


class ANN:
    def __init__(self, algorithm, **args):
        if algorithm=='cnn':
            self.model = self.create_CNN(**args)
        elif algorithm=='lstm':
            self.model = self.create_Bi_LSTM(**args)
        elif algorithm=='inception_cnn':
            self.model = self.create_inception_CNN(**args)

    def create_Bi_LSTM(self, input_shape,
                       num_classes,
                       num_units,
                       dropout,
                       dense_size=512,
                       **args):
        self.num_classes = num_classes
        input_layer = Input(shape=input_shape)
        prev_layer = input_layer
        for n_units in num_units:
            forward_layer = LSTM(n_units, return_sequences=True)
            backward_layer = LSTM(n_units, #activation='relu',
                                  return_sequences=True,
                                  go_backwards=True)
            bidir = Bidirectional(
                            forward_layer,
                            backward_layer=backward_layer)(prev_layer)
            prev_layer = bidir
        flatten = Flatten()(prev_layer)
        fc1 = Dense(dense_size, activation='relu')(flatten)
        fc1 = Dense(dense_size, activation='relu')(fc1)
        dropout1 = Dropout(dropout)(fc1)
        prediction = Dense(num_classes, activation='softmax')(dropout1)
        # Create the final model:
        model = Model(inputs=input_layer,
                      outputs=prediction)
        # summarize model
        model.summary()
        # plot model architecture
        # plot_model(model, show_shapes=True, to_file='own_CNN.png')
        return model

    def create_CNN(self,
                   input_shape,
                   num_classes,
                   num_kernels,
                   kernel_sizes,
                   pooling=False,
                   dense_size=512,
                   dropout=0.1,
                   batch_normalization=False,
                   **args):
        '''A simple CNN with three dense layers at the end.
        
        Parameters
        ----------
        input_shape : tuple of int
        num_kernels : int
            How many kernels to use in one layer
        kernel_sizes : list of int
            For each layer, how large are the kernels
        pooling : bool
            Whether to use MaxPooling after each layer
        num_classes : int
            Output units
        dense_size : int
            Hidden dense layer size

        '''
        self.num_classes = num_classes
        input_layer = Input(shape=input_shape)
        prev_layer = input_layer
        for k_size in kernel_sizes:
            conv = Conv1D(num_kernels, k_size,
                          padding='same',
                          activation='relu')(prev_layer)
            prev_layer = conv
            if pooling:
                pool = MaxPool1D(pool_size=3,
                                 strides=1,
                                 padding='valid')(prev_layer)
                prev_layer = pool
            if batch_normalization:
                batch_n = BatchNormalization()(prev_layer)
                prev_layer = batch_n
            dropout0 = Dropout(dropout)(prev_layer)
            prev_layer = dropout0
        flatten = Flatten()(prev_layer)
        fc1 = Dense(dense_size, activation='relu')(flatten)
        fc1 = Dense(dense_size, activation='relu')(fc1)
        dropout1 = Dropout(dropout/2)(fc1)
        prediction = Dense(num_classes, activation='softmax')(dropout1)
        # Create the final model:
        model = Model(inputs=input_layer,
                      outputs=prediction)
        # summarize model
        model.summary()
        # plot model architecture
        # plot_model(model, show_shapes=True, to_file='own_CNN.png')
        return model

    def _inception_module(self, layer_in, n_filters,
                          kernel_sizes=[3,5,7,9],
                          pooling=False,
                          dropout=0.1,
                          batch_normalization=False):
        '''Create a multi-resolution module [Nafea et al. 2021]

        Parameters
        ----------
        kernel_sizes : list of int, optional
            For each block in the module, the kernel size for Conv1D
            (default is 3,5,7,9)
        '''
        conv_layers = []
        for kernel_size in kernel_sizes:
            conv = Conv1D(n_filters,
                          kernel_size,
                          padding='same',
                          activation='relu')(layer_in)
            conv_layers.append(conv)
        concat = concatenate(conv_layers)
        prev_layer = concat
        if pooling:
            pool1 = MaxPool1D(pool_size=3, strides=1,
                              padding='valid')(prev_layer)
            prev_layer = pool1
        if batch_normalization:
            batch_n = BatchNormalization()(prev_layer)
            prev_layer = batch_n
        dropout0 = Dropout(dropout)(prev_layer)
        return dropout0

    def create_inception_CNN(
                   self,
                   input_shape,
                   num_classes,
                   num_kernels=64,
                   num_layers=2,
                   kernel_sizes_in_module=[3,5,7,9],
                   pooling=False,
                   dense_size=512,
                   dropout=0.1,
                   batch_normalization=False,
                   **args):
        '''Create a CNN based on modules.

        Defaults are based on [Nafea et al. 2021]

        Parameters
        ----------
        input_shape : tuple of int, optional
            timesteps, features. (Default is (250, 6)-> 6 axes)
        kernel_sizes_in_module : list of int, optional
            How are the kernel sizes for the 1DConvs in module.
        dense_size : int, optional
            How many neurons in dense layer(default is 512)

        '''
        self.num_classes = num_classes
        input_layer = Input(shape=input_shape)
        prev_layer = input_layer
        for _ in range(num_layers):
            incept_module = self._inception_module(
                                                prev_layer,
                                                num_kernels,
                                                kernel_sizes_in_module,
                                                pooling,
                                                dropout,
                                                batch_normalization)
            prev_layer = incept_module
        flatten = Flatten()(prev_layer)
        fc1 = Dense(dense_size, activation='relu')(flatten)
        fc1 = Dense(dense_size, activation='relu')(fc1)
        dropout1 = Dropout(dropout/2)(fc1)
        prediction = Dense(num_classes, activation='softmax')(dropout1)
        # Create the final model:
        model = Model(inputs=input_layer,
                      outputs=prediction)
        # summarize model
        model.summary()
        return model


    def fit(self, train, valid, test,
            learning_rate, epochs, reduce_learning_rate=False,
            class_weights=None, early_stopping=False,
            gpu=0, log_path='', log_name='', **args):
        '''Train the model

        Parameters
        ----------
        train : tensorflow.keras.preprocessing.image.ImageDataGenerator
        valid : tensorflow.keras.preprocessing.image.ImageDataGenerator
        test : tensorflow.keras.preprocessing.image.ImageDataGenerator
        learning_rate : float
        epochs : int
        class_weights : np array
            Can be helpful for unbalanced data
        early_stopping : bool
        gpu : int or None, optional
            Which gpu to use if any (default is 0)
        log_path : str
            Where to create CSV logs having training, validation results
            per epoch
        log_name : str
            How to name the log file

        Returns
        -------
            None

        '''
        # Best model found so far will be saved:
        # checkpoint = ModelCheckpoint('model_best_weights_'+timestamp+'.h5',
        #                              monitor='val_loss', verbose=1,
        #                              save_best_only=True, mode='min')
        model = self.model
        # To save the current epoch state
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        csv_log = CSVLogger(log_path + '/' + log_name)
        callbacks = [csv_log]
        if early_stopping:
            # Will reduce learning rate if no improvement over
            # certain amount of epochs:
            reduce_lr = ReduceLROnPlateau(factor=0.1, verbose=1,
                                          mode='min', patience=8)
            # Early stopping
            early_stop = EarlyStopping(monitor='val_loss', patience=32)
            callbacks.append(reduce_lr)
            callbacks.append(early_stop)
        elif reduce_learning_rate:
            def scheduler(epoch, lr):
                '''Reduces lr by factor of 10 after 50 epochs'''
                if epoch != 50:
                    return lr
                else:
                    return lr * 1/10
            lr_scheduler = LearningRateScheduler(scheduler, verbose=1)
            callbacks.append(lr_scheduler)
        STEP_SIZE_TRAIN = len(train)
        STEP_SIZE_VALID = len(valid)
        # Optimizer:
        opt = tf.keras.optimizers.SGD(learning_rate=learning_rate,
                                      momentum=0.9)
        if gpu is not None:
            with tf.device('/device:GPU:' + str(gpu)):
                print('Compiling model...')
                f1score = tfa.metrics.F1Score(self.num_classes)
                recall = tf.keras.metrics.Recall()
                precision = tf.keras.metrics.Precision()
                model.compile(optimizer=opt,
                              loss='categorical_crossentropy',
                              metrics=['accuracy', f1score,
                                       recall, precision])
                print('train on: ', tf.test.gpu_device_name())
                # tf.keras.backend.set_learning_phase(1)
                print('Start fit...')
                model.fit(train,
                          steps_per_epoch=STEP_SIZE_TRAIN,
                          validation_data=valid,
                          validation_steps=STEP_SIZE_VALID,
                          epochs=epochs,
                          callbacks=callbacks,
                          initial_epoch=0,
                          shuffle=False,
                          class_weight=class_weights)
                # tf.keras.backend.set_learning_phase(0)
                print('Done!')
        
    def predict(self, x, inv_class_map, gpu, **args):
        '''Given model and input, prediction is returned'''
        model = self.model
        with tf.device('/device:GPU:' + str(gpu)):
            confidences = model.predict(x, **args)
        label_confidences = [max(x) for x in confidences]
        index_predictions = [np.argmax(x) for x in confidences]
        # Transform index to real label:
        label_predictions = [inv_class_map[x] for x in index_predictions]
        return label_predictions, label_confidences
