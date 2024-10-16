import functools
import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats

import os
import src.utils
from datetime import datetime
import zipfile
import tempfile


def cached(f):
    """
    Decorator that creates a cached property.
    """
    key = f.__name__

    @property
    @functools.wraps(f)
    def decorated(self):
        if key not in self._cache:
            self._cache[key] = f(self)
        return self._cache[key]

    return decorated


def feature(f):
    """
    Feature decorator that wraps results in a DataFrame/Series if not already.
    """

    @cached
    @functools.wraps(f)
    def decorated(self):
        res = f(self)
        if isinstance(res, (pd.DataFrame, pd.Series)):
            pass
        elif len(res.shape) == 2:
            res = pd.DataFrame(res, columns=[f'{f.__name__}_{an}' for an in self.axes_names])
        else:
            res = pd.Series(res, name=f.__name__)
        return res

    return decorated


def hour_sin(series):
    '''Sine transform on timestamps for 24h-based features'''
    return np.sin((series.dt.hour+series.dt.minute/60)*(2.*np.pi/24))

def hour_cos(series):
    '''Cosine transform on timestamps for 24h-based features'''
    return np.cos((series.dt.hour+series.dt.minute/60)*(2.*np.pi/24))

def day_sin(series):
    '''Sine transform on timestamps for 7day-based features'''
    return np.sin((series.dt.day+series.dt.hour/24)*(2.*np.pi/7))

def day_cos(series):
    '''Cosine transform on timestamps for 7day-based features'''
    return np.cos((series.dt.day+series.dt.hour/24)*(2.*np.pi/7))

def month_sin(series):
    '''Sine transform on timestamps for 12month-based features'''
    return np.sin((series.dt.month+series.dt.day/30)*(2.*np.pi/12))

def month_cos(series):
    '''Cosine transform on timestamps for 12month-based features'''
    return np.cos((series.dt.month+series.dt.day/30)*(2.*np.pi/12))


def compute_time_features(ts_arr, kinds=['hour', 'day', 'month']):
    '''Trigonometric features representing timestamps'''
    for kind in kinds:
        if kind not in ['hour', 'day', 'month']:
            raise ValueError(f"{kind} not allowed time feature:'hour','day','month'")
    to_dt = np.vectorize(lambda x: datetime.strptime(x[:26],'%Y-%m-%d %H:%M:%S.%f'))
    s = pd.Series(to_dt(ts_arr))
    res = pd.DataFrame()
    if 'hour' in kinds:
        res['hour_sin'] = hour_sin(s)
        res['hour_cos'] = hour_cos(s)
    if 'day' in kinds:
        res['day_sin'] = day_sin(s)
        res['day_cos'] = day_cos(s)
    if 'month' in kinds:
        res['month_sin'] = month_sin(s)
        res['month_cos'] = month_cos(s)
    return res

def aggregate_labels(labels, frame_length, frame_step=None, pad_end=False):
    """Segmenting a label array

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
    if not labels.ndim == 1:
        raise ValueError('Labels must be a vector')
    # Let frame_step default to one full frame_length
    frame_step = frame_length if frame_step is None else frame_step
    # Process labels with a sliding window.
    output = []
    for i in range(0, len(labels), frame_step):
        chunk = labels[i:i+frame_length]
        # Ignore incomplete end chunk unless padding is enabled
        if len(chunk) < frame_length and not pad_end:
            continue
        # Count the occurences of each label
        counts = np.bincount(chunk, minlength=max(labels))
        output.append(np.argmax(counts))
    return np.array(output)

def get_abs_timedifference(ts1, ts2):
    '''Absolute time difference between 2 timestamps'''
    if ts1>=ts2:
        return ts1-ts2
    else:
        return ts2-ts1

def get_midpoint_sleep(onset_ts, offset_ts):
    '''Computes midpoint sleep based on sleep onset and offset'''
    if type(onset_ts) != datetime:
        onset_ts = datetime.strptime(onset_ts[:26], '%Y-%m-%d %H:%M:%S.%f')
    if type(offset_ts) != datetime:
        offset_ts = datetime.strptime(offset_ts[:26], '%Y-%m-%d %H:%M:%S.%f')
    return onset_ts+(offset_ts-onset_ts)/2

def get_sleep_on_and_offset_indices(arr, sleep_threshold=20, sleep_label=82, idx_dist_th=120):
    '''Computes all sleep on and offset indices in the given label array'''
    indices = get_contin_label_idxs(arr, sleep_threshold, sleep_label)
    onsets = [indices[0][0]]
    offsets = []
    for i in range(len(indices)-1):
        if indices[i+1][0] - indices[i][1] > idx_dist_th:
            offsets.append(indices[i][1])
            onsets.append(indices[i+1][0])
    offsets.append(indices[i+1][1])
    return list(zip(onsets, offsets))

def get_contin_label_idxs(arr, th, label):
    '''Get indices of intervals with only the given label'''
    indices = []
    for i in range(len(arr)):
        chunk = arr[i:i+th]
        if len(chunk) < th:
            continue
        chunk = list(set(chunk))
        if len(chunk)==1 and chunk[0]==label:
            indices.append((i,i+th-1))
    return indices


def _compute_circ_features(model_path, signal, subject, config,
                          sleep_threshold=20, sleep_label=82,
                          idx_dist_th=240):
    if os.path.isdir(model_path):
        if subject.endswith('.csv'):
            df = pd.read_csv(os.path.join(model_path, subject))
        else:
            df = pd.read_parquet(os.path.join(model_path, subject))
        window_size = config.SAMPLE_RATE * 60  # 1min window size
        aggr_p = aggregate_labels(
            labels=df.prediction.values,
            frame_length=window_size,
            frame_step=window_size
        )
        aggr_ts = src.utils.windowed_timestamps(
            df.timestamp.values,
            frame_length=window_size,
            frame_step=window_size,
            kind='center'
        )
        sleep_phase_indices_p = get_sleep_on_and_offset_indices(
            aggr_p,
            sleep_threshold=sleep_threshold,
            sleep_label=sleep_label,
            idx_dist_th=idx_dist_th
        )
        # We consider the first sleep phase for midpoint sleep estimation
        first_onset = aggr_ts[sleep_phase_indices_p[0][0]]
        first_offset = aggr_ts[sleep_phase_indices_p[0][1]]
        mps = get_midpoint_sleep(first_onset, first_offset)
        to_dt = np.vectorize(lambda x: datetime.strptime(x[:26],'%Y-%m-%d %H:%M:%S.%f'))
        s = pd.Series(to_dt(signal.index.values))
        res = np.cos((s.dt.hour+s.dt.minute/60)*(2.*np.pi/24)-((mps.hour+mps.minute/60)*2.*np.pi/24))
        return res
    elif os.path.exists(model_path) and model_path.endswith('.pkl'):
        # TODO
        return None
    else:
        raise ValueError(f'model_path needs to be a folder or path to a pkl file but it is {model_path}')


def compute_circ_features(model_path, signal, subject, config,
                          sleep_threshold=20, sleep_label=82,
                          idx_dist_th=240):
    '''Computes circadian cosine wave depending on midpoint-sleep

    Can be based on a model that creates sleep/wake predictions or
    based on existing predictions created by a model.

    Circadian features is a cosine curve defined between -1 and 1 and
    shifted according to midpoint sleep estimation such that it has
    the max value of 1 at the midpoint sleep estimation.

    Parameters
    ----------
    model_path (str): Either path to model.pkl or path to predictions
    signal (pd.DataFrame): Original signal
    subject (str): Subject filename
    config (src.config.Config): Used config class
    sleep_threshold (int, optional): Amount of minutes to consider
    sleep_label (int, optional): Label number used for sleep class
    idx_dist_th (int, optional): Amount of minutes between sleep phases
        Default is 240min since we don't expect another sleep phase after
        sleep-onset for the next 4h


    Returns
    -------
    (pd.DataFrame)

    '''
    if os.path.exists(model_path) and model_path.endswith('.zip'):
        # Unzip to temp folder first
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            subject_parquet = subject.split('.')[0]+'.parquet'
            tmp_model_path = os.path.join(temp_dir,'predictions')
            return _compute_circ_features(tmp_model_path, signal, subject_parquet, config,
                                          sleep_threshold, sleep_label, idx_dist_th)
    else:
        return _compute_circ_features(model_path, signal, subject, config,
                                      sleep_threshold, sleep_label, idx_dist_th)


class Featurizer:

    def __init__(
        self, data,
        cols_per_sensor,
        additional_features={},
        **kwargs
    ):
        self._data = data
        self.cols_per_sensor = cols_per_sensor
        self.axes_names = [j for i in cols_per_sensor.values() for j in i]
        self._cache = {}
        self.additional_features = additional_features
        if len(self._data.shape) != 3:
            raise ValueError('Data must be a rank 3 tensor (batch,sequence,axes)')
        if not len(self.axes_names) == self.num_axes:
            raise ValueError('Received badly formatted axes_names: length {len(axes_names)} != {self.num_axes}')

    @classmethod
    def get(cls, data, config, additional_features={}):
        """
        Convenience method for creating and using featurizer.
        """
        if config.OLD_FEATURIZER:
            sub_cls = OldFeaturizer(
                data,
                axes_names=config.SENSOR_COLUMNS,
                sample_rate=config.SAMPLE_RATE,
                additional_features=additional_features
            )
            return sub_cls(config.FEATURES)
        else:
            features = []
            for sensor_type, sensor_settings in config.SENSORS.items():
                sub_cls = [sc for sc in cls.__subclasses__() if sc.__name__==f'{sensor_type}Featurizer']
                try:
                    sub_cls = sub_cls[0]
                except IndexError as e:
                    print(f'Unknown sensor featurizer: "{sensor_type}Featurizer", using Featurizer...')
                    sub_cls = cls
                # columns of sensor type
                _cols = config.sensor_type_column_map[sensor_type]
                column_indices = [config.SENSOR_COLUMNS.index(x) for x in _cols]
                sensor_data = data[:,:,column_indices]
                res = sub_cls(
                    sensor_data,
                    sensor_settings['COLUMNS'],
                    additional_features,
                    sample_rate=config.SAMPLE_RATE
                )
                features.append(res(sensor_settings['FEATURES']))
            df = pd.concat(features, axis=1)
            return df

    def __repr__(self):
        return f'<Featurizer data={self.data.shape}>'

    def __call__(self, feature_names):
        """
        Get features by name and concatenate into design matrix.
        """
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        df = pd.concat([getattr(self, fn) for fn in feature_names], axis=1)
        for f, v in self.additional_features.items():
            df[f] = v
        return df

    @property
    def num_data(self):
        return self.data.shape[0]

    @property
    def sequence_length(self):
        return self.data.shape[1]

    @property
    def num_axes(self):
        return self._data.shape[2]

    @feature
    def energy(self):
        '''Energy of each signal'''
        return np.sum(self.data**2, axis=1)


    @feature
    def freq_mean(self):
        """
        Average amplitude.
        """
        return self.fft_ampl.mean(axis=1)

    @feature
    def freq_std(self):
        return self.fft_ampl.std(axis=1)

    @feature
    def freq_cent(self):
        """
        Spectral centroid.
        """
        sums = np.sum(self.fft_ampl, axis=1)
        sums = np.where(sums, sums, 1.)  # Avoid dividing by zero
        return np.sum(self.fft_freq.reshape(1, -1, 1) * self.fft_ampl, axis=1) / sums

    #########################################################################
    #              Features mentioned in [Narayanan et al. 2020]            #
    #########################################################################

    @property
    def raw(self):
        '''Raw signal with magnitude'''
        data_with_magnitudes = [self._data] + self.magnitude
        return np.concatenate(data_with_magnitudes, axis=2)

    @cached
    def data(self):
        return self.raw

    @cached
    def data_sorted(self):
        """
        Sorts acceleration signal for computing quantiles efficiently.
        """
        copy = self.data.copy()
        copy.sort(axis=1)
        return copy


    @cached
    def magnitude(self):
        '''The magnitude of all multi-dimensional sensors'''
        magnitudes = []
        for i, (sensor_name, sensor_cols) in enumerate(self.cols_per_sensor.items()):
            if len(sensor_cols) == 1: continue
            magnitude = np.linalg.norm(self._data[:,:,i*len(sensor_cols):(i+1)*len(sensor_cols)],
                                       axis=2)
            magnitude = magnitude.reshape(list(magnitude.shape)+[1])
            magnitudes.append(magnitude)
            self.axes_names.append(f'magnitude_{sensor_name}')
        return magnitudes

    @feature
    def count(self):
        # TODO
        pass

    def _mean(self, data=None):
        """
        Mean of the signal/component.
        """
        return data.mean(axis=1)

    @feature
    def mean(self):
        """
        Mean of the acceleration signal/component.
        """
        return self._mean(self.data)

    def _std(self, data=None):
        """
        Std of the acceleration signal/component.
        """
        return data.std(axis=1)

    @feature
    def std(self):
        """
        Std of the acceleration signal/component.
        """
        return self._std(self.data)

    def _cv(self, data=None):
        '''The coefficient of variation: std/mean of the signal'''
        cv = scipy.stats.variation(data, axis=1)
        cv[np.isnan(cv)]=0
        return cv

    @feature
    def cv(self):
        '''The coefficient of variation: std/mean of the default signal'''
        return self._cv(self.data)

    def _q0(self, data_sorted=None):
        """
        0th quantile of acceleration (min).
        """
        return data_sorted[:, 0, :]

    def _q25(self, data_sorted=None):
        """
        25th quantile
        """
        return data_sorted[:, int(0.25 * data_sorted.shape[1]), :]

    def _q50(self, data_sorted=None):
        """
        50th quantile
        """
        return data_sorted[:, int(0.50 * data_sorted.shape[1]), :]

    def _q75(self, data_sorted=None):
        """
        75th quantile
        """
        return data_sorted[:, int(0.75 * data_sorted.shape[1]), :]

    def _q100(self, data_sorted=None):
        """
        100th quantile (max).
        """
        return data_sorted[:, -1, :]

    @feature
    def q0(self):
        """
        0th quantile of acceleration (min).
        """
        return self._q0(self.data_sorted)

    @feature
    def q25(self, data_sorted=None):
        """
        25th quantile
        """
        return self._q25(self.data_sorted)

    @feature
    def q50(self, data_sorted=None):
        """
        50th quantile
        """
        return self._q50(self.data_sorted)

    @feature
    def q75(self, data_sorted=None):
        """
        75th quantile
        """
        return self._q75(self.data_sorted)

    @feature
    def q100(self, data_sorted=None):
        """
        100th quantile (max).
        """
        return self._q100(self.data_sorted)

    @feature
    def axes_corr(self):
        """
        Compute correlation between all axes (and the magnitudes).
        """
        corr = pd.DataFrame()
        for i, name1 in enumerate(self.axes_names):
            for j, name2 in enumerate(self.axes_names):
                if j <= i: continue
                # If exactly one of both is magnitude, do nothing
                if ('magnitude' in name1 and 'magnitude' not in name2) or \
                   ('magnitude' in name2 and 'magnitude' not in name1):
                    continue
                x = self.data[:, :, i]
                y = self.data[:, :, j]
                corr[f'axes_corr_{name1}_{name2}'] = (((x * y).mean(axis=1) - x.mean(axis=1) * y.mean(axis=1)) /
                                                      (x.std(axis=1) * y.std(axis=1)).clip(0.000001, None))
        return corr

    def _axes_mean(self, data=None):
        '''Mean across sensor axes'''
        means = pd.DataFrame()
        for i, (name_i, cols_i) in enumerate(self.cols_per_sensor.items()):
            for j, (name_j, cols_j) in enumerate(self.cols_per_sensor.items()):
                if j <= i: continue
                for col_i in cols_i:
                    col_i_idx = self.axes_names.index(col_i)
                    for col_j in cols_j:
                        col_j_idx = self.axes_names.index(col_j)
                        x = data[:,:,col_i_idx]
                        y = data[:,:,col_j_idx]
                        x = x.reshape(list(x.shape)+[1])
                        y = y.reshape(list(y.shape)+[1])
                        z = np.concatenate([x, y], axis=2)
                        m = z.mean(axis=1).mean(axis=1)
                        means[f'mean_{col_i}_{col_j}'] = m
        return means

    @feature
    def axes_mean(self):
        return self._axes_mean(self.data)

    @feature
    def skew(self):
        """
        Skew of the temporal signal without gravity.
        Also different as Narayanan uses the gravity signal
        """

        res = scipy.stats.skew(self.data, axis=1)
        res = np.nan_to_num(res, nan=0.0)
        return res

    @feature
    def kurtosis(self):
        """
        Kurtosis of the temporal signal without gravity.
        Also different as Narayanan uses the gravity signal
        """
        res = scipy.stats.kurtosis(self.data, axis=1)
        res = np.nan_to_num(res, nan=0.0)
        return res

    @feature
    def peaks(self):
        # TODO
        pass

    @feature
    def roll_mean(self):
        # TODO: Do not consider magnitude
        pass

    @feature
    def pitch_mean(self):
        # TODO: Do not consider magnitude
        pass

    @feature
    def yaw_mean(self):
        # TODO: Even possible?: Do not consider magnitude
        pass

    @feature
    def roll_std(self):
        # TODO: Do not consider magnitude
        pass

    @feature
    def pitch_std(self):
        # TODO: Do not consider magnitude
        pass

    @feature
    def yaw_std(self):
        # TODO: Even possible?: Do not consider magnitude
        pass

    @cached
    def fft_ampl(self):
        """
        Fourier transform amplitudes.
        This is actually the magnitude as the absolute value
        is computed.
        """
        return np.abs(np.fft.rfft(self.data, axis=1))

    @cached
    def fft_power(self):
        '''Power spectrum of signal using FFT'''
        return self.fft_ampl**2

    @cached
    def fft_freq(self):
        """
        Actual frequency values corresponding to the amplitudes.
        """
        return np.fft.rfftfreq(self.data.shape[1])

    @feature
    def freq_dom(self):
        '''The dominant frequency (max)'''
        return self.fft_freq[self.fft_ampl.argmax(axis=1)]

    @feature
    def freq_dom_mag(self):
        '''The dominant's frequency magnitude'''
        # Narayanan use signal power of dom freq but in
        # paper they use magnitude, so use it here as well
        return self.fft_ampl.max(axis=1)

    @feature
    def freq_total_signal_power(self):
        '''The total signal power, sum of power amplitudes'''
        # Is only mentioned in paper of Narayanan not in code
        return np.sum(self.fft_power, axis=1)



class AccelerationFeaturizer(Featurizer):
    '''Subclass with functions for acceleration data in particular'''

    def __init__(self, data,
                 cols_per_sensor,
                 additional_features={},
                 sample_rate=50, gravity_freq=1):
        self.gravity_freq = gravity_freq
        self.sample_rate = sample_rate
        super().__init__(
            data=data,
            axes_names=[j for i in cols_per_sensor.values() for j in i],
            cols_per_sensor=cols_per_sensor,
            additional_features=additional_features
        )

    @cached
    def gravity(self):
        """
        Estimates gravity by applying a (very low) low pass filter.
        """
        b, a = scipy.signal.butter(4, self.gravity_freq,
                                   fs=self.sample_rate,
                                   btype='lowpass')
        return scipy.signal.filtfilt(b, a, self.raw, axis=1)

    @cached
    def gravity_sorted(self):
        """
        Sorts gravity signal for computing quantiles efficiently.
        """
        copy = self.gravity.copy()
        copy.sort(axis=1)
        return copy

    @cached
    def data(self):
        """
        Subtracts gravity signal from data.
        """
        return self.raw - self.gravity

    @feature
    def axes_mean_gravity(self):
        '''Mean across all axes of gravity component'''
        return self._axes_mean(data=self.gravity)

    @feature
    def q0_gravity(self):
        """
        0th quantile of gravity (min).
        """
        return self._q0(self.gravity_sorted)

    @feature
    def q25_gravity(self):
        """
        25th quantile of gravity.
        """
        return self._q25(self.gravity_sorted)

    @feature
    def q50_gravity(self):
        """
        50th quantile of gravity.
        """
        return self._q50(self.gravity_sorted)

    @feature
    def q75_gravity(self):
        """
        75th quantile of gravity.
        """
        return self._q75(self.gravity_sorted)

    @feature
    def q100_gravity(self):
        """
        100th quantile of gravity (max).
        """
        return self._q100(self.gravity_sorted)

    @feature
    def cv_gravity(self):
        '''The coefficient of variation: std/mean of the gravity signal'''
        return self._cv(self.gravity)

    @feature
    def std_gravity(self):
        """
        Std of the gravity signal/component.
        """
        return self._std(self.gravity)

    @feature
    def mean_gravity(self):
        """
        Mean of the acceleration signal/component.
        """
        return self._mean(self.gravity)


class OldFeaturizer(Featurizer):
    '''Featurizer until commit 186842d148b0e1e3ad6754538de9e36447011314

    Keep the old implementation since older models depend on it

    '''

    def __init__(self, data, axes_names=None, sample_rate=50,
                 gravity_freq=1, additional_features={}):

        self._data = data
        self.axes_names_without_magnitude = axes_names or list(map(str, range(data.shape[-1])))
        self.axes_names = self.axes_names_without_magnitude.copy()
        self.axes_names.append('magnitude_b')
        self.axes_names.append('magnitude_t')
        self.gravity_freq = gravity_freq
        self.sample_rate = sample_rate
        self._cache = {}
        self.additional_features = additional_features

        if len(self._data.shape) != 3:
            raise ValueError('Data must be a rank 3 tensor (batch,sequence,axes)')

        if not len(self.axes_names) == self.num_axes:
            raise ValueError('Received badly formatted axes_names: length {len(axes_names)} != {self.num_axes}')

    @property
    def num_axes(self):
        return self.data.shape[2]


    #########################################################################
    #              Features mentioned in [Narayanan et al. 2020]            #
    #########################################################################

    @property
    def raw_without_magnitude(self):
        """
        Raw data, without mean subtraction.
        """
        return self._data

    @property
    def raw(self):
        '''Raw signal with magnitude'''
        b_magnitudes, t_magnitudes = self.magnitude
        b_magnitudes = b_magnitudes.reshape(list(b_magnitudes.shape)+[1])
        t_magnitudes = t_magnitudes.reshape(list(t_magnitudes.shape)+[1])
        return np.concatenate([self.raw_without_magnitude,
                               b_magnitudes, t_magnitudes], axis=2)

    @cached
    def data_without_magnitude(self):
        """
        Subtracts gravity signal from data.
        """
        res = self.raw_without_magnitude - self.gravity_without_magnitude
        return res

    @cached
    def data(self):
        """
        Subtracts gravity signal from data.
        """
        return self.raw - self.gravity

    @cached
    def data_sorted(self):
        """
        Sorts acceleration signal for computing quantiles efficiently.
        """
        copy = self.data.copy()
        copy.sort(axis=1)
        return copy

    @cached
    def gravity_without_magnitude(self):
        """
        Estimates gravity without magnitude column.
        """
        b, a = scipy.signal.butter(4, self.gravity_freq,
                                   fs=self.sample_rate,
                                   btype='lowpass')
        return scipy.signal.filtfilt(b, a,
                                     self.raw_without_magnitude,
                                     axis=1)

    @cached
    def gravity(self):
        """
        Estimates gravity by applying a (very low) low pass filter.
        """
        b, a = scipy.signal.butter(4, self.gravity_freq,
                                   fs=self.sample_rate,
                                   btype='lowpass')
        return scipy.signal.filtfilt(b, a, self.raw, axis=1)

    @cached
    def gravity_sorted_without_magnitude(self):
        """
        Sorts gravity signal for computing quantiles efficiently.
        """
        copy = self.gravity_without_magnitude.copy()
        copy.sort(axis=1)
        return copy

    @cached
    def gravity_sorted(self):
        """
        Sorts gravity signal for computing quantiles efficiently.
        """
        copy = self.gravity.copy()
        copy.sort(axis=1)
        return copy

    @cached
    def magnitude(self):
        '''The magnitude of the raw data of the back and thigh sensor'''
        b_magnitude = np.linalg.norm(self.raw_without_magnitude[:,:,:3],
                                     axis=2)
        t_magnitude = np.linalg.norm(self.raw_without_magnitude[:,:,3:],
                                     axis=2)
        return b_magnitude, t_magnitude

    @feature
    def mean_gravity(self):
        """
        Mean of the gravity signal/component.
        """
        return self.gravity.mean(axis=1)

    @feature
    def mean_acceleration(self):
        """
        Mean of the acceleration signal/component.
        """
        return self.data.mean(axis=1)

    @feature
    def std_gravity(self):
        """
        Std of the gravity signal/component.
        """
        return self.gravity.std(axis=1)

    @feature
    def std_acceleration(self):
        """
        Std of the acceleration signal/component.
        """
        return self.data.std(axis=1)

    @feature
    def cv_gravity(self):
        '''The coefficient of variation: std/mean of the gravit signal'''
        cv = scipy.stats.variation(self.gravity, axis=1)
        # Replace nan with 0:
        cv[np.isnan(cv)]=0
        return cv

    @feature
    def cv_acceleration(self):
        '''The coefficient of variation: std/mean of the acc signal'''
        cv = scipy.stats.variation(self.data, axis=1)
        cv[np.isnan(cv)]=0
        return cv

    @feature
    def q0_gravity(self):
        """
        0th quantile of gravity (min).
        """
        return self.gravity_sorted[:, 0, :]

    @feature
    def q0_acceleration(self):
        """
        0th quantile of acceleration (min).
        """
        return self.data_sorted[:, 0, :]

    @feature
    def q25_gravity(self):
        """
        25th quantile of gravity.
        """
        return self.gravity_sorted[:, int(0.25 * self.data.shape[1]), :]

    @feature
    def q25_acceleration(self):
        """
        25th quantile of acc.
        """
        return self.data_sorted[:, int(0.25 * self.data.shape[1]), :]

    @feature
    def q50_gravity(self):
        """
        50th quantile of gravity.
        """
        return self.gravity_sorted[:, int(0.50 * self.data.shape[1]), :]

    @feature
    def q50_acceleration(self):
        """
        50th quantile of acc.
        """
        return self.data_sorted[:, int(0.50 * self.data.shape[1]), :]

    @feature
    def q75_gravity(self):
        """
        75th quantile of gravity.
        """
        return self.gravity_sorted[:, int(0.75 * self.data.shape[1]), :]

    @feature
    def q75_acceleration(self):
        """
        75th quantile of acc.
        """
        return self.data_sorted[:, int(0.75 * self.data.shape[1]), :]

    @feature
    def q100_gravity(self):
        """
        100th quantile of gravity (max).
        """
        return self.gravity_sorted[:, -1, :]

    @feature
    def q100_acceleration(self):
        """
        100th quantile of acc (max).
        """
        return self.data_sorted[:, -1, :]

    @feature
    def axes_corr(self):
        """
        Compute correlation between all axes (and the magnitudes).
        """
        corr = pd.DataFrame()
        for i, name1 in enumerate(self.axes_names_without_magnitude):
            for j, name2 in enumerate(self.axes_names_without_magnitude):
                if j <= i: continue
                x = self.data[:, :, i]
                y = self.data[:, :, j]
                corr[f'axes_corr_{name1}_{name2}'] = (((x * y).mean(axis=1) - x.mean(axis=1) * y.mean(axis=1)) /
                                                      (x.std(axis=1) * y.std(axis=1)).clip(0.000001, None))
        # correlation between the magnitudes:
        x = self.data[:, :, -2]
        y = self.data[:, :, -1]
        name1, name2 = self.axes_names[-2:]
        corr[f'axes_corr_{name1}_{name2}'] = (((x * y).mean(axis=1) - x.mean(axis=1) * y.mean(axis=1)) /
                                              (x.std(axis=1) * y.std(axis=1)).clip(0.000001, None))
        return corr

    @feature
    def axes_mean(self):
        '''Mean across all axes of the gravity component'''
        back_names = self.axes_names_without_magnitude[:3]
        thigh_names = self.axes_names_without_magnitude[3:]
        means = pd.DataFrame()
        for i, b_column_name in enumerate(back_names):
            for j, t_column_name in enumerate(thigh_names):
                x = self.gravity[:, :, i]
                y = self.gravity[:, :, j+3]
                x = x.reshape(list(x.shape)+[1])
                y = y.reshape(list(y.shape)+[1])
                z = np.concatenate([x, y], axis=2)
                m = z.mean(axis=1).mean(axis=1)
                means[f'mean_{b_column_name}_{t_column_name}'] = m
        return means
