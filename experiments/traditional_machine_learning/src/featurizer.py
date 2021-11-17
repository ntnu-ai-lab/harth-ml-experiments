import functools

import numpy as np
import pandas as pd
import scipy.signal
import scipy.stats


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


class Featurizer:

    def __init__(self, data, axes_names=None, sample_rate=50, gravity_freq=1):

        self._data = data
        self.axes_names_without_magnitude = axes_names or list(map(str, range(data.shape[-1])))
        self.axes_names = self.axes_names_without_magnitude.copy()
        self.axes_names.append('magnitude_b')
        self.axes_names.append('magnitude_t')
        self.gravity_freq = gravity_freq
        self.sample_rate = sample_rate
        self._cache = {}

        if len(self._data.shape) != 3:
            raise ValueError('Data must be a rank 3 tensor (batch,sequence,axes)')
        
        if not len(self.axes_names) == self.num_axes:
            raise ValueError('Received badly formatted axes_names: length {len(axes_names)} != {self.num_axes}')

    @classmethod
    def get(cls, feature_names, *args, **kwargs):
        """
        Convenience method for creating and using featurizer.
        """
        return cls(*args, **kwargs)(feature_names)

    def __repr__(self):
        return f'<Featurizer data={self.data.shape}>'

    def __call__(self, feature_names):
        """
        Get features by name and concatenate into design matrix.
        """
        if isinstance(feature_names, str):
            feature_names = [feature_names]
        return pd.concat([getattr(self, fn) for fn in feature_names], axis=1)

    @property
    def num_data(self):
        return self.data.shape[0]

    @property
    def sequence_length(self):
        return self.data.shape[1]

    @property
    def num_axes(self):
        return self.data.shape[2]

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
    def count(self):
        # TODO
        pass

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

    @feature
    def skew(self):
        """
        Skew of the temporal signal without gravity.
        Also different as Narayanan uses the gravity signal
        """
        return scipy.stats.skew(self.data, axis=1)

    @feature
    def kurtosis(self):
        """
        Kurtosis of the temporal signal without gravity.
        Also different as Narayanan uses the gravity signal
        """
        return scipy.stats.kurtosis(self.data, axis=1)

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
