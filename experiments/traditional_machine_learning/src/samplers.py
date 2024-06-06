from imblearn.under_sampling import RandomUnderSampler

class ActivitySubsetRandomSampler(RandomUnderSampler):
    def __init__(self, samples_per_activity):
        self.samples_per_activity = samples_per_activity
        print(f'Do under sampling with {samples_per_activity} samples per activity...')

    def fit_resample(self, X, y):
        classes =  y.drop_duplicates().sort_values().values
        num_classes = len(classes)
        class_dict = dict(zip(classes, [self.samples_per_activity]*num_classes))
        amounts = y.value_counts()
        for c, v in class_dict.items():
            class_dict[c] = min(amounts[c], v)
        print(f'Resample with: {class_dict}')
        super().__init__(sampling_strategy=class_dict, random_state=28)
        X_resampled, y_resampled = super().fit_resample(X, y)
        return X_resampled, y_resampled
