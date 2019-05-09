import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self, samples_per_class: int, num_classes: int, n_features: int = 2):
        self.samples_per_class = samples_per_class
        self.num_classes = num_classes
        self.n_features = n_features

        # Create a dataframe to be consistent with other Datasets
        self.df = pd.DataFrame({
            'class_id': [i % self.num_classes for i in range(len(self))]
        })
        self.df = self.df.assign(id=self.df.index.values)

    def __len__(self):
        return self.samples_per_class * self.num_classes

    def __getitem__(self, item):
        class_id = item % self.num_classes
        return np.array([item] + [class_id]*self.n_features, dtype=np.float), float(class_id)