from torch.utils.data import Dataset
import numpy as np


class NShotWrapper(Dataset):
    """Wraps one of the two Dataset classes to create a new Dataset that returns n-shot, k-way, q-query tasks."""
    def __init__(self, dataset, epoch_length, n, k, q):
        self.dataset = dataset
        self.epoch_length = epoch_length
        self.n_shot = n
        self.k_way = k
        self.q_queries = q

    def __getitem__(self, item):
        """Get a single n-shot, k-way, q-query task."""
        # Select classes
        episode_classes = np.random.choice(self.dataset.df['class_id'].unique(), size=self.k_way, replace=False)
        df = self.dataset.df[self.dataset.df['class_id'].isin(episode_classes)]
        batch = []

        for k in episode_classes:
            # Select support examples
            support = df[df['class_id'] == k].sample(self.n_shot)

            for i, s in support.iterrows():
                x, y = self.dataset[s['id']]
                batch.append(x)

        for k in episode_classes:
            query = df[(df['class_id'] == k) & (~df['id'].isin(support['id']))].sample(self.q_queries)
            for i, q in query.iterrows():
                x, y = self.dataset[q['id']]
                batch.append(x)

        return np.stack(batch), episode_classes

    def __len__(self):
        return self.epoch_length
