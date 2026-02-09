import numpy as np


class SasRecTrainDataset:
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, index):
        sample = self._dataset[index]

        item_sequence = sample['item.ids'][:-1]
        last_item = sample['item.ids'][1:]

        processed_sample = {
            'user.ids': np.array(sample['user.ids'], dtype=np.int64),
            'user.length': np.array([len(sample['user.ids'])], dtype=np.int64),

            'item.ids': np.array(item_sequence, dtype=np.int64),
            'item.length': np.array([len(item_sequence)], dtype=np.int64),

            'label.ids': np.array(last_item, dtype=np.int64),
            'label.length': np.array([len(last_item)], dtype=np.int64),
        }

        if 'num_train_items' in sample:
            processed_sample['num_train_items'] = np.array([sample['num_train_items']], dtype=np.int64)

        return processed_sample

    def __len__(self):
        return len(self._dataset)


class SasRecEvalDataset:
    def __init__(self, dataset):
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        sample = self._dataset[index]
        return {
            'user.ids': np.array(sample['user.ids'], dtype=np.int64),
            'user.length': np.array([len(sample['user.ids'])], dtype=np.int64),

            'item.ids': np.array(sample['item.ids'], dtype=np.int64),
            'item.length': np.array([len(sample['item.ids'])], dtype=np.int64),

            'label.ids': np.array(sample['label.ids'], dtype=np.int64),
            'label.length': np.array([len(sample['label.ids'])], dtype=np.int64),

            'visited.ids': np.array(sample['item.ids'], dtype=np.int64),
            'visited.length': np.array([len(sample['item.ids'])], dtype=np.int64),
        }
