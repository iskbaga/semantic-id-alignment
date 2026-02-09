import numpy as np
import polars as pl
from loguru import logger


class EmbeddingsDataset:
    def __init__(self, all_interactions_path, all_embeddings_path, train_parts=None):
        self.all_interactions_path = all_interactions_path
        self.all_embeddings_path = all_embeddings_path

        logger.info('Loading all interactions...')

        self.all_interactions = pl.read_parquet(self.all_interactions_path)

        if 'part' not in self.all_interactions.columns:
            raise ValueError(f'all interactinons dataset must contain \'part\' column')

        logger.info('Loading all embeddings...')
        all_embeddings = pl.read_parquet(self.all_embeddings_path)

        logger.info(f'Loaded {len(self.all_interactions)} interactions')
        logger.info(f'Loaded {len(all_embeddings)} embeddings')

        if train_parts is not None:
            train_interactions = self.get_interactions_by_part(train_parts[0], train_parts[1])
            train_embeddings_dict = self._get_interactions_embeddings(train_interactions, all_embeddings)
        else:
            logger.info('TRAIN PARTS IS NONE (it is ok if it infer)')
            train_embeddings_dict = self._get_interactions_embeddings(self.all_interactions, all_embeddings)

        self.item_ids = list(train_embeddings_dict.keys())
        self.embeddings = list(train_embeddings_dict.values())

    def __getitem__(self, idx):
        tensor_emb = self.embeddings[idx]
        return {
            'item_id': self.item_ids[idx],
            'embedding': tensor_emb,
            'embedding_dim': len(tensor_emb)
        }

    def _get_interactions_embeddings(self, interactions, embeddings):
        interactions_item_ids = set(interactions['item_id'].unique())

        embeddings_dict = {}
        for row in embeddings.iter_rows(named=True):
            item_id = row['item_id']
            embedding = row['embedding']
            if item_id in interactions_item_ids:
                embeddings_dict[item_id] = np.array(embedding, dtype=np.float32)

        logger.info(f'Loaded {len(interactions_item_ids)} embeddings after filtration')
        return embeddings_dict

    def get_interactions_by_part(self, start_part: int, end_part: int):
        part_interactions = self.all_interactions.filter(
            (pl.col('part') >= start_part) &
            (pl.col('part') < end_part)
        )
        logger.info(f'All interactions size: {len(self.all_interactions)}')
        logger.info(f'Part interactions size: {len(part_interactions)}')
        return part_interactions

    def __len__(self):
        return len(self.embeddings)


class Dataset:
    def __init__(
            self,
            all_interactions_path,
            all_embeddings_path,
            train_parts=(0, 8),
            val_parts=(8, 9),
            test_parts=(9, 10),
            max_seq_len=20
    ):
        self.all_interactions_path = all_interactions_path
        self.all_embeddings_path = all_embeddings_path
        self.train_parts = train_parts
        self.val_parts = val_parts
        self.test_parts = test_parts
        self.max_sequence_length = max_seq_len

        self._validate_parts()
        self._load_data()
        self._create_samples()

    def _validate_parts(self):
        if self.train_parts[0] > self.train_parts[1]:
            raise ValueError('train_parts: start should be less than end')
        if self.val_parts[0] > self.val_parts[1]:
            raise ValueError('val_parts: start should be less than end')
        if self.test_parts[0] > self.test_parts[1]:
            raise ValueError('test_parts: should be less than end')

        if (self.train_parts[1] > self.val_parts[0] and self.train_parts[0] < self.val_parts[1]):
            logger.warning('Train and Val intersect!')
        if (self.train_parts[1] > self.test_parts[0] and self.train_parts[0] < self.test_parts[1]):
            logger.warning('Train and Test intersect!')
        if (self.val_parts[1] > self.test_parts[0] and self.val_parts[0] < self.test_parts[1]):
            logger.warning('Val and Test intersect!')

    def _load_data(self):
        logger.info('Loading all interactions...')

        self.all_interactions = pl.read_parquet(self.all_interactions_path)

        if 'part' not in self.all_interactions.columns:
            raise ValueError('all interactinons dataset must contain \'part\' column')

        logger.info('Loading all embeddings...')
        self.all_embeddings_df = pl.read_parquet(self.all_embeddings_path)

        self.embeddings_dict = {}
        for row in self.all_embeddings_df.iter_rows(named=True):
            item_id = row['item_id']
            embedding = row['embedding']
            self.embeddings_dict[item_id] = np.array(embedding, dtype=np.float32)

        logger.info(f'Loaded {len(self.all_interactions)} interactions')
        logger.info(f'Loaded {len(self.embeddings_dict)} embeddings')

        self.all_item_ids = set(self.all_interactions['item_id'].unique())
        self.all_user_ids = set(self.all_interactions['user_id'].unique())

        self.num_items = len(self.all_item_ids)
        logger.info(f'Unique items: {len(self.all_item_ids)}')
        logger.info(f'Unique users: {len(self.all_user_ids)}')

    def _create_samples(self):
        self.train_samples = self._create_train_samples()

        self.val_samples = self._create_eval_samples(
            eval_start_part=self.val_parts[0],
            eval_end_part=self.val_parts[1],
        )

        self.test_samples = self._create_eval_samples(
            eval_start_part=self.test_parts[0],
            eval_end_part=self.test_parts[1],
        )

    def _create_train_samples(self):
        samples = []

        history_interactions = self._get_interactions_by_part(
            self.train_parts[0],
            self.train_parts[1],
        )
        history_grouped = self._group_by_users(history_interactions)

        for row in history_grouped.iter_rows(named=True):
            user_id = row['user_id']
            items = row['item_ids']

            if len(items) < 2:
                continue

            samples.append({
                'user.ids': [user_id],
                'item.ids': items[-(self.max_sequence_length + 1):]
            })

        logger.info(f'Created {len(samples)} train samples')
        return samples

    def _create_eval_samples(
            self,
            eval_start_part,
            eval_end_part,
    ):
        samples = []

        history_interactions = self._get_interactions_by_part(
            self.train_parts[0],
            eval_start_part
        )
        history_grouped = self._group_by_users(history_interactions)

        label_interactions = self._get_interactions_by_part(
            eval_start_part,
            eval_end_part
        )
        labels_grouped = self._group_by_users(label_interactions)

        history_grouped = history_grouped.rename({'item_ids': 'history'})
        labels_grouped = labels_grouped.rename({'item_ids': 'labels'})

        joined = labels_grouped.join(history_grouped, on='user_id', how='inner')

        for row in joined.iter_rows(named=True):
            user_id = row['user_id']
            history = row['history']
            labels = row['labels']

            truncated_history = history[-self.max_sequence_length:]

            samples.append({
                'user.ids': [user_id],
                'item.ids': truncated_history,
                'label.ids': labels,
                'visited.ids': history,
            })

        logger.info(f'Created {len(samples)} samples')
        return samples

    def _get_item_ids_set(self, data):
        user_ids = set()
        for item_ids_array in data['item_ids']:
            user_ids.update(item_ids_array)
        return user_ids

    def _get_interactions_by_part(self, start_part: int, end_part: int):
        return self.all_interactions.filter(
            (pl.col('part') >= start_part) &
            (pl.col('part') < end_part)
        )

    def _group_by_users(self, interactions: pl.DataFrame) -> pl.DataFrame:
        grouped = (
            interactions
            .sort(by='original_order')
            .select(['original_order', 'user_id', 'item_id'])
            .group_by('user_id', maintain_order=True)
            .agg(
                pl.col('item_id')
                .sort_by('original_order')
                .alias('item_ids'),

                pl.col('original_order')
                .sort()
            )
            .sort('user_id')
        )
        return grouped

class FinetuneDataset(Dataset):

    def __init__(
            self,
            all_interactions_path,
            all_embeddings_path,
            train_parts=(0, 17),
            gap_parts=(17, 18),
            val_parts=(18, 19),
            test_parts=(19, 20),
            max_seq_len=20
    ):
        self.gap_parts = gap_parts
        super().__init__(
            all_interactions_path,
            all_embeddings_path,
            train_parts=train_parts,
            val_parts=val_parts,
            test_parts=test_parts,
            max_seq_len=max_seq_len
        )

    def _create_samples(self):
        self.train_samples = self._create_train_samples()

        self.val_samples = self._create_eval_samples(
            eval_start_part=self.val_parts[0],
            eval_end_part=self.val_parts[1],
        )

        self.test_samples = self._create_eval_samples(
            eval_start_part=self.test_parts[0],
            eval_end_part=self.test_parts[1],
        )

    def _create_train_samples(self):
        samples = []

        base_interactions = self._get_interactions_by_part(
            self.train_parts[0],
            self.train_parts[1],
        )
        base_grouped = self._group_by_users(base_interactions)

        gap_interactions = self._get_interactions_by_part(
            self.gap_parts[0],
            self.gap_parts[1],
        )
        gap_grouped = self._group_by_users(gap_interactions)

        base_grouped = base_grouped.rename({'item_ids': 'base'})
        gap_grouped = gap_grouped.rename({'item_ids': 'gap'})
        joined = gap_grouped.join(base_grouped, on='user_id', how='left')

        for row in joined.iter_rows(named=True):
            user_id = row['user_id']
            base_items = row['base']
            gap_items = row['gap']

            if base_items is None:
                history = gap_items
            else:
                history = base_items + gap_items

            if len(gap_items) < 1 or len(history) < 2:
                continue

            truncated_history = history[-(self.max_sequence_length + 1):]

            samples.append({
                'user.ids': [user_id],
                'item.ids': truncated_history,
                'num_train_items': min(len(gap_items), len(truncated_history) - 1)
            })

        logger.info(f'Created {len(samples)} finetune samples')
        return samples
