import murmurhash
import numpy as np
import torch
from loguru import logger


class TigerTrainDataset:
    def __init__(self, dataset, mapping_array, num_codebooks, num_user_hash):
        self._dataset = dataset
        self.mapping_array = mapping_array
        self.num_codebooks = num_codebooks
        self.num_user_hash = num_user_hash

    def __getitem__(self, index):
        sample = self._dataset[index]

        user_id = sample["user.ids"]
        user_hashed_id = murmurhash.hash(str(user_id)) % self.num_user_hash

        item_sequence = sample["item.ids"]
        semantic_ids = self.mapping_array[item_sequence].flatten()

        assert (semantic_ids != -1).all(), f"Missing mappings detected! Invalid positions: {(semantic_ids == -1).sum()}"

        return {
            "user.ids": np.array([user_id], dtype=np.int64),
            "user.hashed.ids": np.array([user_hashed_id], dtype=np.int64),
            "item.ids": np.array(item_sequence, dtype=np.int64),
            "item.length": np.array([len(item_sequence)], dtype=np.int64),
            "item.semantic.ids": semantic_ids,
            "item.semantic.length": np.array([len(item_sequence) * self.num_codebooks], dtype=np.int64),
        }

    def __len__(self):
        return len(self._dataset)


class TigerEvalDataset:
    def __init__(self, dataset, mapping_array, num_codebooks, num_user_hash):
        self._dataset = dataset
        self.mapping_array = mapping_array
        self.num_codebooks = num_codebooks
        self.num_user_hash = num_user_hash

    def __getitem__(self, index):
        sample = self._dataset[index]

        user_id = sample["user.ids"]
        user_hashed_id = murmurhash.hash(str(user_id)) % self.num_user_hash

        item_ids = sample["item.ids"]
        item_semantic_ids = self.mapping_array[item_ids].flatten()

        assert (item_semantic_ids != -1).all(), (
            f"Missing mappings detected in item! Invalid positions: {(item_semantic_ids == -1).sum()}"
        )

        label_ids = sample["label.ids"]
        label_semantic_ids = self.mapping_array[label_ids].flatten()

        assert (label_semantic_ids != -1).all(), (
            f"Missing mappings detected in label! Invalid positions: {(label_semantic_ids == -1).sum()}"
        )

        return {
            "user.ids": np.array([user_id], dtype=np.int64),
            "user.hashed.ids": np.array([user_hashed_id], dtype=np.int64),
            "item.ids": np.array(item_ids, dtype=np.int64),
            "item.length": np.array([len(item_ids)], dtype=np.int64),
            "item.semantic.ids": item_semantic_ids,
            "item.semantic.length": np.array([len(item_ids) * self.num_codebooks], dtype=np.int64),
            "label.ids": np.array(label_ids, dtype=np.int64),
            "label.length": np.array([len(label_ids)], dtype=np.int64),
            "label.semantic.ids": label_semantic_ids,
            "label.semantic.length": np.array([len(label_ids) * self.num_codebooks], dtype=np.int64),
            "visited.ids": np.array(item_ids, dtype=np.int64),
            "visited.length": np.array([len(item_ids)], dtype=np.int64),
        }

    def __len__(self):
        return len(self._dataset)


def tiger_preprocess(batch, num_codebooks, codebook_size):
    attention_mask = batch["item.semantic.mask"].bool()
    input_semantic_ids = batch["item.semantic.ids"].long()
    input_semantic_length = batch["item.semantic.length"].long()
    user_hashed_ids = batch["user.hashed.ids"].long()

    batch_size, max_seq_len = attention_mask.shape

    total_length = input_semantic_length.sum() + batch_size

    new_flatten_input_semantic_ids = torch.zeros(
        total_length, dtype=input_semantic_ids.dtype, device=input_semantic_ids.device
    )
    new_flatten_input_semantic_mask = torch.zeros(
        total_length, dtype=attention_mask.dtype, device=attention_mask.device
    ).bool()

    start_ids = torch.cumsum(input_semantic_length + 1, dim=-1) - (input_semantic_length + 1)
    new_flatten_input_semantic_mask[start_ids] = True

    new_flatten_input_semantic_ids[new_flatten_input_semantic_mask] = num_codebooks * codebook_size + user_hashed_ids
    new_flatten_input_semantic_ids[~new_flatten_input_semantic_mask] = input_semantic_ids

    attention_mask = torch.cat(
        [attention_mask, torch.ones((batch_size, 1), dtype=attention_mask.dtype, device=attention_mask.device)], dim=-1
    )

    new_input_semantic_ids = torch.zeros(
        batch_size, max_seq_len + 1, dtype=input_semantic_ids.dtype, device=input_semantic_ids.device
    )

    new_input_semantic_ids[attention_mask] = new_flatten_input_semantic_ids

    batch["input.data"] = new_input_semantic_ids
    batch["input.mask"] = attention_mask

    return batch


def to_masked(batch, prefix, is_right_aligned=True):
    data = batch[f"{prefix}.ids"]
    lengths = batch[f"{prefix}.length"]

    batch_size = lengths.shape[0]
    max_sequence_length = int(lengths.max())

    if len(data.shape) == 1:  # only indices
        padded_tensor = torch.zeros(
            (batch_size, max_sequence_length), dtype=data.dtype, device=data.device
        )  # (batch_size, max_seq_len)
    else:
        assert len(data.shape) == 2  # embeddings
        padded_tensor = torch.zeros(
            (batch_size, max_sequence_length, data.shape[-1]), dtype=data.dtype, device=data.device
        )  # (batch_size, max_seq_len, emb_dim)

    mask = torch.arange(max_sequence_length, device=lengths.device)[None] < lengths[:, None]

    if is_right_aligned:
        mask = torch.flip(mask, dims=[-1])

    padded_tensor[mask] = data

    batch[f"{prefix}.padded"] = padded_tensor
    batch[f"{prefix}.mask"] = mask

    return batch


def create_semantic_mapping_array(mapping, num_codebooks):
    max_item_id = max(int(k) for k in mapping)

    data = []
    for i in range(max_item_id + 1):
        if str(i) in mapping:
            data.append(mapping[str(i)])
        else:
            data.append([-1] * num_codebooks)

    mapping_array = np.array(data, dtype=np.int64)

    missing_count = (max_item_id + 1) - len(mapping)
    logger.debug(f"Missing mappings: {missing_count} items (-1 filled)")

    return mapping_array
