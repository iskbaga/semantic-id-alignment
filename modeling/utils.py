import os
import random

import numpy as np
import torch


def collate(batch):
    assert batch and isinstance(batch, list), batch
    processed_batch = {}

    for key in batch[0]:
        values = [sample[key] for sample in batch]
        if isinstance(values[0], dict):
            processed_batch[key] = collate(values)
        elif isinstance(values[0], np.ndarray):
            processed_batch[key] = np.empty(shape=(0,), dtype=values[0].dtype)
            values = [value for value in values if value.size > 0]
            if len(values) > 0:
                processed_batch[key] = np.concatenate(values)
        elif isinstance(values[0], torch.Tensor):
            processed_batch[key] = torch.empty(size=(0,), dtype=values[0].dtype)
            values = [value for value in values if value.numel() > 0]
            if len(values) > 0:
                if values[0].ndim == 0:
                    processed_batch[key] = torch.stack(values)
                else:
                    processed_batch[key] = torch.cat(values)
        else:
            processed_batch[key] = np.array(values)
    return processed_batch


def to_device_transform(batch, device):
    for key, value in list(batch.items()):
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
        elif isinstance(value, dict):
            batch[key] = to_device_transform(value, device)
    return batch


def run_evaluation(model, dataloader, prefix, metric_names=None):
    model.eval()
    metrics_data = {}

    with torch.inference_mode():
        for batch in dataloader:
            _, metrics = model(batch)
            for metric, value in metrics.items():
                if metric_names is None or metric in metric_names:
                    if metric not in metrics_data:
                        metrics_data[metric] = []
                    metrics_data[metric].append(value.item() if torch.is_tensor(value) else value)

    model.train()

    final_metrics = {}
    for metric, values in metrics_data.items():
        final_metrics[f"{prefix}{metric}"] = sum(values) / len(values) if values else 0.0
    return final_metrics


def fix_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
