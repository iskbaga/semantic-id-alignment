import json
import sys
from collections import defaultdict
from pathlib import Path

import hydra
import numpy as np
import torch
from codebook_utils import codebook_initialize, fix_dead_codebooks
from loguru import logger
from models import RQVAE
from omegaconf import DictConfig, OmegaConf
from torchdata.stateful_dataloader import StatefulDataLoader


sys.path.append("..")

from modeling.datasets import EmbeddingsDataset
from modeling.training import EarlyStopper, TensorboardLogger
from modeling.utils import fix_random_seed, run_evaluation


def collate_fn(device):
    def _transform(batch):
        item_ids = [item["item_id"] for item in batch]
        embeddings = [item["embedding"] for item in batch]
        item_ids_tensor = torch.tensor(item_ids, dtype=torch.int64, device=device)
        embeddings_tensor = torch.from_numpy(np.stack(embeddings)).float().to(device, non_blocking=True)
        return {"item_id": item_ids_tensor, "embedding": embeddings_tensor}

    return _transform


def run_inference(model, dataloader, save_path):
    model.eval()

    all_results = []
    accumulators = defaultdict(list)

    with torch.inference_mode():
        for batch in dataloader:
            _, outputs = model(batch)

            accumulators["loss"].append(outputs["loss"])
            accumulators["recon_loss"].append(outputs["recon_loss"])
            accumulators["rqvae_loss"].append(outputs["rqvae_loss"])

            item_ids = batch["item_id"].cpu().numpy().tolist()
            clusters = outputs["clusters"].cpu().numpy().tolist()

            for item_id, cluster in zip(item_ids, clusters, strict=True):
                all_results.append({"item_id": item_id, "clusters": cluster})

    model.train()

    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    final_metrics = {name: sum(values) / len(values) for name, values in accumulators.items()}
    return final_metrics, all_results


def generate_constants(cfg: DictConfig):
    split_name = (
        f"{cfg.dataset.train_parts[0]}-{cfg.dataset.train_parts[1]}TR_"
        f"{cfg.dataset.val_parts[0]}-{cfg.dataset.val_parts[1]}V_"
        f"{cfg.dataset.test_parts[0]}-{cfg.dataset.test_parts[1]}T"
    )

    results_path = Path(cfg.paths.results_dir) / split_name / "rqvae-tiger"
    results_path.mkdir(parents=True, exist_ok=True)

    interactions_path = Path(cfg.paths.data_dir) / "all_data_interactions_with_groups.parquet"
    embeddings_path = Path(cfg.paths.data_dir) / "items_metadata_remapped.parquet"
    experiment_name = f"tiger_{cfg.dataset.name}_{split_name}"

    return {
        "INTERACTIONS_PATH": interactions_path,
        "EMBEDDINGS_PATH": embeddings_path,
        "EXPERIMENT_NAME": experiment_name,
        "INFERENCE_PATH": results_path / "all_clusters.json",
        "ALL_MAPPING_PATH": results_path / "all_clusters_colisionless.json",
        "OUTPUT_TRAIN_MAPPING_PATH": results_path
        / f"only_{cfg.dataset.tiger_train_parts[0]}-{cfg.dataset.tiger_train_parts[1]}TR_clusters_colisionless.json",
    }


def train_rqvae(cfg: DictConfig):
    consts = generate_constants(cfg)
    logger.info(f"Generated constants: {consts}")
    fix_random_seed(cfg.training.seed_value)

    device = cfg.training.device if torch.cuda.is_available() and cfg.training.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    dataset = EmbeddingsDataset(
        all_interactions_path=consts["INTERACTIONS_PATH"],
        all_embeddings_path=consts["EMBEDDINGS_PATH"],
        train_parts=cfg.dataset.train_parts,
    )

    train_dataloader = StatefulDataLoader(
        dataset=dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn(device),
    )

    valid_dataloader = StatefulDataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(device),
    )

    model = RQVAE(
        input_dim=cfg.model.input_dim,
        num_codebooks=cfg.model.num_codebooks,
        codebook_size=cfg.model.codebook_size,
        embedding_dim=cfg.model.hidden_dim,
        beta=cfg.model.beta,
    ).to(device)

    codebook_initialize(model, valid_dataloader)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f"Overall parameters: {total_params:,}")
    logger.debug(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    tensorboard_logger = TensorboardLogger(experiment_name=consts["EXPERIMENT_NAME"], logdir=cfg.paths.tensorboard_dir)

    early_stopper = EarlyStopper(
        metric=cfg.training.metric,
        patience=cfg.training.patience,
        minimize=cfg.training.minimize_metric,
        checkpoints_dir=cfg.paths.checkpoints_dir,
        experiment_name=consts["EXPERIMENT_NAME"],
    )

    logger.debug("Everything is ready for training process!")

    last_max_collisions = 0

    for epoch in range(cfg.training.num_epochs):
        logger.debug(f"Starting epoch: {epoch + 1}")
        model.train()

        train_accumulators = defaultdict(list)

        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, outputs = model(batch)
            loss.backward()
            optimizer.step()

            num_fixed, max_collisions = fix_dead_codebooks(model, valid_dataloader)
            last_max_collisions = max_collisions

            train_accumulators["train/loss"].append(outputs["loss"])
            train_accumulators["train/recon_loss"].append(outputs["recon_loss"])
            train_accumulators["train/rqvae_loss"].append(outputs["rqvae_loss"])
            train_accumulators["num_dead/0"].append(num_fixed[0] if len(num_fixed) > 0 else 0)
            train_accumulators["num_dead/1"].append(num_fixed[1] if len(num_fixed) > 1 else 0)
            train_accumulators["num_dead/2"].append(num_fixed[2] if len(num_fixed) > 2 else 0)

        train_metrics = {key: sum(values) / len(values) for key, values in train_accumulators.items()}
        train_metrics["num_dead/max_collisitons_num"] = last_max_collisions

        validation_metrics = run_evaluation(
            model, valid_dataloader, "validation/", ["loss", "recon_loss", "rqvae_loss"]
        )
        all_metrics = {**train_metrics, **validation_metrics}
        tensorboard_logger.add_metrics((epoch + 1) * (batch_idx + 1), all_metrics)

        if early_stopper.check(all_metrics[cfg.training.metric], model):
            logger.info("Early stopping triggered")
            break

    tensorboard_logger.close()

    best_model_file = early_stopper.get_best_model_path()
    assert best_model_file is not None

    logger.info(f"Loading best model from: {best_model_file}")
    state_dict = torch.load(best_model_file)
    model.load_state_dict(state_dict)

    inference_dataset = EmbeddingsDataset(
        all_interactions_path=consts["INTERACTIONS_PATH"], all_embeddings_path=consts["EMBEDDINGS_PATH"]
    )

    inference_dataloader = StatefulDataLoader(
        inference_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(device),
    )

    inference_metrics, _ = run_inference(model, inference_dataloader, consts["INFERENCE_PATH"])

    inference_metrics_prefixed = {f"valid/{k}": v for k, v in inference_metrics.items()}
    logger.info("Inference metrics: " + ", ".join(f"{k} {v:.4f}" for k, v in inference_metrics_prefixed.items()))

    with open(consts["INFERENCE_PATH"]) as f:
        mappings = json.load(f)

    all_mapping = {}
    sem_2_ids = defaultdict(list)
    for mapping in mappings:
        item_id = mapping["item_id"]
        clusters = mapping["clusters"]
        all_mapping[int(item_id)] = clusters
        sem_2_ids[tuple(clusters)].append(int(item_id))

    for items in sem_2_ids.values():
        assert len(items) <= cfg.model.codebook_size, str(len(items))
        collision_solvers = np.random.permutation(cfg.model.codebook_size)[: len(items)].tolist()
        for item_id, collision_solver in zip(items, collision_solvers, strict=True):
            all_mapping[item_id].append(collision_solver)
            for i in range(len(all_mapping[item_id])):
                all_mapping[item_id][i] += cfg.model.codebook_size * i

    with open(consts["ALL_MAPPING_PATH"], "w") as f:
        json.dump(all_mapping, f, indent=2)

    train_interactions = dataset.get_interactions_by_part(
        cfg.dataset.tiger_train_parts[0], cfg.dataset.tiger_train_parts[1]
    )

    train_item_ids = set(train_interactions["item_id"].unique())

    logger.debug(f"Found {len(train_item_ids)} unique train items")

    train_mapping = {}
    missing_count = 0

    for item_id in train_item_ids:
        item_id_str = item_id
        if item_id_str in all_mapping:
            train_mapping[item_id_str] = all_mapping[item_id_str]
        else:
            missing_count += 1

    if missing_count > 0:
        logger.debug(f"{missing_count} items from train not found in all_mapping")

    logger.debug(f"Created train_mapping with {len(train_mapping)} items")

    with open(consts["OUTPUT_TRAIN_MAPPING_PATH"], "w") as f:
        json.dump(train_mapping, f, indent=2)
    logger.debug(f"Saved to {consts['OUTPUT_TRAIN_MAPPING_PATH']}")

    logger.debug(f"all_mapping size: {len(all_mapping)}")
    logger.debug(f"train_mapping size: {len(train_mapping)}")
    logger.debug(f"train_mapping/all_mapping ratio: {len(train_mapping) / len(all_mapping):.1%}")

    logger.info("Training completed successfully!")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    train_rqvae(cfg)


if __name__ == "__main__":
    main()
