import json
import sys
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf


sys.path.append("..")

from modeling.datasets import EmbeddingsDataset


def generate_constants(cfg: DictConfig):
    rqvae_split_name = (
        f"{cfg.train.rqvae_train_parts[0]}-{cfg.train.rqvae_train_parts[1]}TR_"
        f"{cfg.train.rqvae_eval_parts[0]}-{cfg.train.rqvae_eval_parts[1]}TE"
    )

    results_path = Path(cfg.paths.results_dir) / rqvae_split_name / "rqvae-collab"
    interactions_path = Path(cfg.paths.data_dir) / "all_data_interactions_with_groups.parquet"
    embeddings_path = Path(cfg.paths.data_dir) / "items_metadata_remapped.parquet"

    return {
        "INTERACTIONS_PATH": interactions_path,
        "EMBEDDINGS_PATH": embeddings_path,
        "ALL_MAPPING_PATH": results_path / "all_clusters_colisionless.json",
        "OUTPUT_TRAIN_MAPPING_PATH": results_path
        / (
            f"only_{cfg.inference.allowed_items_parts[0]}-"
            f"{cfg.inference.allowed_items_parts[1]}TR_clusters_colisionless.json"
        ),
    }


def rqvae_inference(cfg: DictConfig):
    consts = generate_constants(cfg)
    consts_str = json.dumps({k: str(v) for k, v in consts.items()}, indent=2)
    logger.info(f"Generated constants:\n{consts_str}")

    all_mapping_path = consts["ALL_MAPPING_PATH"]
    if not all_mapping_path.exists():
        logger.error(f"all_mapping not found at {all_mapping_path}")
        logger.error("Run rqvae training first to generate all_clusters_colisionless.json")
        return

    logger.info(f"Loading all_mapping from {all_mapping_path}")
    with open(all_mapping_path) as f:
        all_mapping = json.load(f)

    all_mapping = {int(k): v for k, v in all_mapping.items()}

    dataset = EmbeddingsDataset(
        all_interactions_path=consts["INTERACTIONS_PATH"],
        all_embeddings_path=consts["EMBEDDINGS_PATH"],
    )

    train_interactions = dataset.get_interactions_by_part(
        cfg.inference.allowed_items_parts[0], cfg.inference.allowed_items_parts[1]
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

    logger.info("Inference completed successfully!")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    rqvae_inference(cfg)


if __name__ == "__main__":
    main()
