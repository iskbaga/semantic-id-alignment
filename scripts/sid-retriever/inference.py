import json
import sys
from pathlib import Path

import hydra
import torch
from data import TigerEvalDataset, create_semantic_mapping_array, tiger_preprocess, to_masked
from loguru import logger
from models import CorrectItemsLogitsProcessorGPT, TigerGptModel
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


sys.path.append("..")

from modeling.datasets import SequentialDataset
from modeling.utils import collate, fix_random_seed


torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True


def create_collate_fn(num_codebooks, codebook_size, device):
    def collate_wrapper(batch):
        processed_batch = collate(batch)
        processed_batch = {
            key: torch.from_numpy(value).to(device, non_blocking=True) for key, value in processed_batch.items()
        }
        processed_batch = to_masked(processed_batch, "item.semantic", is_right_aligned=True)
        processed_batch = to_masked(processed_batch, "label.semantic", is_right_aligned=True)
        processed_batch = to_masked(processed_batch, "visited", is_right_aligned=True)
        processed_batch = tiger_preprocess(processed_batch, num_codebooks, codebook_size)

        return processed_batch

    return collate_wrapper


def generate_constants(cfg: DictConfig):
    pretrained_allowed_parts = cfg.train.allowed_items_parts or cfg.train.sid_retriever_train_parts
    pretrained_sid_retriever_split_name = (
        f"{cfg.train.sid_retriever_train_parts[0]}-{cfg.train.sid_retriever_train_parts[1]}TR_"
        f"{cfg.train.sid_retriever_val_parts[0]}-{cfg.train.sid_retriever_val_parts[1]}V_"
        f"{cfg.train.sid_retriever_test_parts[0]}-{cfg.train.sid_retriever_test_parts[1]}T_"
        f"items-{pretrained_allowed_parts[0]}-{pretrained_allowed_parts[1]}"
    )
    pretrained_rqvae_split_name = (
        f"{cfg.train.rqvae_train_parts[0]}-{cfg.train.rqvae_train_parts[1]}TR_"
        f"{cfg.train.rqvae_val_parts[0]}-{cfg.train.rqvae_val_parts[1]}V_"
        f"{cfg.train.rqvae_test_parts[0]}-{cfg.train.rqvae_test_parts[1]}T"
    )

    pretrained_name = (
        f"sid-retriever_{cfg.dataset.name}_{pretrained_sid_retriever_split_name}_{pretrained_rqvae_split_name}"
    )

    if cfg.inference.use_finetune_model:
        finetune_allowed_parts = (
            cfg.finetune.allowed_items_parts or cfg.train.allowed_items_parts or cfg.train.sid_retriever_train_parts
        )
        finetune_sid_retriever_split_name = (
            f"{cfg.finetune.sid_retriever_train_parts[0]}-{cfg.finetune.sid_retriever_train_parts[1]}TR_"
            f"{cfg.finetune.sid_retriever_val_parts[0]}-{cfg.finetune.sid_retriever_val_parts[1]}V_"
            f"{cfg.finetune.sid_retriever_test_parts[0]}-{cfg.finetune.sid_retriever_test_parts[1]}T_"
            f"items-{finetune_allowed_parts[0]}-{finetune_allowed_parts[1]}"
        )
        finetune_rqvae_split_name = (
            f"{cfg.finetune.rqvae_train_parts[0]}-{cfg.finetune.rqvae_train_parts[1]}TR_"
            f"{cfg.finetune.rqvae_val_parts[0]}-{cfg.finetune.rqvae_val_parts[1]}V_"
            f"{cfg.finetune.rqvae_test_parts[0]}-{cfg.finetune.rqvae_test_parts[1]}T"
        )

        assert cfg.finetune.matching_method in ["greedy", "hungarian", "none"]
        experiment_name = (
            f"{pretrained_name}_finetuned_"
            f"{cfg.finetune.sid_retriever_gap_parts[0]}-{cfg.finetune.sid_retriever_gap_parts[1]}G_"
            f"{cfg.finetune.matching_method}_"
            f"{finetune_sid_retriever_split_name}_"
            f"{finetune_rqvae_split_name}"
        )

        rqvae_results_path = (
            Path(cfg.paths.results_dir) / finetune_rqvae_split_name / f"rqvae-{cfg.dataset.rqvae.model_name}"
        )

        all_items_mapping_path_name = "all_clusters_colisionless.json"
        train_part_mapping_path_name = (
            f"only_{finetune_allowed_parts[0]}-{finetune_allowed_parts[1]}TR_clusters_colisionless.json"
        )

        if cfg.finetune.matching_method != "none":
            all_items_mapping_path_name = (
                f"{cfg.finetune.matching_method}_to_{pretrained_rqvae_split_name}_all_clusters_colisionless.json"
            )
            train_part_mapping_path_name = (
                f"{cfg.finetune.matching_method}_to_{pretrained_rqvae_split_name}_{train_part_mapping_path_name}"
            )
    else:
        experiment_name = pretrained_name
        rqvae_results_path = (
            Path(cfg.paths.results_dir) / pretrained_rqvae_split_name / f"rqvae-{cfg.dataset.rqvae.model_name}"
        )
        all_items_mapping_path_name = "all_clusters_colisionless.json"
        train_part_mapping_path_name = (
            f"only_{pretrained_allowed_parts[0]}-{pretrained_allowed_parts[1]}TR_clusters_colisionless.json"
        )

    return {
        "EXPERIMENT_NAME": experiment_name,
        "INTERACTIONS_PATH": Path(cfg.paths.data_dir) / "all_data_interactions_with_groups.parquet",
        "EMBEDDINGS_PATH": Path(cfg.paths.data_dir) / "items_metadata_remapped.parquet",
        "ALL_ITEMS_SEMANTIC_MAPPING_PATH": rqvae_results_path / all_items_mapping_path_name,
        "TRAIN_PART_SEMANTIC_MAPPING_PATH": rqvae_results_path / train_part_mapping_path_name,
        "PRETRAINED_MODEL_MASK": f"{experiment_name}_best_*.pth",
    }


def tiger_inference(cfg: DictConfig):
    consts = generate_constants(cfg)
    consts_str = json.dumps({k: str(v) for k, v in consts.items()}, indent=2)
    logger.info(f"Generated constants:\n{consts_str}")
    fix_random_seed(cfg.training.seed_value)

    device = cfg.training.device if torch.cuda.is_available() and cfg.training.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    model_files = list(Path(cfg.paths.checkpoints_dir).glob(consts["PRETRAINED_MODEL_MASK"]))
    assert len(model_files) == 1, f"Expected exactly one model file, found {len(model_files)}"
    pretrained_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading pre-trained model from: {pretrained_model_path}")
    logger.info(f"Eval parts interval: [{cfg.inference.test_parts[0]}, {cfg.inference.test_parts[1]})")
    logger.info(f"Semantic IDs train mapping path: {consts['TRAIN_PART_SEMANTIC_MAPPING_PATH']}")

    with open(consts["ALL_ITEMS_SEMANTIC_MAPPING_PATH"]) as f:
        all_mappings = json.load(f)
    with open(consts["TRAIN_PART_SEMANTIC_MAPPING_PATH"]) as f:
        train_mappings = json.load(f)

    all_semantics_mapping_array = create_semantic_mapping_array(all_mappings, cfg.model.num_codebooks)

    test_samples = SequentialDataset(
        all_interactions_path=consts["INTERACTIONS_PATH"],
        all_embeddings_path=consts["EMBEDDINGS_PATH"],
        train_parts=[0, cfg.inference.test_parts[0]],
        val_parts=[0, cfg.inference.test_parts[0]],
        test_parts=cfg.inference.test_parts,
        max_seq_len=cfg.model.max_seq_len,
    ).test_samples

    eval_dataset = TigerEvalDataset(
        test_samples, all_semantics_mapping_array, cfg.model.num_codebooks, cfg.model.num_user_hash
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=create_collate_fn(cfg.model.num_codebooks, cfg.model.codebook_size, device),
    )

    model = TigerGptModel(
        embedding_dim=cfg.model.embedding_dim,
        codebook_size=cfg.model.codebook_size,
        sem_id_len=cfg.model.num_codebooks,
        num_positions=cfg.model.num_codebooks * cfg.model.max_seq_len,
        user_ids_count=cfg.model.num_user_hash,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.feedforward_dim,
        num_beams=cfg.model.num_beams,
        num_return_sequences=cfg.model.top_k,
        activation=cfg.model.activation,
        dropout=cfg.model.dropout,
        layer_norm_eps=1e-6,
        initializer_range=0.02,
        logits_processor=CorrectItemsLogitsProcessorGPT(
            cfg.model.num_codebooks, cfg.model.codebook_size, train_mappings, cfg.model.num_beams, device
        ),
    ).to(device)

    state_dict = torch.load(pretrained_model_path)
    model.load_state_dict(state_dict)

    logger.info("Doing evaluation")

    model.eval()
    metrics_data = {}

    with torch.inference_mode():
        for batch in eval_dataloader:
            _, metrics = model(batch)
            for metric, value in metrics.items():
                if metric not in metrics_data:
                    metrics_data[metric] = []
                metrics_data[metric].append(value.item() if torch.is_tensor(value) else value)

    avg_metrics = {}
    for metric, values in metrics_data.items():
        avg_metrics[metric] = sum(values) / len(values) if values else 0.0

    logger.info("=" * 50)
    logger.info("INFERENCE RESULTS")
    logger.info("=" * 50)
    logger.info(f"Total test batches: {len(metrics_data['ndcg@20'])}")
    for metric, value in avg_metrics.items():
        logger.info(f"{metric}: {value:.4f}")
    logger.info("=" * 50)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    tiger_inference(cfg)


if __name__ == "__main__":
    main()
