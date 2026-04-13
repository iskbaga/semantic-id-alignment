import json
import sys
from pathlib import Path

import hydra
import torch
from loguru import logger
from models import CorrectItemsLogitsProcessorGPT, TigerGptModel
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import TigerEvalDataset, TigerTrainDataset, create_semantic_mapping_array, tiger_preprocess, to_masked


sys.path.append("..")

from modeling.datasets import SequentialDataset
from modeling.training import TensorboardLogger
from modeling.utils import collate, fix_random_seed, run_evaluation


torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True


def create_collate_fn(num_codebooks, codebook_size, device, is_eval):
    def collate_wrapper(batch):
        processed_batch = collate(batch)
        processed_batch = {
            key: torch.from_numpy(value).to(device, non_blocking=True) for key, value in processed_batch.items()
        }
        processed_batch = to_masked(processed_batch, "item.semantic", is_right_aligned=True)
        if is_eval:
            processed_batch = to_masked(processed_batch, "label.semantic", is_right_aligned=True)
            processed_batch = to_masked(processed_batch, "visited", is_right_aligned=True)

        processed_batch = tiger_preprocess(processed_batch, num_codebooks, codebook_size)

        return processed_batch

    return collate_wrapper


def generate_constants(cfg: DictConfig):
    allowed_parts = cfg.train.allowed_items_parts or cfg.train.sid_retriever_train_parts
    sid_retriever_split_name = (
        f"{cfg.train.sid_retriever_train_parts[0]}-{cfg.train.sid_retriever_train_parts[1]}TR_"
        f"{cfg.train.sid_retriever_eval_parts[0]}-{cfg.train.sid_retriever_eval_parts[1]}TE_"
        f"items-{allowed_parts[0]}-{allowed_parts[1]}"
    )

    rqvae_split_name = (
        f"{cfg.train.rqvae_train_parts[0]}-{cfg.train.rqvae_train_parts[1]}TR_"
        f"{cfg.train.rqvae_eval_parts[0]}-{cfg.train.rqvae_eval_parts[1]}TE"
    )

    experiment_name = f"sid-retriever-{cfg.dataset.rqvae.model_name}_{cfg.dataset.name}_{sid_retriever_split_name}_{rqvae_split_name}"

    results_path = Path(cfg.paths.results_dir) / rqvae_split_name / f"rqvae-{cfg.dataset.rqvae.model_name}"

    return {
        "EXPERIMENT_NAME": experiment_name,
        "INTERACTIONS_PATH": Path(cfg.paths.data_dir) / "all_data_interactions_with_groups.parquet",
        "EMBEDDINGS_PATH": Path(cfg.paths.data_dir) / "items_metadata_remapped.parquet",
        "ALL_ITEMS_SEMANTIC_MAPPING_PATH": results_path / "all_clusters_colisionless.json",
        "TRAIN_PART_SEMANTIC_MAPPING_PATH": results_path
        / f"only_{allowed_parts[0]}-{allowed_parts[1]}TR_clusters_colisionless.json",
    }


def train_model(cfg: DictConfig):
    consts = generate_constants(cfg)
    consts_str = json.dumps({k: str(v) for k, v in consts.items()}, indent=2)
    logger.info(f"Generated constants:\n{consts_str}")
    fix_random_seed(cfg.training.seed_value)

    device = cfg.training.device if torch.cuda.is_available() and cfg.training.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    with open(consts["ALL_ITEMS_SEMANTIC_MAPPING_PATH"]) as f:
        all_mappings = json.load(f)
    with open(consts["TRAIN_PART_SEMANTIC_MAPPING_PATH"]) as f:
        train_mappings = json.load(f)

    all_semantics_mapping_array = create_semantic_mapping_array(all_mappings, cfg.model.num_codebooks)

    data = SequentialDataset(
        all_interactions_path=consts["INTERACTIONS_PATH"],
        all_embeddings_path=consts["EMBEDDINGS_PATH"],
        train_parts=cfg.train.sid_retriever_train_parts,
        eval_parts=cfg.train.sid_retriever_eval_parts,
        max_seq_len=cfg.model.max_seq_len,
    )

    train_dataset = TigerTrainDataset(
        data.train_samples, all_semantics_mapping_array, cfg.model.num_codebooks, cfg.model.num_user_hash
    )

    eval_dataset = TigerEvalDataset(
        data.eval_samples, all_semantics_mapping_array, cfg.model.num_codebooks, cfg.model.num_user_hash
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=create_collate_fn(cfg.model.num_codebooks, cfg.model.codebook_size, device, is_eval=False),
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=create_collate_fn(cfg.model.num_codebooks, cfg.model.codebook_size, device, is_eval=True),
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

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f"Overall parameters: {total_params:,}")
    logger.debug(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
    )

    tensorboard_logger = TensorboardLogger(experiment_name=consts["EXPERIMENT_NAME"], logdir=cfg.paths.tensorboard_dir)

    logger.debug("Everything is ready for training process!")

    for epoch in range(cfg.training.num_epochs):
        logger.debug(f"Starting epoch: {epoch + 1}")
        model.train()

        losses = []
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, outputs = model(batch)

            loss.backward()
            optimizer.step()

            losses.append(outputs["loss"].item())

        train_metrics = {"train/loss": sum(losses) / len(losses)}
        all_metrics = train_metrics.copy()

        if (epoch + 1) % 4 == 0:
            logger.info("Doing test evaluation")
            eval_metrics = run_evaluation(model, eval_dataloader, "eval/")
            all_metrics.update(eval_metrics)

            tensorboard_logger.add_metrics((epoch + 1) * (batch_idx + 1), all_metrics)
        else:
            tensorboard_logger.add_metrics((epoch + 1) * (batch_idx + 1), all_metrics)

    tensorboard_logger.close()

    Path(cfg.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    timestamp = tensorboard_logger.get_timestamp()
    last_model_path = Path(cfg.paths.checkpoints_dir) / f"{consts['EXPERIMENT_NAME']}_{timestamp}.pth"
    torch.save(model.state_dict(), last_model_path)
    logger.info(f"Last model saved to: {last_model_path}")
    logger.info("Training completed successfully!")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    train_model(cfg)


if __name__ == "__main__":
    main()
