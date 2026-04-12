import json
import sys
from pathlib import Path

import hydra
import torch
from loguru import logger
from models import SASRecModel
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data import SASRecEvalDataset, SASRecTrainDataset


sys.path.append("..")

from modeling.datasets import SequentialDataset
from modeling.training import TensorboardLogger
from modeling.utils import collate, fix_random_seed, run_evaluation


torch.set_float32_matmul_precision("high")
torch._dynamo.config.capture_scalar_outputs = True


def collate_fn(device):
    def _transform(batch):
        processed_batch = collate(batch)
        torch_batch = {
            key: torch.from_numpy(value).to(device, non_blocking=True) for key, value in processed_batch.items()
        }
        return torch_batch

    return _transform


def generate_constants(cfg: DictConfig):
    split_name = (
        f"{cfg.dataset.train_parts[0]}-{cfg.dataset.train_parts[1]}TR_"
        f"{cfg.dataset.eval_parts[0]}-{cfg.dataset.eval_parts[1]}TE"
    )

    results_path = Path(cfg.paths.results_dir) / split_name / "dense-retriever"
    Path(results_path).mkdir(parents=True, exist_ok=True)

    interactions_path = Path(cfg.paths.data_dir) / "all_data_interactions_with_groups.parquet"
    embeddings_path = Path(cfg.paths.data_dir) / "items_metadata_remapped.parquet"
    experiment_name = f"dense-retriever_{cfg.dataset.name}_{split_name}"

    return {
        "INTERACTIONS_PATH": interactions_path,
        "EMBEDDINGS_PATH": embeddings_path,
        "EXPERIMENT_NAME": experiment_name,
        "EMBEDDINGS_FILE": results_path / "embeddings.pt",
    }


def train_model(cfg: DictConfig):
    consts = generate_constants(cfg)
    consts_str = json.dumps({k: str(v) for k, v in consts.items()}, indent=2)
    logger.info(f"Generated constants:\n{consts_str}")
    fix_random_seed(cfg.training.seed_value)

    device = cfg.training.device if torch.cuda.is_available() and cfg.training.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    data = SequentialDataset(
        all_interactions_path=consts["INTERACTIONS_PATH"],
        all_embeddings_path=consts["EMBEDDINGS_PATH"],
        train_parts=cfg.dataset.train_parts,
        eval_parts=cfg.dataset.eval_parts,
        max_seq_len=cfg.model.max_seq_len,
    )

    train_dataset = SASRecTrainDataset(data.train_samples)
    eval_dataset = SASRecEvalDataset(data.eval_samples)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn(device),
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(device),
    )

    model = SASRecModel(
        num_items=data.num_items,
        max_sequence_length=cfg.model.max_seq_len,
        embedding_dim=cfg.model.embedding_dim,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        dim_feedforward=cfg.model.feedforward_dim,
        activation=cfg.model.activation,
        topk_k=cfg.model.top_k,
        dropout=cfg.model.dropout,
        initializer_range=0.02,
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
        eval_metrics = run_evaluation(model, eval_dataloader, "eval/")
        all_metrics = {**train_metrics, **eval_metrics}
        tensorboard_logger.add_metrics((epoch + 1) * (batch_idx + 1), all_metrics)

    tensorboard_logger.close()

    Path(cfg.paths.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    last_model_path = Path(cfg.paths.checkpoints_dir) / f"{consts['EXPERIMENT_NAME']}_last.pth"
    torch.save(model.state_dict(), last_model_path)
    logger.info(f"Last model saved to: {last_model_path}")
    logger.info("Training completed successfully!")

    if cfg.training.save_embeddings and cfg.model.num_layers == 2:
        torch.save(model._item_embeddings.weight.detach().cpu(), consts["EMBEDDINGS_FILE"])
        logger.info(f"Embeddings saved to {consts['EMBEDDINGS_FILE']}!")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    train_model(cfg)


if __name__ == "__main__":
    main()
