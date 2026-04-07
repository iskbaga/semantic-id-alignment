import json
import sys
from pathlib import Path

import hydra
import torch
from data import SasRecEvalDataset, SasRecTrainDataset
from loguru import logger
from models import SasRecModel
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader


sys.path.append("..")

from modeling.datasets import FinetuneDataset
from modeling.training import EarlyStopper, TensorboardLogger
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
        f"{cfg.dataset.val_parts[0]}-{cfg.dataset.val_parts[1]}V_"
        f"{cfg.dataset.test_parts[0]}-{cfg.dataset.test_parts[1]}T"
    )
    old_experiment_name = f"dense-retriever_{cfg.dataset.name}_{split_name}"

    results_path = Path(cfg.paths.results_dir) / split_name / "dense-retriever"
    interactions_path = Path(cfg.paths.data_dir) / "all_data_interactions_with_groups.parquet"
    embeddings_path = Path(cfg.paths.data_dir) / "items_metadata_remapped.parquet"

    experiment_name = f"{old_experiment_name}_finetuned_on_{cfg.finetune.gap_parts[0]}-{cfg.finetune.gap_parts[1]}"
    previous_model_mask = f"{old_experiment_name}_best_*.pth"

    return {
        "SPLIT_NAME": split_name,
        "OLD_EXPERIMENT_NAME": old_experiment_name,
        "EXPERIMENT_NAME": experiment_name,
        "RESULTS_PATH": results_path,
        "INTERACTIONS_PATH": interactions_path,
        "EMBEDDINGS_PATH": embeddings_path,
        "PREVIOUS_MODEL_MASK": previous_model_mask,
        "EMBEDDINGS_FILE": results_path / "embeddings.pt",
    }


def finetune_sasrec(cfg: DictConfig):
    consts = generate_constants(cfg)
    consts_str = json.dumps({k: str(v) for k, v in consts.items()}, indent=2)
    logger.info(f"Generated constants:\n{consts_str}")
    fix_random_seed(cfg.training.seed_value)

    device = cfg.training.device if torch.cuda.is_available() and cfg.training.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    data = FinetuneDataset(
        all_interactions_path=consts["INTERACTIONS_PATH"],
        all_embeddings_path=consts["EMBEDDINGS_PATH"],
        train_parts=cfg.finetune.train_parts,
        gap_parts=cfg.finetune.gap_parts,
        val_parts=cfg.finetune.val_parts,
        test_parts=cfg.finetune.test_parts,
        max_seq_len=cfg.model.max_seq_len,
    )

    train_dataset = SasRecTrainDataset(data.train_samples)
    valid_dataset = SasRecEvalDataset(data.val_samples)
    eval_dataset = SasRecEvalDataset(data.test_samples)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn(device),
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(device),
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn(device),
    )

    model = SasRecModel(
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

    model_files = list(Path(cfg.paths.checkpoints_dir).glob(consts["PREVIOUS_MODEL_MASK"]))
    assert len(model_files) == 1, f"Expected exactly one model file, found {len(model_files)}"
    finetune_model_path = max(model_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"MODEL TO FINETUNE: {finetune_model_path}")
    state_dict = torch.load(finetune_model_path)
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k[len("module.") :] if k.startswith("module.") else k
        new_state_dict[new_k] = v
    model.load_state_dict(new_state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f"Overall parameters: {total_params:,}")
    logger.debug(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
    )

    consts["RESULTS_PATH"].mkdir(parents=True, exist_ok=True)

    tensorboard_logger = TensorboardLogger(experiment_name=consts["EXPERIMENT_NAME"], logdir=cfg.paths.tensorboard_dir)

    early_stopper = EarlyStopper(
        metric=cfg.training.metric,
        patience=cfg.training.patience,
        minimize=cfg.training.minimize_metric,
        checkpoints_dir=cfg.paths.checkpoints_dir,
        experiment_name=consts["EXPERIMENT_NAME"],
    )

    logger.debug("Everything is ready for finetuning process!")

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
        validation_metrics = run_evaluation(model, valid_dataloader, "validation/")
        eval_metrics = run_evaluation(model, eval_dataloader, "eval/")
        all_metrics = {**train_metrics, **validation_metrics, **eval_metrics}
        tensorboard_logger.add_metrics((epoch + 1) * (batch_idx + 1), all_metrics)

        if early_stopper.check(all_metrics[cfg.training.metric], model):
            logger.info("Early stopping triggered")
            break

    tensorboard_logger.close()

    logger.info("Finetuning completed successfully!")

    best_model_file = early_stopper.get_best_model_path()
    logger.info(f"Best model path is: {best_model_file}")

    if cfg.training.save_embeddings and cfg.model.num_layers == 2:
        state_dict = torch.load(best_model_file)
        model.load_state_dict(state_dict)
        torch.save(model._item_embeddings.weight.detach().cpu(), consts["EMBEDDINGS_FILE"])
        logger.info(f"Embeddings saved to {consts['EMBEDDINGS_FILE']}!")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    finetune_sasrec(cfg)


if __name__ == "__main__":
    main()
