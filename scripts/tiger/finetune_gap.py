import glob
import os
import sys

sys.path.append('..')

import json
import torch
from loguru import logger
from torch.utils.data import DataLoader

import hydra
from omegaconf import DictConfig, OmegaConf
from data import TigerTrainDataset, TigerEvalDataset, tiger_preprocess, to_masked, create_semantic_mapping_array
from models import TigerGptModel, CorrectItemsLogitsProcessorGPT
from modeling.datasets import FinetuneDataset
from modeling.training import TensorboardLogger, EarlyStopper
from modeling.utils import collate, fix_random_seed, run_evaluation

torch.set_float32_matmul_precision('high')
torch._dynamo.config.capture_scalar_outputs = True
torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True


def create_collate_fn(num_codebooks, codebook_size, device, is_eval):
    def collate_wrapper(batch):
        processed_batch = collate(batch)
        processed_batch = {key: torch.from_numpy(value).to(device, non_blocking=True)
                           for key, value in processed_batch.items()}
        processed_batch = to_masked(processed_batch, 'item.semantic', is_right_aligned=True)
        if is_eval:
            processed_batch = to_masked(processed_batch, 'label.semantic', is_right_aligned=True)
            processed_batch = to_masked(processed_batch, 'visited', is_right_aligned=True)

        processed_batch = tiger_preprocess(processed_batch, num_codebooks, codebook_size)

        return processed_batch

    return collate_wrapper


def generate_constants(cfg: DictConfig):
    rqvae_split_name = f"{cfg.dataset.rqvae.train_parts[0]}-{cfg.dataset.rqvae.train_parts[1]}TR_" \
                       f"{cfg.dataset.rqvae.val_parts[0]}-{cfg.dataset.rqvae.val_parts[1]}V_" \
                       f"{cfg.dataset.rqvae.test_parts[0]}-{cfg.dataset.rqvae.test_parts[1]}T"
    finetune_rqvae_split_name = f"{cfg.finetune.rqvae.train_parts[0]}-{cfg.finetune.rqvae.train_parts[1]}TR_" \
                       f"{cfg.finetune.rqvae.val_parts[0]}-{cfg.finetune.rqvae.val_parts[1]}V_" \
                       f"{cfg.finetune.rqvae.test_parts[0]}-{cfg.finetune.rqvae.test_parts[1]}T"
    tiger_split_name = f"{cfg.dataset.tiger.train_parts[0]}-{cfg.dataset.tiger.train_parts[1]}TR_" \
                       f"{cfg.dataset.tiger.val_parts[0]}-{cfg.dataset.tiger.val_parts[1]}V_" \
                       f"{cfg.dataset.tiger.test_parts[0]}-{cfg.dataset.tiger.test_parts[1]}T"

    old_experiment_name = f"tiger_{cfg.dataset.rqvae.model_name}-{rqvae_split_name}_{cfg.dataset.name}_{tiger_split_name}"
    experiment_name = f'{old_experiment_name}_finetuned_on_{cfg.finetune.gap_parts[0]}-{cfg.finetune.gap_parts[1]}G_{finetune_rqvae_split_name}'
    rqvae_results_path = os.path.join(cfg.paths.results_dir, finetune_rqvae_split_name, f'rqvae-{cfg.dataset.rqvae.model_name}')

    all_items_mapping_path_name = f'all_clusters_colisionless.json'
    train_part_mapping_path_name = f'only_{cfg.dataset.tiger.train_parts[0]}-{cfg.dataset.tiger.train_parts[1]}TR_clusters_colisionless.json'

    assert cfg.finetune.matching_method in ['greedy', 'hungarian', 'none']
    if cfg.finetune.matching_method != 'none':
        experiment_name += f'_{cfg.finetune.matching_method}'
        all_items_mapping_path_name = f'{cfg.finetune.matching_method}_{all_items_mapping_path_name}'
        train_part_mapping_path_name = f'{cfg.finetune.matching_method}_{train_part_mapping_path_name}'

    return {
        'EXPERIMENT_NAME': experiment_name,
        'INTERACTIONS_PATH': os.path.join(cfg.paths.data_dir, 'all_data_interactions_with_groups.parquet'),
        'EMBEDDINGS_PATH': os.path.join(cfg.paths.data_dir, 'items_metadata_remapped.parquet'),
        'ALL_ITEMS_SEMANTIC_MAPPING_PATH': os.path.join(rqvae_results_path, all_items_mapping_path_name),
        'TRAIN_PART_SEMANTIC_MAPPING_PATH': os.path.join(rqvae_results_path, train_part_mapping_path_name),
        'PRETRAINED_MODEL_MASK': f'{old_experiment_name}_best_*.pth',
    }


def train_tiger_finetune(cfg: DictConfig):
    consts = generate_constants(cfg)
    logger.info(f'Generated constants: {consts}')
    fix_random_seed(cfg.training.seed_value)

    device = cfg.training.device if torch.cuda.is_available() and cfg.training.device != "cpu" else "cpu"
    logger.info(f"Using device: {device}")

    model_files = glob.glob(os.path.join(cfg.paths.checkpoints_dir, consts['PRETRAINED_MODEL_MASK']))
    assert len(model_files) == 1, f'Expected exactly one model file, found {len(model_files)}'
    pretrained_model_path = max(model_files, key=os.path.getmtime)
    logger.info(f'Loading pre-trained model from: {pretrained_model_path}')
    logger.info(f"Semantic IDs train mapping path: {consts['TRAIN_PART_SEMANTIC_MAPPING_PATH']}")

    with open(consts['ALL_ITEMS_SEMANTIC_MAPPING_PATH'], 'r') as f:
        all_mappings = json.load(f)
    with open(consts['TRAIN_PART_SEMANTIC_MAPPING_PATH'], 'r') as f:
        train_mappings = json.load(f)

    all_semantics_mapping_array = create_semantic_mapping_array(all_mappings, cfg.model.num_codebooks)

    data = FinetuneDataset(
        all_interactions_path=consts['INTERACTIONS_PATH'],
        all_embeddings_path=consts['EMBEDDINGS_PATH'],
        train_parts=cfg.dataset.tiger.train_parts,
        gap_parts=cfg.finetune.gap_parts,
        val_parts=cfg.dataset.tiger.val_parts,
        test_parts=cfg.finetune.test_parts,
        max_seq_len=cfg.model.max_seq_len,
    )

    train_dataset = TigerTrainDataset(
        data.train_samples,
        all_semantics_mapping_array,
        cfg.model.num_codebooks,
        cfg.model.num_user_hash
    )

    eval_dataset = TigerEvalDataset(
        data.test_samples,
        all_semantics_mapping_array,
        cfg.model.num_codebooks,
        cfg.model.num_user_hash
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=cfg.training.train_batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=create_collate_fn(
            cfg.model.num_codebooks,
            cfg.model.codebook_size,
            device,
            is_eval=False
        )
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=cfg.training.valid_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=create_collate_fn(
            cfg.model.num_codebooks,
            cfg.model.codebook_size,
            device,
            is_eval=True
        )
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
            cfg.model.num_codebooks,
            cfg.model.codebook_size,
            train_mappings,
            cfg.model.num_beams,
            device
        )
    ).to(device)

    state_dict = torch.load(pretrained_model_path)
    model.load_state_dict(state_dict)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.debug(f'Overall parameters: {total_params:,}')
    logger.debug(f'Trainable parameters: {trainable_params:,}')

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
    )

    tensorboard_logger = TensorboardLogger(
        experiment_name=consts['EXPERIMENT_NAME'],
        logdir=cfg.paths.tensorboard_dir
    )

    early_stopper = EarlyStopper(
        metric=cfg.training.metric,
        patience=cfg.training.patience,
        minimize=cfg.training.minimize_metric,
        checkpoints_dir=cfg.paths.checkpoints_dir,
        experiment_name=consts['EXPERIMENT_NAME']
    )

    logger.debug('Everything is ready for fine-tuning process!')

    for epoch in range(cfg.training.num_epochs):
        logger.debug(f'Starting epoch: {epoch + 1}')
        model.train()

        losses = []
        for batch_idx, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            loss, outputs = model(batch)

            loss.backward()
            optimizer.step()

            losses.append(outputs['loss'].item())

        all_metrics = {f'train/loss': sum(losses) / len(losses)}

        if (epoch + 1) % 2 == 0:
            logger.info('Doing evaluation')

            eval_metrics = run_evaluation(model, eval_dataloader, 'eval/')
            all_metrics.update(eval_metrics)

            tensorboard_logger.add_metrics((epoch + 1) * (batch_idx + 1), all_metrics)

            if early_stopper.check(all_metrics[cfg.training.metric], model):
                logger.info('Early stopping triggered')
                break
        else:
            tensorboard_logger.add_metrics((epoch + 1) * (batch_idx + 1), all_metrics)

    tensorboard_logger.close()

    best_model_file = early_stopper.get_best_model_path()
    logger.info(f'Best model path is: {best_model_file}')
    logger.info('Fine-tuning completed successfully!')


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    logger.info(OmegaConf.to_yaml(cfg, resolve=True))
    train_tiger_finetune(cfg)


if __name__ == '__main__':
    main()
