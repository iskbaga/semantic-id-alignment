# Mitigating Collaborative Semantic ID Staleness in Generative Retrieval

This is the official implementation of the paper **"Mitigating Collaborative Semantic ID Staleness in Generative Retrieval"**.

The repository contains the complete source code and experimental assets required to reproduce all results reported in the paper, including:
- Data preprocessing notebooks for 3 benchmarks (Amazon Beauty, Yambda, VK-LSVD)
- Training code for **dense retriever**, **RQ‑VAE tokenizers** (content-based / collaborative), and **SID-based retriever**
- A **checkpoint-compatible Semantic ID (SID) refresh** via **token alignment** (Greedy / Hungarian)

This repo contains:
- data preparation notebooks for Amazon Beauty, Yambda, and VK-LSVD;
- training code for dense retriever, RQ-VAE tokenizers (content/collaborative), and SID-based retriever;
- SID refresh workflow with token alignment (greedy or Hungarian).

## Repository layout

```text
semantic-id-alignment/
├── data/
├── modeling/
├── notebooks/
├── scripts/
│   ├── dense-retriever/
│   ├── rqvae-content/
│   ├── rqvae-collab/
│   └── sid-retriever/
├── temp_files/
├── pyproject.toml
└── requirements.txt
```

## Setup

Requirements:
- Python 3.10+
- GPU recommended for training

Install environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
pip install -r requirements.txt
```

## Data preparation

Run one notebook per dataset:
- `notebooks/AmazonBeautyDownload.ipynb` -> `data/beauty/`
- `notebooks/YambdaDownload.ipynb` -> `data/yambda/`
- `notebooks/VkLsvdDownload.ipynb` -> `data/vk-lsvd/`

Expected artifacts (per dataset):
- `global_item_mapping.json` - mapping from original item ids to remapped integer ids
- `all_data_interactions_with_groups.parquet` - user-item interactions with temporal `part` column
- `items_metadata_remapped.parquet` - item metadata/embeddings aligned with remapped ids

## Temporal split convention

All experiments use chronological `part` values in `[0..9]`.

Ranges are half-open intervals `[start, end)`:
- `train_parts=[0,8]` -> parts `0..7`
- `val_parts=[8,9]` -> part `8`
- `test_parts=[9,10]` -> part `9`

Paper block `k` corresponds to code `part=k-1`.

## Quick run (main pipeline)

Commands below use `vk-lsvd` (default dataset in configs).

```bash
cd scripts
```

1) Train dense retriever (for collaborative embeddings):

```bash
python dense-retriever/train.py
```

2) Train tokenizer:

```bash
# Collaborative (uses dense retriever collaborative embeddings)
python rqvae-collab/train.py

# Content (uses content embeddings)
python rqvae-content/train.py
```

3) Train base SID retriever:

```bash
python sid-retriever/train.py
```

4) Fine-tune on gap (FT-old / stale SIDs):

```bash
python sid-retriever/finetune_gap.py finetune.matching_method='none'
```

## FT-new / FT-ours / Full

### FT-new

```bash
# retrain dense retriever for embeddings on [0,9)
python dense-retriever/train.py dataset.train_parts='[0,9]'

# retrain tokenizer on [0,9)
python rqvae-collab/train.py train.rqvae_train_parts='[0,9]'

# generate mapping for allowed_items=[0,8) from new RQVAE
python rqvae-collab/inference.py train.rqvae_train_parts='[0,9]' inference.allowed_items_parts='[0,8]'

# fine-tune retriever using refreshed SIDs
python sid-retriever/finetune_gap.py finetune.rqvae_train_parts='[0,9]' finetune.matching_method='none' finetune.allowed_items_parts='[0,8]'
```

### FT-ours

```bash
# retrain dense retriever for embeddings on [0,9)
python dense-retriever/train.py dataset.train_parts='[0,9]'

# retrain tokenizer on [0,9)
python rqvae-collab/train.py train.rqvae_train_parts='[0,9]'

# generate mapping for allowed_items=[0,8) from new RQVAE
python rqvae-collab/inference.py train.rqvae_train_parts='[0,9]' inference.allowed_items_parts='[0,8]'

# align SIDs
cd ../notebooks
jupyter notebook AlignSemanticIDs.ipynb

# fine-tune retriever on gap with aligned mapping
cd ../scripts
python sid-retriever/finetune_gap.py finetune.rqvae_train_parts='[0,9]' finetune.matching_method='hungarian' finetune.allowed_items_parts='[0,8]'
```

### Full retraining on [0,9)

```bash
# train RQVAE and retriever from scratch
python rqvae-collab/train.py train.rqvae_train_parts='[0,9]'
python rqvae-collab/inference.py train.rqvae_train_parts='[0,9]' inference.allowed_items_parts='[0,9]'
python sid-retriever/train.py train.sid_retriever_train_parts='[0,9]' train.allowed_items_parts='[0,9]' train.rqvae_train_parts='[0,9]'
```

**Note:** Full is the only experiment with `allowed_items_parts=[0,9]`. All other experiments use `allowed_items_parts=[0,8]`.

## Inference and logs

Run inference:

```bash
# Base model
python sid-retriever/inference.py inference.use_finetune_model='false'

# Finetuned model
python sid-retriever/inference.py inference.use_finetune_model='true'
```

Open TensorBoard:

```bash
tensorboard --logdir=temp_files/logs_<dataset>
```
