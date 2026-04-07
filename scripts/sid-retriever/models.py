import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, LogitsProcessor


class CorrectItemsLogitsProcessorGPT(LogitsProcessor):
    def __init__(self, num_codebooks, codebook_size, train_mapping, num_beams, device):
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.num_beams = num_beams
        self.device = device

        sem_ids = []
        for codes in train_mapping.values():
            assert len(codes) == num_codebooks
            sem_ids.append(codes)

        self.index_semantic_ids = torch.tensor(sem_ids, dtype=torch.long, device=self.device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_beams, seq_len = input_ids.shape
        allowed_mask = torch.zeros_like(scores, dtype=torch.bool, device=self.device)

        next_sid_codebook_num = (seq_len - 1) % self.num_codebooks  # -1 because of uid
        if next_sid_codebook_num == 0:
            first_tokens = self.index_semantic_ids[:, 0].unique()
            allowed_mask[:, first_tokens] = True
        else:
            prefix_len = next_sid_codebook_num
            current_prefix = input_ids[:, -prefix_len:]  # (batch_beams, seq_len)
            item_prefixes = self.index_semantic_ids[:, :prefix_len]  # (all_sequences, seq_len)

            matches = (current_prefix[:, None] == item_prefixes[None,]).all(dim=-1)  # (batch_beams, all_sequences)

            next_tokens = self.index_semantic_ids[:, prefix_len][None].expand(
                batch_beams, -1
            )  # (batch_beams, all_sequences)

            row_ids = torch.arange(batch_beams, device=self.device)[:, None].expand_as(
                next_tokens
            )  # (batch_beams, all_sequences)

            allowed_mask[row_ids[matches], next_tokens[matches]] = True

        start = next_sid_codebook_num * self.codebook_size
        end = (next_sid_codebook_num + 1) * self.codebook_size
        codebook_mask = torch.zeros_like(scores, dtype=torch.bool, device=self.device)
        codebook_mask[:, start:end] = True
        final_mask = allowed_mask & codebook_mask
        scores[~final_mask] = -torch.inf

        return scores


class TigerGptModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        codebook_size,
        sem_id_len,
        num_positions,
        user_ids_count,
        num_heads,
        num_layers,
        dim_feedforward,
        num_beams,
        num_return_sequences,
        layer_norm_eps=1e-6,
        activation="relu",
        dropout=0.1,
        initializer_range=0.02,
        logits_processor=None,
    ):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._codebook_size = codebook_size
        self._sem_id_len = sem_id_len
        self._num_positions = num_positions
        self.user_ids_count = user_ids_count
        self._num_heads = num_heads
        self._num_layers = num_layers
        self._dim_feedforward = dim_feedforward
        self._num_beams = num_beams
        self._num_return_sequences = num_return_sequences
        self._layer_norm_eps = layer_norm_eps
        self._activation = activation
        self._dropout = dropout
        self._initializer_range = initializer_range
        self.logits_processor = logits_processor

        self._unified_vocab_size = codebook_size * self._sem_id_len + self.user_ids_count + 10  # 10 for utilities

        self._bos_token_id = self._unified_vocab_size - 3
        self._eos_token_id = self._unified_vocab_size - 2
        self._pad_token_id = self._unified_vocab_size - 1

        self.config = GPT2Config(
            vocab_size=self._unified_vocab_size,
            n_positions=num_positions + 10,
            n_embd=embedding_dim * num_heads,
            n_layer=num_layers,
            n_head=num_heads,
            n_inner=dim_feedforward,
            activation_function=activation,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            layer_norm_epsilon=layer_norm_eps,
            initializer_range=initializer_range,
            scale_attn_weights=True,
            use_cache=True,
            bos_token_id=self._bos_token_id,
            eos_token_id=self._eos_token_id,
            pad_token_id=self._pad_token_id,
            tie_word_embeddings=False,
        )
        self.model = GPT2LMHeadModel(config=self.config)
        self._init_weights(initializer_range)

    @torch.no_grad()
    def _init_weights(self, initializer_range):
        for key, value in self.named_parameters():
            if "weight" in key:
                if "norm" in key:
                    nn.init.ones_(value.data)
                else:
                    nn.init.trunc_normal_(
                        value.data, std=initializer_range, a=-2 * initializer_range, b=2 * initializer_range
                    )
            elif "bias" in key:
                nn.init.zeros_(value.data)
            elif "codebook" in key or "bos_embedding" in key:
                nn.init.trunc_normal_(
                    value.data, std=initializer_range, a=-2 * initializer_range, b=2 * initializer_range
                )
            else:
                raise ValueError(f"Unknown transformer weight: {key}")

    def forward(self, inputs):
        input_semantic_ids = inputs["input.data"].long()
        input_mask = inputs["input.mask"].bool()

        input_ids = input_semantic_ids.clone()
        input_ids[~input_mask] = self._pad_token_id

        if self.training:
            labels = input_ids.clone()
            labels[labels == self._pad_token_id] = -100

            if "num_train_items" in inputs:
                num_unmasked = inputs["num_train_items"] * 4
                positions = torch.arange(labels.shape[1], device=labels.device, dtype=labels.dtype)[None].tile(
                    dims=[labels.shape[0], 1]
                )

                labels_mask = positions >= (labels.shape[1] - num_unmasked[:, None])
                labels[~labels_mask] = -100

            model_output = self.model(
                input_ids=input_ids,
                attention_mask=input_mask,
                labels=labels,
                use_cache=False,
            )

            loss = model_output["loss"]
            return loss, {"loss": loss.detach()}
        else:
            loss = torch.as_tensor(0.0)
            metrics = {"loss": loss}

            output = self.model.generate(
                input_ids=input_ids,
                attention_mask=input_mask,
                num_beams=self._num_beams,
                num_return_sequences=self._num_return_sequences,
                max_new_tokens=self._sem_id_len,
                eos_token_id=self.config.eos_token_id,
                pad_token_id=self.config.pad_token_id,
                do_sample=False,
                early_stopping=False,
                use_cache=True,
                logits_processor=[self.logits_processor] if self.logits_processor is not None else [],
            )[:, -self._sem_id_len :]

            predicted_sids = output.reshape(
                -1, self._num_return_sequences, self._sem_id_len
            )  # (batch_size, k, seq_len)

            positive_length = inputs["label.length"].float()
            positive_semantic_ids = inputs["label.semantic.padded"].long()

            positive_semantic_ids = positive_semantic_ids.reshape(
                positive_semantic_ids.shape[0], -1, self._sem_id_len
            )  # (batch_size, pos_num, seq_len)
            all_hits = (
                torch.eq(predicted_sids[:, :, None, :], positive_semantic_ids[:, None, :, :]).all(dim=-1).any(dim=-1)
            )  # (batch_size, k)

            for k in [1, 5, 10, 20]:
                metrics[f"recall@{k}"] = recall(all_hits=all_hits, positive_lengths=positive_length, k=k)

                metrics[f"ndcg@{k}"] = ndcg(all_hits=all_hits, positive_lengths=positive_length, k=k)

            return loss, metrics


def recall(all_hits: torch.Tensor, positive_lengths: torch.Tensor, k: int) -> torch.Tensor:
    hits = all_hits[:, :k].float()  # (batch_size, k)
    num_positives_clamped = torch.clamp(positive_lengths, max=k).to(torch.long)  # (batch_size)
    recall = hits.sum(dim=-1) / num_positives_clamped  # (batch_size)
    return recall.mean()


def ndcg(all_hits: torch.Tensor, positive_lengths: torch.Tensor, k: int) -> torch.Tensor:
    hits = all_hits[:, :k].float()  # (batch_size, k)

    num_positives_clamped = torch.clamp(positive_lengths, max=k).to(torch.long)  # (batch_size)
    positions = torch.arange(1, k + 1, device=positive_lengths.device).float()  # (k)
    discounts = 1.0 / torch.log2(positions + 1.0)  # (k)

    dcg = (hits * discounts[None, :]).sum(dim=1)  # (batch_size)
    idcg = torch.cumsum(discounts, dim=0)[num_positives_clamped - 1]  # (batch_size)
    ndcg = dcg / idcg  # (batch_size)

    return ndcg.mean()
