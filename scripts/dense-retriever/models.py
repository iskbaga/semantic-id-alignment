import torch
import torch.nn as nn


def create_masked_tensor(data, lengths, is_right_aligned=False, padding_value=0):
    batch_size = lengths.shape[0]
    max_seq_len = int(lengths.max().item())

    if data.dim() == 1:
        out_shape = (batch_size, max_seq_len)
    elif data.dim() == 2:
        out_shape = (batch_size, max_seq_len, data.size(-1))
    else:
        raise ValueError("data must be 1D (indices) or 2D (embeddings)")

    padded_tensor = torch.full(
        out_shape,
        padding_value,
        dtype=data.dtype,
        device=data.device,
    )
    positions = torch.arange(max_seq_len, device=data.device)
    mask = positions[None, :] < lengths[:, None]

    if is_right_aligned:
        mask = torch.flip(mask, dims=[-1])

    padded_tensor[mask] = data

    return padded_tensor, mask


class TransformerEncoder(nn.Module):
    def __init__(self, embedding_dim, layers, dim_feedforward, num_heads, dropout, activation, causal, prenorm=False):
        super().__init__()
        self.causal = causal
        layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            layer_norm_eps=1e-5,
            batch_first=True,
            norm_first=prenorm,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self._init_weights(initializer_range=0.02)

    @torch.no_grad()
    def _init_weights(self, initializer_range: float) -> None:
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
            else:
                raise ValueError(f"Unknown transformer weight: {key}")

    def forward(self, embeddings, lengths, max_seqlen):
        padded_embeddings, mask = create_masked_tensor(data=embeddings, lengths=lengths)

        if self.causal:
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                sz=mask.shape[-1], device=mask.device, dtype=mask.dtype
            )
        else:
            causal_mask = None

        padded_embeddings = self.encoder(
            padded_embeddings, mask=causal_mask, src_key_padding_mask=~mask, is_causal=self.causal
        )
        return padded_embeddings[mask]


class SASRecModel(nn.Module):
    def __init__(
        self,
        num_items,
        max_sequence_length,
        embedding_dim,
        num_heads,
        num_layers,
        dim_feedforward,
        activation,
        topk_k,
        dropout=0.0,
        initializer_range=0.02,
    ):
        super().__init__()
        self._num_items = num_items
        self._num_heads = num_heads
        self._embedding_dim = embedding_dim

        self._item_embeddings = nn.Embedding(num_embeddings=num_items, embedding_dim=embedding_dim)
        self._position_embeddings = nn.Embedding(num_embeddings=max_sequence_length, embedding_dim=embedding_dim)

        self._topk_k = topk_k

        self._encoder = TransformerEncoder(
            embedding_dim=embedding_dim,
            dim_feedforward=dim_feedforward,
            layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            activation=activation,
            causal=True,
        )

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
        all_sample_events = inputs["item.ids"]
        all_sample_lengths = inputs["item.length"]

        all_positive_sample_events = inputs["label.ids"]
        all_positive_sample_lengths = inputs["label.length"]

        max_seqlen = int(all_sample_lengths.max().item())

        embeddings = self._item_embeddings(all_sample_events)

        end_indices = all_sample_lengths.cumsum(dim=0)
        start_indices = end_indices - all_sample_lengths

        sample_indices = torch.arange(all_sample_lengths.shape[0], device=all_sample_lengths.device).repeat_interleave(
            all_sample_lengths
        )

        positions = (
            torch.arange(all_sample_events.shape[0], device=all_sample_events.device) - start_indices[sample_indices]
        )

        position_embeddings = self._position_embeddings(positions)

        embeddings = embeddings + position_embeddings

        all_sample_embeddings = self._encoder(embeddings=embeddings, lengths=all_sample_lengths, max_seqlen=max_seqlen)

        all_embeddings = self._item_embeddings.weight

        if self.training:
            if "num_train_items" in inputs:
                num_train_items = inputs["num_train_items"]
                train_items_mask = torch.zeros_like(all_positive_sample_events)

                starts = torch.cumsum(all_sample_lengths, dim=0) - all_sample_lengths
                ends = starts + all_sample_lengths
                train_starts = ends - torch.minimum(num_train_items, all_sample_lengths)

                ones = torch.ones_like(train_starts, dtype=train_items_mask.dtype)
                train_items_mask.scatter_add_(0, train_starts, ones)
                train_items_mask.scatter_add_(0, ends[:-1], -ones[:-1])

                train_items_mask = torch.cumsum(train_items_mask, dim=-1) > 0
                all_sample_embeddings = all_sample_embeddings[train_items_mask]
                all_positive_sample_events = all_positive_sample_events[train_items_mask]

            all_scores = torch.einsum("ad,nd->an", all_sample_embeddings, all_embeddings)

            positive_scores = torch.gather(input=all_scores, dim=1, index=all_positive_sample_events[..., None])[:, 0]

            negative_scores = torch.gather(
                input=all_scores,
                dim=1,
                index=torch.randint(
                    low=0,
                    high=all_scores.shape[1],
                    size=all_positive_sample_events.shape,
                    device=all_positive_sample_events.device,
                )[..., None],
            )[:, 0]

            with torch.autocast(device_type="cuda", enabled=False):
                loss = self._compute_loss(positive_scores.float(), negative_scores.float())

            return loss, {"loss": loss.detach()}
        else:
            loss = torch.as_tensor(0.0)
            metrics = {"loss": loss}

            offsets = torch.cumsum(all_sample_lengths, dim=-1)
            all_sample_embeddings = all_sample_embeddings[offsets - 1]

            all_scores = torch.einsum("ad,nd->an", all_sample_embeddings, all_embeddings)

            positive_items, _ = create_masked_tensor(
                data=all_positive_sample_events, lengths=all_positive_sample_lengths, padding_value=-1
            )

            _, topk_indices = torch.topk(all_scores, k=20, dim=-1, largest=True, sorted=True)

            all_hits = torch.eq(positive_items[:, None, :], topk_indices[:, :, None]).any(dim=-1)

            for k in [1, 5, 10, 20]:
                metrics[f"recall@{k}"] = recall(all_hits=all_hits, positive_lengths=all_positive_sample_lengths, k=k)

                metrics[f"ndcg@{k}"] = ndcg(all_hits=all_hits, positive_lengths=all_positive_sample_lengths, k=k)

            return loss, metrics

    def _compute_loss(self, positive_scores, negative_scores):
        assert positive_scores.shape[0] == negative_scores.shape[0]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            positive_scores, torch.ones_like(positive_scores)
        ) + torch.nn.functional.binary_cross_entropy_with_logits(negative_scores, torch.zeros_like(negative_scores))

        return loss


def recall(all_hits: torch.Tensor, positive_lengths: torch.Tensor, k: int) -> torch.Tensor:
    hits = all_hits[:, :k].float()
    num_positives_clamped = torch.clamp(positive_lengths, max=k).to(torch.long)
    recall = hits.sum(dim=-1) / num_positives_clamped

    return recall.mean()


def ndcg(all_hits: torch.Tensor, positive_lengths: torch.Tensor, k: int) -> torch.Tensor:
    hits = all_hits[:, :k].float()

    num_positives_clamped = torch.clamp(positive_lengths, max=k).to(torch.long)
    positions = torch.arange(1, k + 1, device=positive_lengths.device).float()
    discounts = 1.0 / torch.log2(positions + 1.0)

    dcg = (hits * discounts[None, :]).sum(dim=1)
    idcg = torch.cumsum(discounts, dim=0)[num_positives_clamped - 1]
    ndcg = dcg / idcg

    return ndcg.mean()
