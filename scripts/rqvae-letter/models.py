import torch
import torch.nn as nn


class Letter(nn.Module):
    def __init__(
            self,
            input_dim,
            num_codebooks,
            codebook_size,
            embedding_dim,
            beta=0.25,
            quant_loss_weight=1.0,
            cf_loss_weight=1.0,
            cf_embeddings=None
    ):
        super().__init__()
        self.register_buffer('beta', torch.tensor(beta))
        self.register_buffer('cf_embeddings', cf_embeddings.float())

        self.input_dim = input_dim
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.embedding_dim = embedding_dim
        self.quant_loss_weight = quant_loss_weight

        self.cf_loss_weight = cf_loss_weight

        self.codebooks = torch.nn.ParameterList()
        for _ in range(num_codebooks):
            cb = torch.FloatTensor(codebook_size, embedding_dim)
            self.codebooks.append(cb)

    @staticmethod
    def get_codebook_indices(remainder, codebook):
        dist = torch.cdist(remainder, codebook)
        return dist.argmin(dim=-1)

    def forward(self, inputs):
        item_ids = inputs['item_id']
        latent_vector = self.cf_embeddings[item_ids]

        latent_restored = 0
        rqvae_loss = 0
        clusters = []
        remainder = latent_vector
        for codebook in self.codebooks:
            codebook_indices = self.get_codebook_indices(remainder, codebook)
            clusters.append(codebook_indices)

            quantized = codebook[codebook_indices]
            codebook_vectors = remainder + (quantized - remainder).detach()

            rqvae_loss += self.beta * torch.nn.functional.mse_loss(remainder, quantized.detach())
            rqvae_loss += torch.nn.functional.mse_loss(quantized, remainder.detach())

            latent_restored += codebook_vectors
            remainder = remainder - codebook_vectors

        recon_loss = torch.nn.functional.mse_loss(latent_restored, latent_vector)

        loss = (recon_loss + self.quant_loss_weight * rqvae_loss).mean()

        clusters_counts = []
        for cluster in clusters:
            clusters_counts.append(torch.bincount(cluster, minlength=self.codebook_size))

        return loss, {
            'loss': loss.item(),
            'recon_loss': recon_loss.mean().item(),
            'rqvae_loss': rqvae_loss.mean().item(),

            'clusters_counts': clusters_counts,
            'clusters': torch.stack(clusters).T,
        }
