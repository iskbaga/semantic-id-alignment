import torch


@torch.no_grad()
def codebook_initialize(model, dataloader):
    for i in range(len(model.codebooks)):
        x = model.cf_embeddings[next(iter(dataloader))["item_id"]]
        idx = torch.randperm(x.shape[0], device=x.device)[: len(model.codebooks[i])]
        remainder = x[idx]

        for j in range(i):
            codebook_indices = model.get_codebook_indices(remainder, model.codebooks[j])
            codebook_vectors = model.codebooks[j][codebook_indices]
            remainder = remainder - codebook_vectors

        model.codebooks[i].data = remainder.detach()


@torch.no_grad()
def fix_dead_codebooks(model, dataloader):
    num_fixed = []
    for codebook_idx, codebook in enumerate(model.codebooks):
        centroid_counts = torch.zeros(codebook.shape[0], dtype=torch.long, device=codebook.device)
        random_batch = model.cf_embeddings[next(iter(dataloader))["item_id"]]

        for batch in dataloader:
            remainder = model.cf_embeddings[batch["item_id"]]
            for i in range(codebook_idx):
                ind = model.get_codebook_indices(remainder, model.codebooks[i])
                remainder = remainder - model.codebooks[i][ind]

            indices = model.get_codebook_indices(remainder, codebook)
            centroid_counts.scatter_add_(0, indices, torch.ones_like(indices))

        dead_mask = centroid_counts == 0
        num_dead = int(dead_mask.sum().item())
        num_fixed.append(num_dead)
        if num_dead == 0:
            continue

        remainder = random_batch
        for i in range(codebook_idx):
            ind = model.get_codebook_indices(remainder, model.codebooks[i])
            remainder = remainder - model.codebooks[i][ind]
        remainder = remainder[torch.randperm(remainder.shape[0], device=codebook.device)][:num_dead]
        codebook[dead_mask] = remainder.detach()

    return num_fixed, centroid_counts.max().item()
