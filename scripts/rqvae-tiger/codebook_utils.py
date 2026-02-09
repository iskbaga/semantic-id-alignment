import torch


@torch.no_grad()
def codebook_initialize(model, dataloader):
    for i in range(len(model.codebooks)):
        X = next(iter(dataloader))['embedding']
        idx = torch.randperm(X.shape[0], device=X.device)[:len(model.codebooks[i])]
        remainder = model.encoder(X[idx])

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
        random_batch = next(iter(dataloader))['embedding']

        for batch in dataloader:
            remainder = model.encoder(batch['embedding'])
            for l in range(codebook_idx):
                ind = model.get_codebook_indices(remainder, model.codebooks[l])
                remainder = remainder - model.codebooks[l][ind]

            indices = model.get_codebook_indices(remainder, codebook)
            centroid_counts.scatter_add_(0, indices, torch.ones_like(indices))

        dead_mask = (centroid_counts == 0)
        num_dead = int(dead_mask.sum().item())
        num_fixed.append(num_dead)
        if num_dead == 0:
            continue

        remainder = model.encoder(random_batch)
        for l in range(codebook_idx):
            ind = model.get_codebook_indices(remainder, model.codebooks[l])
            remainder = remainder - model.codebooks[l][ind]
        remainder = remainder[torch.randperm(remainder.shape[0], device=codebook.device)][:num_dead]
        codebook[dead_mask] = remainder.detach()

    return num_fixed, centroid_counts.max().item()
