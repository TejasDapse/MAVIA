"""Greedy k-center coreset subsampling.

A category's training images yield on the order of 10^5 patch embeddings. Keeping
all of them would make the memory bank slow to search and large to ship. Random
subsampling is the obvious shortcut and the wrong one: it samples proportionally
to density, so it over-represents the redundant bulk of "ordinary" patches and
under-represents the rare-but-normal ones (an edge, a printed character, a
reflective corner) that are exactly where false positives come from.

Greedy k-center instead maximises coverage — it repeatedly selects the point
furthest from everything already chosen, which preserves the *extent* of the
normal manifold rather than its density. PatchCore reports that this retains
accuracy at 1% of the original bank size.

To keep the O(N*k) distance computations affordable, distances are measured in a
Johnson-Lindenstrauss random projection of the embedding space, which preserves
pairwise distances within a bounded distortion.
"""

from __future__ import annotations

import torch
from torch import Tensor


def johnson_lindenstrauss_projection(
    embeddings: Tensor, target_dim: int, generator: torch.Generator | None = None
) -> Tensor:
    """Project to ``target_dim`` with a scaled Gaussian random matrix."""
    n_features = embeddings.shape[1]
    if target_dim >= n_features:
        return embeddings
    matrix = torch.randn(
        n_features,
        target_dim,
        device=embeddings.device,
        dtype=embeddings.dtype,
        generator=generator,
    )
    return embeddings @ (matrix / (target_dim**0.5))


def greedy_coreset_indices(
    embeddings: Tensor,
    sampling_ratio: float = 0.01,
    projection_dim: int | None = 128,
    seed: int = 0,
) -> Tensor:
    """Select a coverage-maximising subset, returning indices into ``embeddings``.

    Args:
        embeddings: ``(N, C)`` patch embeddings.
        sampling_ratio: fraction of points to keep, in (0, 1].
        projection_dim: dimensionality for the distance computation; None disables.
        seed: makes selection reproducible, which the evaluation depends on.
    """
    if not 0.0 < sampling_ratio <= 1.0:
        raise ValueError(f"sampling_ratio must be in (0, 1], got {sampling_ratio}")

    n_points = embeddings.shape[0]
    n_select = max(1, round(n_points * sampling_ratio))
    if n_select >= n_points:
        return torch.arange(n_points, device=embeddings.device)

    generator = torch.Generator(device=embeddings.device).manual_seed(seed)
    working = embeddings
    if projection_dim is not None:
        working = johnson_lindenstrauss_projection(embeddings, projection_dim, generator)
    working = working.float()

    # Precomputing squared norms turns each distance update into a matrix-vector
    # product rather than a full pairwise expansion.
    sq_norms = (working**2).sum(dim=1)

    start = int(torch.randint(n_points, (1,), generator=generator, device=working.device).item())
    selected = torch.empty(n_select, dtype=torch.long, device=working.device)
    selected[0] = start

    # min_dist[i] = squared distance from point i to the nearest selected point.
    min_dist = sq_norms + sq_norms[start] - 2.0 * (working @ working[start])
    min_dist.clamp_(min=0)

    for step in range(1, n_select):
        newest = int(torch.argmax(min_dist).item())
        selected[step] = newest
        dist_to_newest = sq_norms + sq_norms[newest] - 2.0 * (working @ working[newest])
        torch.minimum(min_dist, dist_to_newest.clamp_(min=0), out=min_dist)

    return selected


def greedy_coreset(
    embeddings: Tensor,
    sampling_ratio: float = 0.01,
    projection_dim: int | None = 128,
    seed: int = 0,
) -> Tensor:
    """Convenience wrapper returning the selected embeddings themselves."""
    indices = greedy_coreset_indices(embeddings, sampling_ratio, projection_dim, seed)
    return embeddings[indices]
