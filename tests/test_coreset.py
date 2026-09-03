"""Tests for greedy k-center coreset selection.

The property that matters is *coverage*: the coreset must span the full extent of
the normal manifold, including sparse regions. Random sampling does not, and that
difference is the whole reason PatchCore uses k-center. The tests below assert
the coverage behaviour directly rather than just checking shapes.
"""

from __future__ import annotations

import torch

from mavia.vision.coreset import (
    greedy_coreset,
    greedy_coreset_indices,
    johnson_lindenstrauss_projection,
)


def _three_clusters(per_cluster: int = 200, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Three tight, well-separated clusters of very different population sizes."""
    generator = torch.Generator().manual_seed(seed)
    centres = torch.tensor([[0.0, 0.0], [50.0, 0.0], [0.0, 50.0]])
    counts = [per_cluster * 10, per_cluster, 5]  # deliberately imbalanced
    points, labels = [], []
    for cluster_id, (centre, count) in enumerate(zip(centres, counts, strict=True)):
        noise = torch.randn(count, 2, generator=generator) * 0.5
        points.append(centre + noise)
        labels.append(torch.full((count,), cluster_id))
    return torch.cat(points), torch.cat(labels)


def test_selects_requested_number_of_points() -> None:
    embeddings = torch.randn(1000, 32)
    indices = greedy_coreset_indices(embeddings, sampling_ratio=0.05, projection_dim=None)
    assert indices.shape[0] == 50
    assert indices.unique().numel() == 50, "coreset must not select duplicates"


def test_is_deterministic_for_a_given_seed() -> None:
    embeddings = torch.randn(500, 16)
    first = greedy_coreset_indices(embeddings, sampling_ratio=0.1, seed=42)
    second = greedy_coreset_indices(embeddings, sampling_ratio=0.1, seed=42)
    assert torch.equal(first, second)


def test_ratio_of_one_returns_everything() -> None:
    embeddings = torch.randn(37, 8)
    indices = greedy_coreset_indices(embeddings, sampling_ratio=1.0)
    assert indices.shape[0] == 37


def test_rejects_invalid_ratio() -> None:
    embeddings = torch.randn(10, 4)
    for bad in (0.0, -0.1, 1.5):
        try:
            greedy_coreset_indices(embeddings, sampling_ratio=bad)
        except ValueError:
            continue
        raise AssertionError(f"sampling_ratio={bad} should have raised")


def test_covers_every_cluster_including_the_tiny_one() -> None:
    """The core guarantee: a 5-point cluster among 2000 points is not dropped.

    In manufacturing terms that tiny cluster is a rare-but-normal appearance - a
    printed marking, a reflective edge. Losing it from the memory bank turns it
    into a false defect at inference.
    """
    embeddings, labels = _three_clusters()
    indices = greedy_coreset_indices(embeddings, sampling_ratio=0.01, projection_dim=None, seed=0)
    selected_clusters = set(labels[indices].tolist())
    assert selected_clusters == {0, 1, 2}


def test_beats_random_sampling_on_coverage() -> None:
    """Explicitly demonstrate why k-center is used instead of random subsampling."""
    embeddings, labels = _three_clusters()
    n_select = max(1, int(0.01 * embeddings.shape[0]))

    coreset_clusters = set(labels[greedy_coreset_indices(embeddings, 0.01, None, seed=0)].tolist())

    # Random sampling, averaged over many draws, usually misses the rare cluster.
    generator = torch.Generator().manual_seed(0)
    misses = 0
    trials = 50
    for _ in range(trials):
        random_idx = torch.randperm(embeddings.shape[0], generator=generator)[:n_select]
        if set(labels[random_idx].tolist()) != {0, 1, 2}:
            misses += 1

    assert coreset_clusters == {0, 1, 2}
    assert misses > trials * 0.5, "random sampling was expected to miss the rare cluster often"


def test_projection_preserves_distances_approximately() -> None:
    """Johnson-Lindenstrauss: distances survive projection within a bounded distortion."""
    generator = torch.Generator().manual_seed(0)
    embeddings = torch.randn(200, 512, generator=generator)
    projected = johnson_lindenstrauss_projection(embeddings, target_dim=128, generator=generator)

    original = torch.cdist(embeddings[:50], embeddings[:50])
    reduced = torch.cdist(projected[:50], projected[:50])

    mask = original > 0
    ratio = (reduced[mask] / original[mask]).mean().item()
    assert 0.8 < ratio < 1.2, f"mean distance ratio {ratio:.3f} outside acceptable distortion"


def test_projection_is_a_noop_when_target_exceeds_dimensionality() -> None:
    embeddings = torch.randn(20, 16)
    assert torch.equal(johnson_lindenstrauss_projection(embeddings, target_dim=32), embeddings)


def test_greedy_coreset_returns_embeddings_matching_indices() -> None:
    embeddings = torch.randn(300, 12)
    indices = greedy_coreset_indices(embeddings, sampling_ratio=0.1, seed=7)
    subset = greedy_coreset(embeddings, sampling_ratio=0.1, seed=7)
    assert torch.equal(subset, embeddings[indices])
