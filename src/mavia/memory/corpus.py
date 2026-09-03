"""Build the defect-history corpus the retrieval agent searches.

The corpus is what gives MAVIA memory across batches. Each record is a past
inspection: what was seen, what it turned out to be, what was done about it.

Two design decisions carry the quality of this phase.

**1. Morphology comes from the real ground-truth masks, not from invention.**
For every (category, defect_type) the mask statistics - what fraction of the part
the defect covers, how many separate regions it forms, how elongated they are -
are measured from MVTec's own annotations. A `bottle/broken_large` record
therefore describes a large single region because that is what broken_large
actually looks like, and `carpet/thread` describes a thin elongated one. Only the
process narrative (root cause, corrective action) is synthesised, from
`knowledge.py`.

**2. Query and document text are produced by the same function.**
``describe_observation`` renders both the stored record and the live query. If
the two were phrased differently the embeddings would drift apart and retrieval
would degrade for reasons that have nothing to do with the defect. This symmetry
is why the retrieval numbers in EVALUATION.md hold up.

Critically, the stored text describes only what an inspector can *see*. The
defect type and root cause live in the payload, never in the embedded text -
otherwise retrieval would be scored on information the vision agent does not have
at query time, and the evaluation would be measuring nothing.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from mavia.memory.knowledge import KNOWLEDGE_BASE, DefectKnowledge
from mavia.schemas import utc_now

PRODUCTION_LINES = ("LINE-A", "LINE-B", "LINE-C")
SHIFTS = ("morning", "afternoon", "night")
OUTCOMES = (
    "Corrective action verified effective; defect rate returned to baseline",
    "Action applied; recurrence observed within two weeks, escalated to engineering",
    "Root cause confirmed by process data review; permanent countermeasure implemented",
    "Batch quarantined and reworked; supplier corrective action requested",
)


@dataclass(frozen=True)
class MorphologyStats:
    """Defect geometry measured from MVTec's ground-truth masks.

    ``samples`` holds the individual per-mask measurements, and it is what case
    generation draws from. Summary statistics are retained for reporting only.

    This distinction is not cosmetic. An earlier version generated cases by
    sampling a Gaussian fitted to ``mean_area_fraction`` / ``std_area_fraction``,
    which produced a corpus whose defect modes were *more separable than the real
    masks are* - real defect geometry is heavy-tailed and overlapping, a fitted
    Gaussian is neither. Retrieval then scored 0.498 precision@3 against a true
    morphology ceiling of 0.449: the evaluation was measuring the generator, not
    the retriever. Bootstrapping the real measurements removes that bias.
    """

    category: str
    defect_type: str
    n_samples: int
    mean_area_fraction: float
    std_area_fraction: float
    mean_region_count: float
    mean_elongation: float
    samples: tuple[tuple[float, int, float], ...] = ()


@dataclass(frozen=True)
class DefectCase:
    """One historical inspection record."""

    case_id: str
    category: str
    defect_type: str
    observation: str
    root_cause: str
    action_taken: str
    outcome: str
    process_step: str
    severity: str
    occurred_at: datetime
    production_line: str
    shift: str
    area_fraction: float
    region_count: int
    elongation: float = 1.0
    metadata: dict[str, str] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        payload = asdict(self)
        payload["occurred_at"] = self.occurred_at.isoformat()
        return payload


# --------------------------------------------------------------------- text


def _severity_band(area_fraction: float) -> str:
    if area_fraction < 0.005:
        return "minimal"
    if area_fraction < 0.02:
        return "small"
    if area_fraction < 0.08:
        return "moderate"
    return "extensive"


def _extent_band(region_count: int) -> str:
    if region_count <= 1:
        return "a single localised region"
    if region_count == 2:
        return "two separate regions"
    if region_count <= 4:
        return f"{region_count} separate regions"
    return f"{region_count} scattered regions"


def describe_observation(
    category: str,
    area_fraction: float,
    region_count: int,
    elongation: float | None = None,
) -> str:
    """Render an inspection observation. Used for BOTH stored records and queries.

    Deliberately contains no defect type and no root cause: at query time the
    vision agent knows only the product category and the geometry of what it
    found, so a stored record must be findable from exactly that much.
    """
    parts = [
        f"Visual inspection of a {category.replace('_', ' ')} unit.",
        f"Anomaly detected covering {area_fraction * 100:.2f}% of the inspected area",
        f"in {_extent_band(region_count)}.",
        f"Defect extent is {_severity_band(area_fraction)}.",
    ]
    if elongation is not None:
        shape = "elongated and linear" if elongation > 2.5 else "compact and rounded"
        parts.append(f"The affected region is {shape} in shape.")
    return " ".join(parts)


# --------------------------------------------------------- mask morphology


def _mask_statistics(mask: np.ndarray) -> tuple[float, int, float]:
    """Area fraction, connected-region count, and mean elongation for one mask."""
    binary = mask > 0
    area_fraction = float(binary.mean())
    labelled, count = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))

    elongations: list[float] = []
    for region_id in range(1, count + 1):
        rows, cols = np.where(labelled == region_id)
        height = rows.max() - rows.min() + 1
        width = cols.max() - cols.min() + 1
        long_side, short_side = max(height, width), max(1, min(height, width))
        elongations.append(long_side / short_side)

    return area_fraction, count, float(np.mean(elongations)) if elongations else 1.0


def compute_morphology(
    root: Path, category: str, defect_type: str, max_masks: int = 40
) -> MorphologyStats | None:
    """Measure real defect geometry from the ground-truth masks."""
    mask_dir = Path(root) / category / "ground_truth" / defect_type
    if not mask_dir.is_dir():
        return None

    masks = sorted(mask_dir.glob("*.png"))[:max_masks]
    if not masks:
        return None

    fractions, counts, elongations = [], [], []
    for path in masks:
        array = np.asarray(Image.open(path).convert("L"))
        fraction, count, elongation = _mask_statistics(array)
        fractions.append(fraction)
        counts.append(count)
        elongations.append(elongation)

    return MorphologyStats(
        category=category,
        defect_type=defect_type,
        n_samples=len(masks),
        mean_area_fraction=float(np.mean(fractions)),
        std_area_fraction=float(np.std(fractions)),
        mean_region_count=float(np.mean(counts)),
        mean_elongation=float(np.mean(elongations)),
        samples=tuple(zip(fractions, counts, elongations, strict=True)),
    )


# ------------------------------------------------------------ case sampling


def _pareto_choice(options: tuple[str, ...], rng: random.Random) -> tuple[str, int]:
    """Pick an option with a decreasing-probability bias.

    Real failure data is Pareto-shaped: a handful of causes account for most
    events. Sampling uniformly would produce a corpus in which every cause is
    equally common, which is both unrealistic and easier to retrieve from than it
    should be.
    """
    weights = [1.0 / (index + 1) ** 1.5 for index in range(len(options))]
    index = rng.choices(range(len(options)), weights=weights, k=1)[0]
    return options[index], index


def generate_cases_for_defect(
    category: str,
    defect_type: str,
    knowledge: DefectKnowledge,
    morphology: MorphologyStats | None,
    n_cases: int,
    rng: random.Random,
    now: datetime | None = None,
) -> list[DefectCase]:
    now = now or utc_now()
    cases: list[DefectCase] = []

    for index in range(n_cases):
        if morphology is not None and morphology.samples:
            # Bootstrap a real mask measurement, with only enough jitter to avoid
            # duplicate records. Sampling a fitted distribution instead would
            # make the corpus more separable than the real defects are.
            area_fraction, region_count, elongation = rng.choice(morphology.samples)
            area_fraction = float(np.clip(area_fraction * rng.uniform(0.92, 1.08), 1e-4, 0.6))
            elongation = elongation * rng.uniform(0.95, 1.05)
        elif morphology is not None:
            area_fraction = float(np.clip(morphology.mean_area_fraction, 1e-4, 0.6))
            region_count = max(1, round(morphology.mean_region_count))
            elongation = morphology.mean_elongation
        else:
            area_fraction = rng.uniform(0.002, 0.05)
            region_count = rng.randint(1, 3)
            elongation = None

        root_cause, cause_rank = _pareto_choice(knowledge.root_causes, rng)
        action = knowledge.actions[min(cause_rank, len(knowledge.actions) - 1)]

        cases.append(
            DefectCase(
                case_id=f"{category}-{defect_type}-{index:03d}",
                category=category,
                defect_type=defect_type,
                observation=describe_observation(category, area_fraction, region_count, elongation),
                root_cause=root_cause,
                action_taken=action,
                outcome=rng.choice(OUTCOMES),
                process_step=knowledge.process_step,
                severity=knowledge.severity.value,
                occurred_at=now - timedelta(days=rng.randint(1, 540), hours=rng.randint(0, 23)),
                production_line=rng.choice(PRODUCTION_LINES),
                shift=rng.choice(SHIFTS),
                area_fraction=round(area_fraction, 5),
                region_count=region_count,
                elongation=round(float(elongation) if elongation else 1.0, 3),
                metadata={"defect_description": knowledge.description},
            )
        )

    return cases


def split_morphology(
    stats: MorphologyStats, holdout_fraction: float = 0.3
) -> tuple[MorphologyStats, MorphologyStats]:
    """Partition the mask measurements into two disjoint pools.

    Evaluation requires this. If corpus records and evaluation queries can
    bootstrap the *same* underlying mask, a query is trivially matched by its own
    twin and precision is inflated - the retriever gets credit for finding a
    duplicate rather than a genuinely comparable case.
    """
    n_holdout = max(1, round(len(stats.samples) * holdout_fraction))
    index_samples = stats.samples[:-n_holdout] or stats.samples[:1]
    query_samples = stats.samples[-n_holdout:]
    return (
        replace(stats, samples=index_samples, n_samples=len(index_samples)),
        replace(stats, samples=query_samples, n_samples=len(query_samples)),
    )


def build_corpus(
    dataset_root: Path,
    cases_per_defect: int = 10,
    seed: int = 42,
    categories: list[str] | None = None,
    mask_subset: str = "all",
    holdout_fraction: float = 0.3,
) -> list[DefectCase]:
    """Generate the defect-history corpus.

    ``mask_subset`` selects which pool of real mask measurements to draw from:
    ``"all"`` for production use, or ``"index"`` / ``"query"`` for a leakage-free
    evaluation split.
    """
    if mask_subset not in {"all", "index", "query"}:
        raise ValueError(f"mask_subset must be all/index/query, got {mask_subset!r}")

    rng = random.Random(seed)
    now = utc_now()
    corpus: list[DefectCase] = []

    for (category, defect_type), knowledge in sorted(KNOWLEDGE_BASE.items()):
        if categories and category not in categories:
            continue
        morphology = compute_morphology(dataset_root, category, defect_type)

        if morphology is not None and mask_subset != "all":
            index_stats, query_stats = split_morphology(morphology, holdout_fraction)
            morphology = index_stats if mask_subset == "index" else query_stats

        corpus.extend(
            generate_cases_for_defect(
                category, defect_type, knowledge, morphology, cases_per_defect, rng, now
            )
        )

    return corpus
