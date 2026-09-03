# MAVIA — Evaluation

Every number here is reproducible from the repo. No figure is quoted without the
command that produced it.

## Method summary

| Layer | Metrics | Where |
|---|---|---|
| Detection | Image AUROC, pixel AUROC, PRO, latency | `scripts/train_vision.py` |
| Retrieval | precision@3, recall@3, MRR | Phase 3 |
| Analysis | Citation grounding rate, risk-level agreement | Phase 4 |
| End-to-end | Per-stage latency, throughput, retrieval ablation | Phase 5+ |
| Governance | Chain integrity, audit completeness | `mavia audit verify`, `tests/test_audit.py` |

---

## 1. Detection — MVTec AD, all 15 categories

**Reproduce:** `uv run python scripts/train_vision.py`

PatchCore, WideResNet-50-2 (`layer2`+`layer3`), 1% greedy k-center coreset,
256→224 centre crop. Apple M-series GPU (MPS). No gradients are computed; "fit"
means embedding the defect-free training images and selecting the memory bank.

| Category | Image AUROC | Pixel AUROC | PRO | ms/image |
|---|---|---|---|---|
| bottle | 1.0000 | 0.9829 | 0.9219 | 35.0 |
| cable | 0.9978 | 0.9842 | 0.8919 | 39.2 |
| capsule | 0.9761 | 0.9863 | 0.7852 | 36.8 |
| carpet | 0.9904 | 0.9879 | 0.7952 | 42.1 |
| grid | 0.9708 | 0.9687 | 0.7980 | 31.5 |
| hazelnut | 1.0000 | 0.9841 | 0.7988 | 40.0 |
| leather | 1.0000 | 0.9906 | 0.7474 | 38.5 |
| metal_nut | 0.9976 | 0.9854 | 0.8246 | 29.1 |
| pill | 0.9594 | 0.9745 | 0.8943 | 31.7 |
| screw | 0.9332 | 0.9719 | 0.7815 | 31.6 |
| tile | 0.9906 | 0.9555 | 0.7288 | 34.4 |
| toothbrush | 1.0000 | 0.9843 | 0.8129 | 34.9 |
| transistor | 1.0000 | 0.9715 | 0.7039 | 37.5 |
| wood | 0.9895 | 0.9393 | 0.7487 | 39.9 |
| zipper | 0.9966 | 0.9829 | 0.8353 | 30.2 |
| **Mean** | **0.9868** | **0.9767** | **0.8046** | **35.5** |

### Comparison against the published baseline

| | This implementation | Roth et al. 2022 (PatchCore-1%) |
|---|---|---|
| Mean image AUROC | 0.9868 | 0.991 |
| Mean pixel AUROC | 0.9767 | 0.981 |

Within ~0.4 points of the paper on both metrics, which is the expected margin for
a from-scratch implementation without the paper's test-time augmentation and
multi-scale ensembling. Five categories reach a perfect 1.0000 image AUROC.

**Where it is weakest, and why.** `screw` (0.9332) is the clear outlier — the
paper reports ~0.983. Screws are photographed at arbitrary rotations, so a patch
in the memory bank rarely aligns with the corresponding patch in a query image,
and the defects (thread damage, tip scratches) are small and low-contrast.
Rotation augmentation of the memory bank is the standard remedy and is listed in
Future Work rather than quietly applied. `pill` (0.9594) suffers from a related
issue: legitimate colour speckling looks much like contamination.

**PRO is reported at 0.8046** but is not directly comparable to the paper's
~0.935, because PRO's integration limit and thresholding protocol vary between
implementations. It is used here for *relative* comparison across categories,
not as a claim against published work. Why PRO is tracked at all is shown below.

### Why PRO is reported alongside pixel AUROC

Measured on a synthetic case (`tests/test_metrics.py`): one image, a 400px defect
and a 16px defect. The model finds the large one and misses the small one.

| | PRO | Pixel AUROC |
|---|---|---|
| Both regions found | 0.946 | 1.000 |
| Small region missed | 0.493 | 0.978 |
| **Drop** | **0.452** | **0.022** |

Pixel AUROC falls 2 points while the system missed **half the defects on the
part**. Averaging over pixels lets one large defect mask several small misses;
PRO weights each connected region equally. On a production line a missed flaw is
a missed flaw regardless of its area, so PRO is the metric that reflects the
operational risk.

### Coreset selection: why greedy k-center, not random

Measured in `tests/test_coreset.py`. Three clusters of 2,000 / 200 / **5**
points, selecting 1%:

| Strategy | Rare 5-point cluster retained |
|---|---|
| Greedy k-center | Always |
| Random sampling | Missed in >50% of 50 trials |

Random sampling follows density, so it drops rare-but-normal appearances — a
printed marking, a reflective edge. Anything absent from the memory bank becomes
a *false defect* at inference. This is why the coreset algorithm is load-bearing
rather than an optimisation.

### Latency

35.5 ms/image mean on MPS, single image, no batching — ~28 units/second on a
laptop GPU. Dominated by the backbone forward pass, not the nearest-neighbour
search: a 1% coreset holds only ~470–3,000 entries per category.

---

## 2. Retrieval

_Phase 3._

## 3. Root-cause analysis

_Phase 4._

## 4. End-to-end

_Phase 5+._

## 5. Governance

| Property | Result |
|---|---|
| Payload tampering detected | ✅ `test_verify_chain_detects_payload_tampering` |
| Entry deletion detected | ✅ `test_verify_chain_detects_deleted_entry` |
| Chain resumes across process restart | ✅ `test_logger_resumes_chain_across_restarts` |
| Chain intact over a full pipeline run | Phase 5 |

---

## Test suite

48 tests: 42 unit (no dataset or model required, run in CI) and 6 integration
(need MVTec AD plus fitted memory banks, skipped automatically when absent).

```bash
make test                              # unit only
uv run pytest -m integration           # end-to-end, needs models
```
