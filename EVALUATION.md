# MAVIA — Evaluation

Every number here is reproducible from the repo. No figure is quoted without the
command that produced it.

## Method summary

| Layer | Metrics | Where |
|---|---|---|
| Detection | Image AUROC, pixel AUROC, PRO, latency | `scripts/train_vision.py` |
| Retrieval | precision@3, hit-rate@3, MRR, vs. measured ceiling | `scripts/eval_retrieval.py`, `scripts/retrieval_ceiling.py` |
| Analysis | Citation grounding, risk agreement, retrieval ablation | `scripts/eval_analysis.py` |
| End-to-end | Interrupt durability, escalation, audit completeness | `tests/test_orchestrator.py`, `mavia` CLI |
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

## 2. Retrieval — defect-history corpus

**Reproduce:**
```bash
uv run python scripts/build_memory.py --recreate   # build + index 730 cases
uv run python scripts/retrieval_ceiling.py         # what is achievable at all
uv run python scripts/eval_retrieval.py            # what we achieve
```

730 historical cases across all 73 defect modes. Defect *morphology* is measured
from MVTec's real ground-truth masks; only the process narrative (root cause,
corrective action) is synthesised, from a 73-entry manufacturing knowledge base.

### 2.1 The ceiling comes first

The retriever sees only what the vision agent can measure: product category and
defect geometry. Before tuning anything, we measured what *any* retriever could
achieve on that information — leave-one-out k-NN on the real per-mask features:

| | precision@3 |
|---|---|
| Random within category | 0.217 |
| **Oracle k-NN on real mask geometry (ceiling)** | **0.449** |

The ceiling is 0.449, not 1.0, because several defect types of the same product
are genuinely indistinguishable by geometry. `bottle/broken_large` covers
11.7%±5.1 of the part and `bottle/contamination` covers 8.5%±5.5 — overlapping
distributions. No embedding model or re-ranker can separate them, because the
information is not present in what the vision agent measured.

Every number below should be read against 0.449.

### 2.2 Results

| Configuration | precision@3 | hit-rate@3 | MRR |
|---|---|---|---|
| Random baseline | 0.217 | — | — |
| Dense only, no category filter | 0.3936 | 0.6795 | 0.5333 |
| Dense only, category filter | 0.3936 | 0.6795 | 0.5333 |
| **Hybrid α=0.7 (dense + geometry)** | **0.4548** | **0.6904** | **0.5521** |
| Hybrid α=0.5 | 0.4530 | 0.6849 | 0.5502 |
| Hybrid α=0.3 | 0.4539 | 0.6822 | 0.5493 |
| Hybrid α=0.0 (geometry only) | 0.4539 | 0.6822 | 0.5507 |
| *Oracle ceiling* | *0.449* | *0.741* | — |

Hybrid retrieval reaches the ceiling: **0.4548 against 0.449**, i.e. essentially
all of the signal available in the observable features is being used. Dense
retrieval alone recovers only 88% of it.

### 2.3 Why hybrid re-ranking was needed

Sentence embeddings encode *magnitude* badly. "covering 0.31%" and "covering
11.70%" differ by one token and embed close together, though one is a speck and
the other a shattered part. The geometry term restores that signal by comparing
log-area, region count, and elongation directly.

The α sweep is informative in a way that is worth stating plainly: **α=0.0 (pure
geometry) performs as well as the hybrid.** The dense embedding contributes
almost nothing beyond separating product categories — unsurprising, since the
observation text is itself a rendering of the geometry. The honest conclusion is
that this is a *structured* retrieval problem wearing semantic clothing, and the
vector store earns its place through category separation and extensibility to
richer future text, not through semantic magic.

The category filter changes no metric, because the category name already sits in
the embedded text (same-category similarity 0.99 vs 0.70 cross-category). It is
retained as a hard guarantee against cross-product contamination rather than as
an accuracy measure.

### 2.4 Two measurement bugs found and fixed

Both inflated the score while looking perfectly healthy — recorded here because
the corrections matter more than the final number.

| Version | precision@3 | Why it was wrong |
|---|---|---|
| Gaussian-sampled corpus, case-level split | 0.498 | Cases were drawn from a Gaussian fitted per defect mode, making the synthetic history **more separable than real defects are**. Scored *above* the true ceiling. |
| Bootstrapped corpus, case-level split | 0.539 | Query and indexed cases could bootstrap the **same underlying mask**, so the retriever was rewarded for finding a duplicate of itself. |
| **Bootstrapped corpus, mask-level split** | **0.394 → 0.455 hybrid** | Honest. Disjoint mask pools, real measurements. |

The first version scored 0.498 against a ceiling of 0.449 — a result above the
information-theoretic limit, which is how the bug was caught. Computing the
ceiling before optimising is what made both failures visible; without it, 0.539
would have looked like success.

Both properties are now covered by tests
(`test_split_produces_disjoint_sample_pools`,
`test_generated_cases_bootstrap_real_measurements`).

### 2.5 Latency

| Stage | Time |
|---|---|
| Retrieval, warm | **9 ms** |
| First call (embedding model load) | ~7.9 s, one-off |

### 2.6 Honest limitation

The corpus is synthesised. Morphology is real — measured from MVTec's masks — but
no public dataset records why those specific samples failed, so root causes are
drawn from standard failure modes for the relevant processes. These metrics
therefore demonstrate that the retrieval *mechanism* works; they are not a claim
about field accuracy on a real production line.

## 3. Root-cause analysis

**Reproduce:** `uv run python scripts/eval_analysis.py --samples 25`

25 defective images sampled across all categories, run through the full
Vision → Retrieval → Analyst chain. Ground truth is the image's directory name,
which the pipeline never sees.

### 3.1 What is measured, and why these things

| Metric | Result (fallback path) | What it catches |
|---|---|---|
| Citation grounding rate | **1.000** | Citing a `case_id` that was never retrieved — the specific failure that makes "grounded in history" untrue |
| Share of analyses citing history | 1.000 | Silent ungrounded reasoning |
| Risk exact agreement | 0.480 | Whether the risk level matches the defect actually present |
| Risk within one level | 0.680 | Near-misses vs. wild disagreement |
| **Risk under-called** | **0.280** | The safety-relevant direction: calling a defect *less* serious than it is |
| Action specificity | 1.000 | "Investigate further" instead of a real intervention |
| Mean latency | 0.04 ms | — |

### 3.2 The retrieval ablation

The same 25 images analysed twice — once with retrieved history, once with the
history withheld:

| | With history | Without history |
|---|---|---|
| Mean confidence | **0.436** | **0.100** |
| Risk level changed | — | **76% of cases** |

Memory is load-bearing, not decorative. Removing it changes the risk decision on
three quarters of inspections and collapses confidence by 4×. This is the
measurement that justifies Phase 3 existing at all.

### 3.3 Honest reading of the risk numbers

**These numbers describe the deterministic fallback path, not Claude.** No
`ANTHROPIC_API_KEY` was configured when this was run, so the analyst used its
rule-based path: copy the nearest retrieved case's recorded severity. That makes
the ceiling explicit — fallback risk accuracy is bounded by retrieval accuracy:

| | |
|---|---|
| Retrieval top-1 correct | 0.440 |
| Risk exact agreement | 0.480 |

The two track each other because the fallback *is* the retrieval result. When the
retriever surfaces the wrong defect mode, the fallback inherits its severity. The
28% under-call rate is the number that would matter on a real line, and it is
reported rather than buried.

The LLM path is expected to beat this, because it can weigh the detection
evidence against several retrieved cases instead of copying the top one — but
that is a hypothesis until measured, and it is not claimed here as a result.
Re-run the same command with a key set to fill in the LLM column.

### 3.4 Guardrails in the LLM path

Three, all tested in `tests/test_analyst.py` with a stub client so they are
verified deterministically rather than against a live model:

1. **Structured output** via `client.messages.parse` into a Pydantic model.
   `risk_level` is an enum because the orchestrator branches on it — free text
   would put a regex in the control path.
2. **Citations verified, not trusted.** Every cited `case_id` is checked against
   what was actually retrieved. Invented ids are stripped, and a model that
   invents one has its confidence capped at 0.5 for the whole analysis.
3. **Degradation, not failure.** No key, or an API error, falls through to the
   deterministic path at explicitly reduced confidence, labelled in
   `model_name`. A QA line does not stop because a vendor is having an outage.

The system prompt is fixed across every inspection and marked
`cache_control: ephemeral`, so a line running thousands of units pays for it once
per cache window rather than once per part.

## 4. End-to-end orchestration

**Reproduce:**
```bash
uv run mavia inspect data/mvtec_ad/bottle/test/broken_large/001.png
uv run mavia pending
uv run mavia approve <inspection_id> --approver you@plant --rationale "..."
uv run mavia audit verify
```

### 4.1 The property that makes this an agent

A high-risk inspection calls LangGraph's `interrupt()`, which suspends the graph
at a checkpoint written to SQLite. Demonstrated across **three separate OS
processes**:

| Process | Command | Result |
|---|---|---|
| 1 | `mavia inspect …/broken_large/001.png` | Suspended — `risk_level=CRITICAL; anomaly_score=0.820 >= 0.75` |
| 2 | `mavia pending` | Listed the waiting inspection with its reason |
| 3 | `mavia approve insp_0738337ed95e --approver qa.lead@plant` | Resumed and completed |

Total latency 54,366 ms — that figure spans the human's thinking time, which is
the point. The process that started the inspection had exited before the
decision was made.

`tests/test_orchestrator.py::test_pending_approval_survives_losing_the_pipeline_object`
asserts this directly: the first pipeline object is deleted after suspending, and
a completely new one sharing only the checkpoint database resumes it.

An agent permitted to stop a production line must be interruptible, and the
interrupt must outlive the request that created it. That is the difference
between this and a script with an `input()` call in the middle.

### 4.2 Escalation policy

Two independent signals, OR-ed:

| Condition | Effect |
|---|---|
| `risk_level` ∈ {HIGH, CRITICAL} | Human review |
| `anomaly_score >= MAVIA_HIGH_RISK_THRESHOLD` (0.75) | Human review |
| Neither | Auto-proceed, still fully audited |

Deliberately conservative: the two can disagree, and a disagreement escalates
rather than resolves. A LOW-risk analysis on a 0.92 anomaly score still goes to a
human, and the reason string records which signal fired.

### 4.3 The gate fails closed

Ten resume payload shapes are tested. Anything not recognisably an approval —
`None`, `{}`, `42`, `"gibberish"` — is recorded as a **rejection**, and a
rejected inspection reports `INCONCLUSIVE` rather than a confirmed verdict. A
human-in-the-loop gate that fails open is not a gate.

### 4.4 Audit completeness

Auditing is structural: every node is wrapped by `_audited` before being added to
the graph, so no node is responsible for logging itself and none can forget to.

| Property | Result |
|---|---|
| Events per full inspection | 9 (start, 3× agent start/complete, approval, finalise) |
| Chain intact after full run + approval | ✅ 17 entries verified across 3 processes |
| Human decision recorded with approver and rationale | ✅ |
| Node failure recorded, chain still valid | ✅ (`camera offline` test) |
| Bulky fields excluded from hash payloads | ✅ (paths, pixel regions) |

A failing node produces an audited, degraded inspection — not a traceback.

### 4.5 Checkpoint serialisation

LangGraph's default serialiser deserialises arbitrary types and warns that this
will be blocked in a future release. MAVIA passes an explicit allowlist of its
own state types instead. This keeps checkpoints loadable across upgrades and
means a checkpoint database cannot smuggle an arbitrary class past the loader.

## 5. QA report generation

**Reproduce:** any `mavia inspect` run writes to `artifacts/reports/`.

Three formats per inspection: Markdown (repo, dashboard), styled HTML, and PDF
(audit binder). A real run produces ~3.9 KB of Markdown and a 40 KB PDF.

### 5.1 Templated, not generated

The report is rendered from a fixed template that quotes the Phase 4 analysis
fields verbatim. Handing the whole inspection to an LLM and asking for prose
would be easier and worse: an audited quality document needs the same sections in
the same order on every part, so a reviewer can find the verdict without reading
and two reports are diffable. Nothing in the template invents content, which is
also why the report cannot hallucinate.

### 5.2 The audit chain travels with the document

Every report embeds the full event sequence for its inspection plus the SHA-256
chain head at time of writing:

```
| Seq | Timestamp | Agent              | Action             | Entry hash          |
| 0   | 04:48:37  | pipeline           | inspection_started | 8b0c79a441f0f0b5... |
| 7   | 04:48:57  | approval_gate      | approval_approved  | 7f8264f8fb05dd38... |
| 8   | 04:48:57  | report_writer      | started            | 6bc70f07943c9be5... |

Chain head at time of writing: 6bc70f07943c9be538ff767c657f9e68...
```

A reader can re-derive the chain from the log and confirm the document describes
what the system actually did, rather than trusting the PDF. Writing the report is
itself audited, and the audit entry records the chain head it was bound to.

### 5.3 Ordering

The report node runs **after** the approval gate, not before. A QA record must
state what a human decided, not merely what the model proposed — so a suspended
inspection has no report until the decision is made.

### 5.4 Degradation and escaping

| Property | Behaviour |
|---|---|
| WeasyPrint/pango missing | Falls back to styled HTML, logs a warning, inspection completes |
| macOS Homebrew library path | Set programmatically before import, so `brew install pango` is sufficient with no env var |
| LLM text in the document | HTML-escaped — analysis text is model output and must not inject markup (tested) |

---

## 6. Dashboard

**Reproduce:** `make dashboard` → http://localhost:8501

### 6.1 Logic lives outside the UI

Streamlit re-executes its entire script on every widget interaction, which makes
UI code a poor place for logic and an impossible place to test. All of it lives
in `dashboard/service.py` — plain functions over the audit log — and `app.py` is
a rendering layer. 15 unit tests cover the service; the UI is covered by a
headless smoke test that executes all four views via `streamlit.testing.v1` and
asserts zero page exceptions.

That harness earned its place immediately: it caught that `use_container_width`
was deprecated past 2025-12-31 (silently still working, but on borrowed time).

### 6.2 The dashboard reads only the audit log

Inspection history is reconstructed **entirely** from audit events — verdict,
risk, approver, latency, errors, report id. Nothing else is consulted.

This is a deliberate test of the audit trail itself: if the log cannot rebuild
what happened, the log is not a complete record, and
`test_inspection_is_reconstructed_from_the_audit_log_alone` fails. Using a second
store would have hidden that weakness.

The checkpoint database remains the source of truth for what is still *waiting*,
because that is state, not history.

### 6.3 Chain status is surfaced, not buried

The sidebar shows live chain verification on every page load, and a broken chain
renders as a red error naming the failing sequence number rather than quietly
displaying stale data (`test_chain_status_reports_tampering`).

---

## 7. Packaging

Multi-stage Dockerfile: dependencies resolve from the lockfile in a builder
layer, and only the resulting virtualenv is copied into a slim runtime — the
image ships no uv, no build toolchain, no package cache. Runs as a non-root user
with a healthcheck on Streamlit's `/_stcore/health`.

System libraries are installed for the two things that need them: pango/cairo
for WeasyPrint PDFs, libgl/glib for OpenCV.

Dataset and fitted memory banks are mounted read-only rather than baked in.
MVTec AD is 4.9 GB and CC BY-NC-SA licensed, so redistributing it inside an image
would be both impractical and a licence breach.

### 7.1 Verified

Built and run on Apple Silicon (Docker Desktop 29.7.2, native `linux/aarch64` —
no Rosetta, since `torch 2.14.0+cpu` ships `manylinux_2_28_aarch64` wheels).

| Check | Result |
|---|---|
| Image size | **3.28 GB** (mostly PyTorch) |
| Build | Clean on first attempt |
| Container healthcheck | `healthy` within seconds |
| `mavia doctor` in container | Dataset, models, vector store, audit log all resolved |
| Full inspection in container | `leather/cut` → FAIL, score 0.709, 3 cases retrieved, 24.6 s cold |
| PDF rendering on Linux | **22 KB PDF produced** — the pango/cairo package set is correct |
| File ownership | `mavia:mavia` — the non-root user is real, not decorative |
| Audit chain across host + container | **45 entries verified intact** |

That last row is the interesting one. The audit log is a mounted volume, so the
hash chain spans entries written by the host CLI *and* entries written inside the
container, and `verify_chain` accepts the whole sequence. The governance property
holds across the process boundary, not just within one runtime.

Cold-start latency is 24.6 s because the container downloads WideResNet-50
weights on first use. `TORCH_HOME` and `HF_HOME` point into the mounted
`artifacts/` volume, so that cost is paid once rather than per container start.

---

## 8. Governance

| Property | Result |
|---|---|
| Payload tampering detected | ✅ `test_verify_chain_detects_payload_tampering` |
| Entry deletion detected | ✅ `test_verify_chain_detects_deleted_entry` |
| Chain resumes across process restart | ✅ `test_logger_resumes_chain_across_restarts` |
| Chain intact over a full pipeline run | Phase 5 |

---

## Test suite

61 unit tests (no dataset or model required, run in CI) plus 6 integration tests
(need MVTec AD plus fitted memory banks, skipped automatically when absent).

```bash
make test                              # unit only
uv run pytest -m integration           # end-to-end, needs models
```
