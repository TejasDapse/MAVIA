# MAVIA — Multi-Agent Visual Inspection & Audit System

> A governed, explainable agentic QA system for manufacturing. It does not stop at
> "defect found" — it localises the anomaly, retrieves comparable historical cases,
> reasons about root cause with an LLM, escalates high-risk calls to a human, and
> writes every decision into a tamper-evident SHA-256 hash chain.

**Status:** Phase 6 of 8 complete — the full agent loop runs end to end and files a signed QA report with its audit chain embedded. See [Roadmap](#roadmap).

---

## Problem Statement

Automated visual inspection is mature at the detection layer — Landing AI, Overview AI,
Cognex and Keyence all ship models that find defects reliably. What blocks enterprise
deployment is everything *around* the model. Three gaps recur:

1. **No root-cause explanation.** Systems output pass/fail. A line engineer still has to
   work out *why* the defect appeared and what to change upstream.
2. **No memory across batches.** Each inspection is stateless. The system cannot say
   "this is the fourth contamination event on this line this week, and last time it was
   a coolant filter."
3. **No governance layer.** Decisions are not traceable, not attributable, and not
   verifiable after the fact — which makes them unusable in an audited quality process.

MAVIA is built specifically to close those three gaps on top of a state-of-the-art
detection backbone.

## Research Gap

| Existing systems | MAVIA |
|---|---|
| Detect defects | Detects, localises, **and explains probable root cause** |
| Stateless inspections | **Cross-batch memory** via vector retrieval over defect history |
| No verifiable record | **Tamper-evident hash-chain audit trail** (SHA-256, append-only) |
| Fully automatic or fully manual | **Risk-routed human-in-the-loop** approval gate |
| Black-box decision path | Every agent step traced, timed, and confidence-scored |

## Why anomaly detection, not YOLOv8

This is the central modelling decision, and it is deliberate.

MVTec AD is an **unsupervised anomaly detection** benchmark, not an object detection
dataset. Training images contain only defect-free products; the defective examples exist
solely in the test split, and per-defect bounding boxes are not provided (ground truth is
given as pixel-level segmentation masks). Fine-tuning a supervised detector like YOLOv8
on it means training on the test set — the result overstates accuracy and does not match
how the problem appears in a real factory, where you have thousands of good units and
almost no labelled examples of the failure you have not seen yet.

MAVIA therefore uses the approach the benchmark was designed for:

| Layer | Model | Why |
|---|---|---|
| Primary detector | **PatchCore**, implemented from scratch | Trains on defect-free images only; memory-bank approach needs no defect labels. Measured **0.9868** mean image AUROC here |
| Unseen categories | **CLIP zero-shot** | Handles product classes the memory bank has never seen, without retraining — covers a new SKU on day one |
| Localisation | Pixel anomaly maps → connected-component boxes | Gives the regions the report and dashboard cite as evidence |

This is a stronger engineering story than "I fine-tuned YOLO": it is the correct method
for the data, it reflects the real cold-start constraint in manufacturing, and it gives a
defensible answer when an interviewer asks why.

## Architecture

```
                        [ Product image ]
                                |
        +-----------------------v------------------------+
        | Agent 1 — Vision Inspector                     |
        |   PatchCore anomaly map + score                |
        |   CLIP zero-shot fallback for unseen classes   |
        |   -> verdict, anomaly_score, regions           |
        +-----------------------+------------------------+
                                |
        +-----------------------v------------------------+
        | Agent 2 — History Retriever                    |
        |   Qdrant kNN over embedded defect records      |
        |   -> top-k comparable cases + past root causes |
        +-----------------------+------------------------+
                                |
        +-----------------------v------------------------+
        | Agent 3 — Root Cause Analyst                   |
        |   Claude reasons over detection + history      |
        |   -> root_cause, action, risk_level, evidence  |
        +-----------------------+------------------------+
                                |
                   risk_level >= HIGH ?
                    /                    \
                 yes                      no
                  |                        |
        +---------v----------+             |
        | Human approval gate |            |
        | LangGraph interrupt |            |
        +---------+----------+             |
                  \                       /
        +----------v---------------------v---------------+
        | Agent 4 — Report Writer                        |
        |   Markdown + PDF QA report                     |
        +-----------------------+------------------------+
                                |
        +-----------------------v------------------------+
        | Audit trail — SHA-256 hash chain (append-only) |
        | every step: timestamp, agent, payload, hashes  |
        +------------------------------------------------+
```

Full design rationale: [ARCHITECTURE.md](ARCHITECTURE.md).

## Tech Stack

| Component | Choice |
|---|---|
| Anomaly detection | PatchCore, implemented from scratch on PyTorch |
| Zero-shot vision | open_clip (ViT-B-32) |
| Agent orchestration | LangGraph (stateful graph, `interrupt` for HITL, SQLite checkpointer) |
| LLM | Claude (`claude-opus-5`) via the Anthropic SDK |
| Vector memory | Qdrant (embedded on-disk by default, server/cloud optional) |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` |
| Observability | structlog + LangSmith traces |
| Governance | SHA-256 hash-chain audit log, verifiable via `mavia audit verify` |
| Dashboard | Streamlit |
| Tooling | uv, ruff, mypy (strict), pytest, GitHub Actions |
| Dataset | MVTec AD (15 categories, CC BY-NC-SA 4.0) |

## Quickstart

### Run an inspection

```bash
uv run mavia inspect data/mvtec_ad/bottle/test/broken_large/000.png
uv run mavia pending                      # what is awaiting a human
uv run mavia approve <id> --approver you@plant --rationale "confirmed"
uv run mavia audit verify                 # prove the trail is intact
```

Each completed inspection writes a QA report to `artifacts/reports/` as
Markdown, HTML and PDF, with the full SHA-256 audit chain embedded in the
document.

High-risk inspections **pause** and persist. The process can exit; approve
minutes later from a different process and the graph resumes where it stopped.

### Setup

```bash
# 1. Install (uv manages the pinned Python 3.11 itself)
make install          # core + dev tooling
make install-all      # everything: vision, memory, agents, report, dashboard, eval

# 2. Configure
cp .env.example .env  # add ANTHROPIC_API_KEY when you reach Phase 4

# 3. Check what is wired up
make doctor

# 4. Get the dataset (~4.9 GB)
make data

# 5. Run the tests
make test
```

Qdrant runs **embedded on disk** by default (`artifacts/qdrant_storage`), so no Docker is
required for development. Point `MAVIA_QDRANT_URL` at a server or Qdrant Cloud to switch.

## Repository layout

```
src/mavia/
  config.py          # pydantic-settings, single source of configuration
  schemas.py         # typed contracts every agent reads and writes
  audit.py           # SHA-256 hash-chain audit log + chain verification
  logging_setup.py   # structlog configuration
  cli.py             # `mavia doctor`, `mavia audit verify|show`
  vision/            # PatchCore, coreset, metrics, CLIP fallback, inspector
  memory/            # knowledge base, corpus, Qdrant store, retrieval agent
  agents/            # root cause analyst, report writer
  orchestrator/      # LangGraph graph, risk routing, HITL interrupt
  dashboard/         # Phase 7 — Streamlit app
scripts/             # dataset download, index seeding, training entrypoints
notebooks/           # EDA + Colab training notebooks
evaluation/          # detection / retrieval / end-to-end benchmarks
tests/               # unit + integration tests
```

## Audit trail

Each entry commits to the previous entry's digest:

```
entry_hash = SHA256(seq | inspection_id | timestamp | agent | action | payload_hash | prev_hash)
```

Editing or deleting any historical record breaks every entry after it. Verify at any time:

```bash
mavia audit verify     # -> "Chain intact - 128 entries verified"
mavia audit show       # tabular view of recent decisions
```

Tamper detection is covered by unit tests: payload edits and deleted entries are both
caught, and the chain resumes correctly across process restarts.

## Results

Populated as each phase lands. See [EVALUATION.md](EVALUATION.md).

| Metric | Target | Measured |
|---|---|---|
| Image-level AUROC (MVTec AD, 15 categories) | > 0.98 | **0.9868** |
| Pixel-level AUROC | > 0.97 | **0.9767** |
| PRO | — | **0.8046** |
| Detection latency | < 100 ms | **35.5 ms/image** (MPS) |
| Retrieval precision@3 | ceiling 0.449 | **0.4548** (hybrid) |
| Retrieval hit-rate@3 | — | **0.6904** |
| Retrieval latency (warm) | — | **9 ms** |
| Citation grounding rate | 1.00 | **1.000** |
| Risk agreement (within one level) | — | **0.680** |
| Retrieval ablation: risk changed without history | — | **76%** |
| End-to-end latency (auto-approved) | < 10 s | **~1.5 s** warm |
| Approval survives process restart | required | **✅ verified across 3 processes** |
| Audit chain integrity | 100% | **✅ 17/17 entries** |

Reference: Roth et al. 2022 report 0.991 image / 0.981 pixel AUROC for
PatchCore-1%. This from-scratch implementation lands within ~0.4 points of both,
without the paper's test-time augmentation or multi-scale ensembling. Five
categories reach a perfect 1.0000. Full per-category table, the weakest cases and
why, plus the coreset and PRO ablations: [EVALUATION.md](EVALUATION.md).

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| 1 | Repo, config, typed schemas, audit hash chain, CLI, CI | ✅ done |
| 2 | Vision agent: PatchCore + CLIP fallback, detection eval | ✅ done |
| 3 | Defect-history corpus, Qdrant store, retrieval agent + retrieval eval | ✅ done |
| 4 | Root Cause Analyst on Claude, structured output, prompt evaluation | ✅ done |
| 5 | LangGraph orchestration, HITL interrupt, durable checkpoints | ✅ done |
| 6 | Report Writer (Markdown → PDF), full audit integration | ✅ done |
| 7 | Streamlit dashboard, Docker, demo video | next |
| 8 | End-to-end evaluation, written report, deployment | |

## License

Code: MIT. The MVTec AD dataset is CC BY-NC-SA 4.0 and is **not** redistributed here —
`make data` fetches it from MVTec.
