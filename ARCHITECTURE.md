# MAVIA — System Design

This document records the design decisions behind MAVIA and, more importantly, the
reasoning and rejected alternatives behind each. It is the source material for
Sections 4 and 5 of the final report.

---

## 1. Design principles

1. **Typed contracts between agents.** Every agent consumes and produces a Pydantic
   model defined in `src/mavia/schemas.py`. No agent passes free-form dicts. This makes
   the graph statically checkable, the dashboard trivially renderable, and the audit
   payloads self-describing.
2. **Every decision is evidence-bearing.** No step emits a conclusion without a score, a
   latency, and a model identifier. "Defect found" is not an acceptable output; "defect
   found, score 0.83 against threshold 0.61, PatchCore/WideResNet50, 41 ms" is.
3. **Governance is not a feature bolted on at the end.** The audit chain was built in
   Phase 1, before any model, so that every later component is written against it rather
   than retrofitted into it.
4. **Degrade, do not crash.** A missing LLM key, an unreachable vector store, or an
   unseen product category must produce a lower-confidence result with a recorded
   error — never a dropped inspection. A production QA line cannot stop because a
   dependency is unavailable.

## 2. The modelling decision: anomaly detection over supervised detection

The original project brief specified YOLOv8. That is the wrong tool for this dataset and
this problem, and the substitution is the single most defensible technical decision in
the project.

**Why YOLOv8 does not fit MVTec AD.** The dataset provides ~3,600 defect-free training
images and ~1,700 test images. Defective samples appear *only* in the test split, and
ground truth is supplied as pixel-level segmentation masks, not bounding boxes. To train
a supervised detector you would have to (a) split the test set to obtain defect examples,
which contaminates evaluation, and (b) derive boxes from masks. Any reported mAP is then
measured on data the model has effectively been tuned against.

**Why it does not fit the real problem either.** In a factory, defect-free product is
abundant and each new failure mode is rare and initially unlabelled. A supervised
detector can only find the defect classes it was shown. The operationally interesting
question — "is this unit unlike every good unit we have ever seen?" — is by construction
an anomaly detection question.

**What MAVIA uses instead.**

| Role | Model | Rationale |
|---|---|---|
| Primary | **PatchCore** | Builds a coreset memory bank of patch embeddings from defect-free images only. ~99.1% image AUROC on MVTec AD. No defect labels required — matches the real cold-start constraint. |
| Efficiency comparison | **EfficientAD** | Student-teacher distillation, dramatically lower inference cost. Included so the evaluation can present an accuracy/latency Pareto curve rather than a single number. |
| Unseen categories | **CLIP / WinCLIP zero-shot** | When a product category has no memory bank, fall back to language-prompted classification ("a photo of a damaged {object}" vs "a photo of a flawless {object}"). Keeps the system useful on day one for a new SKU. |
| Localisation | Threshold + connected components over the anomaly map | Converts the pixel heatmap into the discrete regions the report cites as evidence. |

**Reported metrics.** Image-level AUROC, pixel-level AUROC, and PRO score — the metrics
MVTec AD is actually benchmarked with, allowing direct comparison against published
numbers. Not mAP.

## 3. Agent responsibilities

### Agent 1 — Vision Inspector
Input: image path (+ optional category). Output: `VisionResult`.

Loads the category's PatchCore memory bank, produces an image-level score and an anomaly
map, thresholds the map into regions, and writes a heatmap overlay for the dashboard. If
no memory bank exists for the category, it falls back to CLIP zero-shot and marks the
result accordingly so downstream agents can discount confidence.

The decision threshold is calibrated per category on a held-out validation split
(maximising F1), not fixed globally — different textures have very different score
distributions.

### Agent 2 — History Retriever
Input: `VisionResult`. Output: `RetrievalResult`.

Builds a natural-language query from the detection (category, defect morphology, affected
region, severity band), embeds it with `all-MiniLM-L6-v2`, and runs kNN against the
Qdrant `defect_history` collection with a metadata filter on category. Returns top-k
cases with their recorded root causes and remediations.

Qdrant runs embedded on-disk by default so the project has no Docker dependency during
development; `MAVIA_QDRANT_URL` switches it to a server or Qdrant Cloud with no code
change.

### Agent 3 — Root Cause Analyst
Input: `VisionResult` + `RetrievalResult`. Output: `RootCauseAnalysis`.

Prompts Claude with the detection evidence and the retrieved cases, constrained to
structured output. Two guardrails matter here:

- The model must cite `case_id`s from the retrieved set when it attributes a root cause,
  which makes grounding checkable rather than assumed.
- `risk_level` drives control flow, so it is an enum with defined semantics, not free
  text.

If no API key is configured, this agent emits a deterministic rule-based analysis at
reduced confidence rather than failing the inspection.

### Human approval gate
Not an agent — a routing decision plus a LangGraph `interrupt`.

`risk_level >= HIGH` (or anomaly score above `MAVIA_HIGH_RISK_THRESHOLD`) suspends the
graph at a checkpoint. The state is durable via the SQLite checkpointer, so the pause can
outlive the process: a reviewer can approve from the dashboard minutes later and the
graph resumes from exactly where it stopped. This is the difference between a demo and a
system — an autonomous agent that can stop a production line must be interruptible, and
the interrupt must survive a restart.

Approvals and rejections are themselves audited, with approver identity and rationale.

### Agent 4 — Report Writer
Input: the full `InspectionState`. Output: `QAReport`.

Renders a Jinja template to Markdown, then to PDF. The report is deliberately templated
rather than free-form LLM prose: an audited QA document needs a stable structure, and the
LLM's contribution belongs in the analysis fields it already produced.

## 4. The audit chain

```
entry_hash = SHA256(seq | inspection_id | timestamp | agent | action | payload_hash | prev_hash)
payload_hash = SHA256(canonical_json(payload))
```

Append-only JSONL. Each entry commits to its predecessor's digest, so any edit or
deletion invalidates the entire suffix — detected by `verify_chain`, exposed as
`mavia audit verify`.

Design notes:

- **Canonical JSON** (sorted keys, no whitespace) so an identical payload always yields
  an identical digest across processes and Python versions.
- **The chain is global, not per-inspection.** A per-inspection chain would let an
  attacker delete an entire inspection undetected; a global chain makes any removal
  visible as a sequence break.
- **The logger resumes from the existing head on restart**, so the chain survives process
  boundaries — verified by test.
- **Honest limitation:** this is tamper-*evident*, not tamper-*proof*. An attacker with
  write access to the log file can recompute the whole chain. Production hardening would
  anchor periodic head digests somewhere append-only and externally controlled (a
  timestamping authority, a WORM bucket, or a signed transparency log). Stated as such in
  the report's Limitations section rather than overclaimed.

## 5. Orchestration

LangGraph, chosen over a hand-rolled loop or a linear chain for three specific
capabilities the project needs:

- **`interrupt`** — first-class human-in-the-loop suspension, which is the governance
  requirement.
- **Checkpointer** — durable state, so a pending approval survives a restart.
- **Conditional edges** — risk-based routing between the auto-approve and
  human-review paths.

State is a single `InspectionState` model threaded through every node. Each node is
wrapped so that entry and exit are audited automatically; no node is responsible for
remembering to log itself.

## 6. Evaluation strategy

| Layer | Metrics | Method |
|---|---|---|
| Detection | Image AUROC, pixel AUROC, PRO, latency | All 15 MVTec AD categories, compared against published PatchCore/EfficientAD numbers |
| Retrieval | precision@k, recall@k, MRR | Labelled query set over the defect-history corpus |
| Analysis | Citation grounding rate, risk-level agreement | Does the analysis cite real retrieved cases; does its risk level agree with a human-labelled subset |
| End-to-end | Wall-clock latency per stage, throughput | Per-node timing already carried in the state |
| Governance | Chain integrity, audit completeness | `verify_chain` over the full run; every inspection must have an unbroken event sequence |

The evaluation deliberately includes an **ablation**: pipeline without retrieval vs with
retrieval, to demonstrate that the memory layer measurably changes analysis quality
rather than being decorative.

## 7. Known limitations

- MVTec AD is a benchmark, not a production line: fixed lighting, fixed pose, no
  conveyor motion blur. Real deployment needs domain adaptation.
- The defect-history corpus is synthesised from MVTec's defect taxonomy plus plausible
  manufacturing root causes. It is realistic but not real factory data, so retrieval
  quality numbers describe the mechanism, not field performance.
- Root-cause attribution is a plausibility judgement by an LLM over retrieved precedent.
  It is decision *support*, and the human gate exists precisely because it is not
  decision *authority*.
