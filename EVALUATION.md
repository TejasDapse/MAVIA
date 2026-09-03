# MAVIA — Evaluation

Results are filled in as each phase completes. Every number here is reproducible from a
notebook in [evaluation/](evaluation/); no figure is quoted without the script that
produced it.

## Method summary

| Layer | Metrics | Where |
|---|---|---|
| Detection | Image AUROC, pixel AUROC, PRO, latency (p50/p95) | `evaluation/detection_eval.ipynb` |
| Retrieval | precision@3, recall@3, MRR | `evaluation/retrieval_eval.ipynb` |
| Analysis | Citation grounding rate, risk-level agreement vs human labels | `evaluation/analysis_eval.ipynb` |
| End-to-end | Per-stage latency, throughput, ablation with/without retrieval | `evaluation/system_eval.ipynb` |
| Governance | Chain integrity, audit completeness | `mavia audit verify` + `tests/test_audit.py` |

## 1. Detection — MVTec AD (15 categories)

_Pending Phase 2._ Reported against published PatchCore and EfficientAD baselines.

| Category | Image AUROC | Pixel AUROC | PRO | Latency (ms) |
|---|---|---|---|---|
| bottle … zipper | — | — | — | — |
| **Mean** | — | — | — | — |

## 2. Retrieval

_Pending Phase 3._

## 3. Root-cause analysis

_Pending Phase 4._

## 4. End-to-end

_Pending Phase 5+._

## 5. Governance

| Property | Result |
|---|---|
| Hash chain intact over full evaluation run | pending |
| Payload tampering detected | ✅ covered by `test_verify_chain_detects_payload_tampering` |
| Entry deletion detected | ✅ covered by `test_verify_chain_detects_deleted_entry` |
| Chain resumes across process restart | ✅ covered by `test_logger_resumes_chain_across_restarts` |
