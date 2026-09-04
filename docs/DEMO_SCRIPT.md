# MAVIA — 2-Minute Demo Script

Recording notes: 1080p, terminal at a readable font size, browser at 100% zoom.
Rehearse once; the pauses matter more than the words.

**Before recording**

```bash
make dashboard                    # leave running at :8501
rm -f artifacts/checkpoints.sqlite artifacts/audit/audit_log.jsonl
uv run mavia inspect data/mvtec_ad/bottle/test/broken_large/000.png   # seed one PASS-through
```

Have two terminal windows and a browser tab ready.

---

## 0:00–0:15 · The problem

> "Automated visual inspection is a solved problem at the detection layer. What
> isn't solved is everything around it. A model that says 'defect found' can't
> tell you *why*, doesn't remember that this is the fourth time this week, and
> leaves no record you could put in front of an auditor.
>
> MAVIA is four agents that close those three gaps."

*Screen: the architecture diagram in README.*

---

## 0:15–0:40 · Detection

*Screen: dashboard → Inspect → pick `bottle / broken_large / 000.png` → Inspect.*

> "Agent one detects. This is PatchCore, implemented from scratch — 0.9868 mean
> image AUROC across all fifteen MVTec categories, within 0.4 points of the
> published paper.
>
> And note what it's trained on: only defect-free images. That's deliberate. The
> brief asked for YOLOv8, but MVTec has no defect training data — supervised
> detection would mean training on the test set. It also mismatches the real
> problem, where good parts are abundant and each new failure mode is unlabelled."

*Point at the heatmap overlay appearing beside the input.*

---

## 0:40–1:05 · Memory and reasoning

*Screen: scroll to the retrieved cases table and the root-cause block.*

> "Agent two retrieves comparable cases from 730 historical defect records.
> Agent three reasons over the detection plus that history to produce a probable
> cause, a recommended action, and a risk level.
>
> Two things I'd point at. Every cited case ID is verified against what was
> actually retrieved — invent one and the confidence gets capped. And the
> ablation: withhold the history and the risk decision changes on 76% of
> inspections. The memory layer is load-bearing, not decoration."

---

## 1:05–1:35 · The human gate — *the part that matters*

*Screen: the amber "Human approval required" banner.*

> "Risk came back CRITICAL, so the graph suspended. This isn't a modal dialog —
> the state is checkpointed to SQLite."

*Switch to terminal 1. Show the process is gone. Terminal 2:*

```bash
uv run mavia pending
```

> "A completely different process can see it waiting."

```bash
uv run mavia approve <id> --approver qa.lead@plant --rationale "Confirmed fracture"
```

> "And approve it. The graph resumes at exactly the node that stopped. An agent
> allowed to stop a production line has to be interruptible, and the interrupt
> has to outlive the request that created it."

---

## 1:35–2:00 · Governance

*Screen: dashboard → Audit trail.*

```bash
uv run mavia audit verify
```

> "Every decision is a link in a SHA-256 hash chain. Each entry commits to its
> predecessor, so altering any record invalidates everything after it."

*Open the generated PDF, scroll to section 6.*

> "And the chain is embedded in the QA report itself — an auditor re-derives it
> from the log rather than trusting the document.
>
> The whole thing runs in a 3.3 GB container. The methodological point I'd leave
> you with: I measured the information ceiling of each component before tuning
> it, and that caught two evaluation bugs whose scores looked like success. One
> of them scored *above* the theoretical limit. That's not a triumph, it's a
> defect report."

---

## Timing checkpoints

| Mark | Should be showing |
|---|---|
| 0:15 | Architecture diagram |
| 0:40 | Heatmap overlay rendered |
| 1:05 | Approval banner |
| 1:35 | Resumed from a second process |
| 2:00 | `Chain intact — N entries verified` |

## If asked to go longer (5-minute cut)

Add, in order of value: the retrieval ceiling analysis (§6.2–6.3 of the report —
the strongest single segment), the greedy k-center vs random coreset result, the
PRO vs pixel-AUROC discrimination table, and the container running the same
pipeline with an audit chain spanning host and container.
