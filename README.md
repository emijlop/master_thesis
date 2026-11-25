# Same Category, Different Territory

**Institutional and Community Framings of Social Bias in LLM Benchmarks** (tentative title)

Master thesis, Hertie School Data Science Lab, October 2026 – April 2027
Elena Mijares Lopez · Supervisor: Dr. Drew Dimmery · Advisors: Dr. Angelie Kraft, Laura Pföhler, M.A.

---

## Question

Do institutional and community-built LLM bias benchmarks operationalize the same demographic and intersecting constructs at the level of their dataset content, or do they diverge in ways that reveal gaps in construct (content) validity?

| | Sub-question | Status |
|---|---|---|
| **RQ1** | *Internal coherence*: does each benchmark's own category map onto a topically coherent cluster? | BBQ: `notebooks/01` |
| **RQ2** | *Cross-benchmark convergence*: when two benchmarks claim the same category, do their items land in the same semantic region? | planned |
| **RQ3** | *Origin-based differentiation*: does the community-built benchmark occupy semantic territory absent from institutional ones? | planned |

Benchmarks: **BBQ** (industry-adopted), **HESEIA** (community-built, Latin America, Spanish), **WinoBias** (academic). **EsBBQ** serves as a language control.

## Pipeline

```
 1  framing extraction            docs/benchmarks.md, docs/crosswalk.md
 2  units + normalization         bbq.py (masked / unmasked), heseia.py
 3  frozen LaBSE embeddings       embed.py   ── shared by both branches ──
                                   │
        ┌──────────────────────────┴──────────────────────────┐
  4A UNSUPERVISED (nb 01)                          4B SUPERVISED (nb 02)
     UMAP → HDBSCAN          cluster.py              identity probes trained on HESEIA
     coherence metrics       metrics.py              (community labels = reference frame)
     BBQ category probe      probe.py                quality · geometry · transfer to BBQ
        └──────────────────────────┬──────────────────────────┘
 5  projection: probes → clusters  which dimension drives each cluster? co-activation
 6  robustness                     sensitivity grid · 2nd encoder · language centering
                                   (optional) fine-tuned LaBSE, comparison only
 7  interpretation                 content validity: contestedness / substantive / structural
                                   (Wallach et al. 2025; Shelby et al. 2023)
```

LaBSE stays **frozen** in the main analysis: clustering in a space fine-tuned on identity labels would be circular. Fine-tuning (then freezing and probing) is only a robustness comparison; see `docs/decision_log.md`.

Details and metric definitions: [`docs/research_design.md`](docs/research_design.md). Every methodological choice, with its reason: [`docs/decision_log.md`](docs/decision_log.md).

## Repository map

```
notebooks/
  00_exploration_bertopic_bbq.ipynb     first BERTopic run (superseded, kept for the record)
  01_bbq_rq1_internal_coherence.ipynb   RQ1 on BBQ: masked/unmasked views, metrics, sensitivity, probe
  02_identity_probes.ipynb              HESEIA-trained identity probes on frozen LaBSE -> which dimension drives each BBQ cluster
src/benchbias/                           shared code imported by all notebooks
  bbq.py       loading, units, masking         metrics.py  RQ1 metrics
  heseia.py    HESEIA loading, label mapping   identity_probes.py  per-dimension probes, transfer, cluster drivers
  embed.py     cached frozen embeddings        cluster.py  UMAP→HDBSCAN, sweep
  probe.py     template-grouped linear probe   plots.py    figures
  schema.py    common cross-benchmark schema + category crosswalks
tests/                                   fast tests on synthetic data (no downloads)
docs/
  research_design.md   RQs, pipeline, operationalization of metrics, confounds
  decision_log.md      dated methodological decisions
  benchmarks.md        characteristics table of the benchmarks
  crosswalk.md         mapping of native categories to shared constructs
  presentation/        material for presenting the project
data/README.md         where each dataset comes from (no data committed)
outputs/               generated tables, figures, embeddings (git-ignored)
```

## Running

**Colab:** open a notebook from GitHub using the badge at its top. The first cell clones this repo and installs `benchbias`.

**Locally:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[pipeline,dev]"
pytest                       # fast checks, no downloads
jupyter lab notebooks/
```

Seeds are fixed (`benchbias.SEED = 42`). Embeddings are cached in `outputs/`, so re-runs are fast and give identical results.

## Conventions

- New analyses go in a new numbered notebook (`02_heseia_...`). Reusable code goes in `src/benchbias/` with a test.
- Methodological choices go in `docs/decision_log.md` when they are made, not only in a notebook cell.
- Optionally clear notebook outputs before committing (`nbstripout --install` once) to keep diffs readable.
