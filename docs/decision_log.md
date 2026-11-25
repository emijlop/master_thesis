# Decision log

One entry per methodological decision: what was decided, why, and what it affects. Newest first.

---

### 2026-11-25 · Inference at template level; run manifests
**Decision:**
- Every RQ gets an explicit H0 with a permutation or bootstrap procedure (`stats.py`).
- Nulls permute whole templates.
- RQ2 is tested by cross-benchmark label matching, with MMD and C2ST as secondary measures.
- RQ3 is tested by k-NN manifold coverage and hypergeometric orphan clusters.
- Every notebook run writes a `run_*.json` manifest (`runlog.py`) and saves all tables as CSV, tagged with `RUN`.

**Why:**
- Item-level nulls overstate significance when clusters follow templates. A unit test shows this: clusters equal to templates give p_item < .05 but p_template > .05.
- Label matching is robust to a global language or format offset, which a plain two-sample test would always detect.
- Manifests make results comparable across setups and document hardware and versions.

**Open:**
- Power for categories with few templates.
- The choice of k for manifold coverage.

**Presentation:** the walkthrough page shows these tests as pseudocode, marked as a draft. The full formal version is parked in `docs/statistical_tests_full.md`; consider including it in the thesis once the first runs confirm the design.

### 2026-11-25 · Identity-dimension probes on frozen LaBSE (notebook 02)
**Decision:** one linear (logistic) probe per identity dimension, trained on frozen LaBSE embeddings of **HESEIA**. The probes are applied to other benchmarks' items and averaged per cluster to find which dimension "drives" each cluster.
**Why:**
- It complements clustering (where items fall) with a labelled view (which dimension the content expresses).
- HESEIA's community labels serve as the *reference frame*, not ground truth, consistent with the contestedness argument.
- Linear probes keep the result interpretable: one direction per dimension.

**Safeguards:**
- grouped CV;
- control task with selectivity (Hewitt & Liang, 2019);
- transfer check against BBQ's own harmonized labels before interpreting projections;
- masked vs. unmasked BBQ;
- per-language centering;
- thresholds fixed in advance (AUC ≥ 0.75, selectivity ≥ 0.2, transfer AUC ≥ 0.70).

**Not chosen:**
- *Fine-tuning LaBSE for the main analysis.* Clustering a space trained on identity labels would be circular. Fine-tuning (e.g. SetFit) stays an optional robustness comparison, and its embeddings are never used for clustering.
- *Mixing HESEIA and WinoQueer as training data.* Probes would learn the dataset source, not the dimension. WinoQueer is better kept as a held-out check for sexual-orientation and gender-identity probes.

**Open:** HESEIA column names and label map (fill after inspecting the CSV); grouping variable for CV.

### 2026-11-25 · De-duplicate within category, report cross-category duplicates
**Decision:** masked/unmasked units are de-duplicated on `(category, text)`, not on text alone. Texts identical across categories are listed with `bbq.shared_texts()`.
**Why:** de-duplicating on text alone silently deletes one category's items when two categories share an identical scenario after masking, e.g. Race×Gender vs. Race_ethnicity. That overlap is itself evidence that the category is carried only by the substituted label.
**Affects:** RQ1 counts; §2 of notebook 01.

### 2026-11-25 · Repository structure and shared package
**Decision:** shared code lives in `src/benchbias/` with tests; notebooks import it.
**Why:** HESEIA, WinoBias and EsBBQ need identical masking, metrics and schema. Copy-pasting functions between notebooks would let them drift apart.

### 2026-11-25 · Template-level control
**Decision:** agreement is computed against `template_id` as well as category. Masked units are de-duplicated (template level); unmasked units are capped at 6 per template. The probe uses template-grouped cross-validation.
**Why:** BBQ has 58,492 items from ~325 templates. The first run (notebook 00) produced 449 clusters, with keywords characteristic of single templates (e.g. "card, pulled, snap, amex"). Without this control, RQ1 coherence would be trivially high.

### 2026-11-25 · Unit of analysis: ambiguous context + negative question
**Decision:** replaces the disambiguated context used in notebook 00.
**Why:** the disambiguating sentence encodes the answer, and is counter-stereotypical half the time. The ambiguous context plus the negative question is what BBQ's authors present as the stereotype probe.

### 2026-11-25 · Masked and unmasked views
**Decision:** every analysis is run on both views.
**Why:** this separates *who* a category targets (group labels) from *what* is said (stereotype content). Coherence only when unmasked means the category is defined by labels, not by a shared construct.

### 2026-11-25 · Encoder: frozen LaBSE, mpnet as robustness check
**Decision:** use `sentence-transformers/LaBSE` (proposal §4.2.1), not BERTopic's default `all-MiniLM-L6-v2` (English-only, used in notebook 00). Use `paraphrase-multilingual-mpnet-base-v2` as a robustness check.
**Why:** HESEIA is Spanish, so a cross-lingual space is required. LaBSE is optimized for translation retrieval rather than similarity, which is why a second encoder is needed.

### 2026-11-25 · Fixed seeds and sensitivity sweep
**Decision:** `SEED = 42`. Results are reported over `min_cluster_size` ∈ {5, 10, 15, 25, 40} × 3 UMAP seeds, together with seed-stability ARI.
**Why:** UMAP is stochastic, and the proposal commits to reporting cluster-assignment sensitivity.

---

## Open decisions
- [ ] HESEIA unit of analysis: single stereotype sentence? How to treat multi-label (intersectional) items in purity metrics?
- [ ] HESEIA → harmonized category crosswalk (see `crosswalk.md`)
- [ ] Merged-corpus sampling: equal per benchmark × category, or proportional?
- [ ] Add EsBBQ as a language control (likely yes; also needed to read probe transfer in notebook 02)
- [ ] Optional: fine-tuned-encoder comparison for the probes (SetFit), only as a robustness check
- [ ] Orphan-cluster threshold (90% single-benchmark?)
