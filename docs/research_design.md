# Research design

Living summary of proposal v3 (Nov 2026), as operationalized in code. When the proposal and this file disagree, update one of them and log the change in `decision_log.md`.

## 1. Argument in one paragraph

Bias benchmarks are artifacts of the institutions that build them (Raji et al., 2021; Branford & Kraft, 2026). Benchmarks that use the *same category names* (gender, race, religion, ...) may still operationalize *different constructs*. That is a content-validity problem in the sense of Jacobs & Wallach (2019) and Wallach et al. (2024). Industry-adopted benchmarks such as BBQ are U.S.-centric and template-based. Community-built ones such as HESEIA are participatory, situated and intersectional, yet they rarely enter industry evaluation loops. If their content occupies territory that institutional benchmarks do not cover, that is evidence of **erasure** in institutional framings.

## 2. Research questions → evidence

| RQ | Claim tested | Primary evidence | Notebook |
|---|---|---|---|
| RQ1 internal coherence | A benchmark's category corresponds to a coherent region of content | homogeneity / completeness vs. category labels, against a permutation null and template labels; masked vs. unmasked | 01 (BBQ), 02+ (others) |
| RQ2 convergence | Same-named categories across benchmarks share a region | per-cluster benchmark composition within a harmonized category; centroid distances; cross-benchmark probe transfer | planned |
| RQ3 origin | Community content occupies territory absent from institutional benchmarks | orphan clusters (single-benchmark), **after** language and format controls | planned |

## 3. Pipeline

1. **Framing extraction** (qualitative): each benchmark's stated definitions and categories. Also count items vs. distinct templates per category.
2. **Unit of analysis and normalization.** BBQ: ambiguous context + negative question, in two views:
   - *unmasked*, identity terms kept: **who** the category targets;
   - *masked*, targets → Person A/B: **what** is said about them.
   Masked texts are de-duplicated within each category, which collapses name substitutions to template level. Unmasked texts are capped at 6 per template.
3. **Embedding:** `sentence-transformers/LaBSE`, frozen, L2-normalized. Robustness check: `paraphrase-multilingual-mpnet-base-v2`.
4. **UMAP:** 5 dimensions, cosine, `n_neighbors=15`, `min_dist=0`, seeded.
5. **HDBSCAN:** `min_cluster_size=10` by default, EOM selection. Sensitivity over {5, 10, 15, 25, 40} × 3 seeds.
6. **Metrics** (below).
7. **Interpretation:** substantive vs. structural validity (Wallach et al., 2024); representational-harm types (Shelby et al., 2023).

## 4. Metrics

| Proposal term | Operationalization | Reading |
|---|---|---|
| (i) category purity per cluster | homogeneity; dominant-category share per cluster | high = clusters do not mix categories |
| (ii) category dispersion | completeness; # clusters covering 80% of a category; normalized entropy | low dispersion = category occupies one region |
| (iii) cross-benchmark composition | share of each benchmark per cluster, within a harmonized category | RQ2 |
| (iv) orphan clusters | clusters with ≥ 90% items from one benchmark | RQ3, only after controls |

Reference points:
- **Permutation null:** shuffled category labels give the chance floor.
- **Template labels:** if agreement with `template_id` greatly exceeds agreement with category, the clustering recovers the authors' writing units, not a construct.
- Noise points (HDBSCAN −1) are excluded from agreement scores and reported per category.

**Intersectionality:** intersectional items are placed either in "own" clusters (≥ 50% intersectional) or "inside" a single-axis cluster. A probe trained on single-axis categories only then classifies intersectional items. Additive operationalization predicts they are read as one parent axis.

**Linear probe:** multinomial logistic regression on frozen embeddings, with cross-validation **grouped by template**. The gap between random-split and grouped scores quantifies template leakage. The confusion matrix shows which categories the content fails to separate (structural validity).

## 5. Confounds to control before claiming erasure (RQ3)

| Confound | Why it matters | Control |
|---|---|---|
| Language | ES vs. EN items can separate even in LaBSE | EsBBQ as a control (if it sits with BBQ and not HESEIA, the split is about origin); per-language mean-centering of embeddings |
| Format | QA scenarios vs. stereotype statements vs. coreference sentences | report clusters that are single-benchmark *and* single-format; compare masked units; qualitative reading of orphans |
| Size / redundancy | BBQ's 58k items come from ~325 templates | template-level normalization; balanced samples per category in the merged corpus |
| Encoder | LaBSE is trained for translation retrieval, not similarity | repeat with an STS-trained multilingual encoder |
