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

Two analytical branches share one frozen embedding space.

1. **Framing extraction** (qualitative): each benchmark's stated definitions and categories. Also count items vs. distinct templates per category (`docs/benchmarks.md`, `docs/crosswalk.md`).
2. **Unit of analysis and normalization.** BBQ: ambiguous context + negative question, in two views:
   - *unmasked*, identity terms kept: **who** the category targets;
   - *masked*, targets → Person A/B: **what** is said about them.
   Masked texts are de-duplicated within each category, collapsing name substitutions to template level; unmasked texts are capped at 6 per template. HESEIA: one stereotype sentence per item, with its (possibly multiple) labels mapped to harmonized dimensions.
3. **Embedding:** `sentence-transformers/LaBSE`, frozen, L2-normalized. Robustness: `paraphrase-multilingual-mpnet-base-v2`. Never fine-tuned for the main analysis.

**Branch A: unsupervised (notebook 01, RQ1)**

4A. UMAP (5 dimensions, cosine, `n_neighbors=15`, `min_dist=0`, seeded) → HDBSCAN (`min_cluster_size=10` by default, EOM selection).
5A. Coherence metrics against each benchmark's own categories (§4), plus a linear probe on the benchmark's own categories with template-grouped CV.

**Branch B: supervised reference frame (notebook 02)**

4B. One logistic probe per identity dimension, trained on HESEIA (community labels as reference frame, not ground truth).
5B. Probe quality (AUC, selectivity vs. control task), geometry (direction similarity), and transfer to BBQ's own labels.

**Join**

6. Probes from B are projected onto the clusters from A, giving each cluster's driving dimension and the co-activation of dimensions on intersectional items.
7. **Robustness:** sensitivity grid (`min_cluster_size` × seeds, plus SOFA-style settings); second encoder; per-language centering; EsBBQ as language control; optional fine-tuned encoder as a comparison only.
8. **Interpretation:** content validity facets (contestedness, substantive, structural; Wallach et al., 2025) and representational-harm types (Shelby et al., 2023).

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

**Identity-dimension probes (notebook 02):** one logistic probe per dimension, trained on frozen LaBSE embeddings of HESEIA (community labels as reference frame). Reported per dimension:
- cross-validated AUC and selectivity against a control task;
- cosine similarity between probe directions (structural validity);
- transfer AUC on BBQ's own labels (RQ2);
- per-cluster driver dimension vs. BBQ's own category (RQ1/RQ2);
- co-activation on intersectional items.

Projections are interpreted only for dimensions that pass the thresholds fixed in the decision log.

## 4b. Statistical tests (`src/benchbias/stats.py`)

The template, not the item, is the unit of independent information in BBQ (58,492 items from ~325 templates). All nulls reassign whole templates; all CIs resample templates; tests across categories or clusters are BH-corrected. Permutation p-values are (1 + #null ≥ observed) / (1 + B), with B = 999.

| RQ | H0 | Statistic | Null / inference | Function |
|---|---|---|---|---|
| RQ1 | Given templates, category carries no information about cluster membership | AMI(category, cluster) | Template-level permutation (item-level shown for contrast) | `rq1_permutation_test` |
| RQ1 | A category is no more internally similar than to the rest, beyond templates | Coherence gap Δ (within − between mean cosine similarity, full 768-d space) | Template-level permutation, BH across categories; template bootstrap CI | `rq1_gap_test`, `bootstrap_by_group` |
| RQ2 | Category labels unrelated to content across benchmarks | Mean rank of the same-labelled category in the cross-benchmark distance matrix | Permute the other benchmark's labels (robust to global language/format offset) | `category_distance_matrix`, `label_match_test` |
| RQ2 | Same-labelled categories have the same distribution | Unbiased MMD² (Gaussian kernel, median heuristic); C2ST balanced accuracy as effect size | Permutation; grouped CV for C2ST | `mmd_test`, `c2st` |
| RQ3 | Coverage is symmetric between institutional and community benchmark | k-NN manifold coverage Cov(I→H), asymmetry A = Cov(H→I) − Cov(I→H) | Bootstrap CI; EsBBQ as language-matched I | `manifold_coverage` |
| RQ3 | No cluster over-represents one benchmark | Hypergeometric tail per cluster | BH; orphan = q < .05 and share ≥ .9 | `orphan_clusters` |

Full formulas: `docs/statistical_tests_full.md`. The walkthrough page shows a pseudocode draft.

## 5. Confounds to control before claiming erasure (RQ3)

| Confound | Why it matters | Control |
|---|---|---|
| Language | ES vs. EN items can separate even in LaBSE | EsBBQ as a control (if it sits with BBQ and not HESEIA, the split is about origin); per-language mean-centering of embeddings |
| Format | QA scenarios vs. stereotype statements vs. coreference sentences | report clusters that are single-benchmark *and* single-format; compare masked units; qualitative reading of orphans |
| Size / redundancy | BBQ's 58k items come from ~325 templates | template-level normalization; balanced samples per category in the merged corpus |
| Encoder | LaBSE is trained for translation retrieval, not similarity | repeat with an STS-trained multilingual encoder |
| Probe shortcut | probes may read identity words or dataset style, not the dimension | control task / selectivity; masked BBQ; train on one dataset only; per-language centering |
