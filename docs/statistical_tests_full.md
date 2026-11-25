# Formal model and statistical tests (full version, for later)

Parked on 2026-11-25. The walkthrough page now shows a pseudocode version, marked as a draft. Consider including this full version in the thesis once the first runs confirm the design. The code for every test is in `src/benchbias/stats.py`.

## Notation

Benchmark $b$ has items $i = 1,\dots,N_b$ with text $s_i$, native category $c_i \in \mathcal{C}_b$ and template $\tau_i$. Clusters are $\kappa_i$; $\kappa_i = -1$ is noise.

## Formal model

**Embedding.** $e_i = f_\theta(s_i)/\lVert f_\theta(s_i)\rVert_2 \in \mathbb{S}^{767}$, with $\theta$ frozen (LaBSE). The dot product $e_i^\top e_j$ is cosine similarity.

**UMAP** (McInnes et al., 2018).
- High-dimensional weights: $w_{ij} = \exp\!\big(-(d(e_i,e_j)-\rho_i)/\sigma_i\big)$, symmetrized as $\bar w_{ij} = w_{ij}+w_{ji}-w_{ij}w_{ji}$.
- Low-dimensional similarities: $q_{ij} = (1 + a\lVert y_i-y_j\rVert^{2b})^{-1}$.
- Objective, minimized over $y$: $\sum_{i\ne j} \bar w_{ij}\log(\bar w_{ij}/q_{ij}) + (1-\bar w_{ij})\log\big((1-\bar w_{ij})/(1-q_{ij})\big)$.

**HDBSCAN** (McInnes & Healy, 2017).
- Mutual reachability distance: $d_{\mathrm{mr}}(i,j) = \max\{\mathrm{core}_k(i), \mathrm{core}_k(j), \lVert y_i - y_j\rVert\}$.
- Cluster stability: $S(C) = \sum_{i\in C}(\lambda_i - \lambda_{\mathrm{birth}}(C))$, with $\lambda = 1/d_{\mathrm{mr}}$.
- The selected clusters are the non-overlapping set that maximizes $\sum S(C)$. Points outside them are noise.

**Coherence** (Rosenberg & Hirschberg, 2007).
- Homogeneity $h = 1 - H(C\mid K)/H(C)$ and completeness $c = 1 - H(K\mid C)/H(K)$; $V = 2hc/(h+c)$.
- Clustering-free gap: $\Delta = \overline{e_i^\top e_j}|_{c_i=c_j} - \overline{e_i^\top e_j}|_{c_i\ne c_j}$.

**Probes.** $\hat p_{id} = \sigma(w_d^\top e_i + \beta_d)$, fitted by class-weighted log loss plus $\frac{1}{2C}\lVert w_d\rVert^2$. Selectivity $= \mathrm{AUC}_d - \mathrm{AUC}^{\mathrm{ctrl}}_d$, where the control is trained on shuffled labels (Hewitt & Liang, 2019).

**Projection.** $\bar p_{kd} = \frac{1}{|k|}\sum_{i\in k}\hat p_{id}$; $\mathrm{driver}(k) = \arg\max_d \bar p_{kd}$.

## Inference

The template is the unit of independent information (58,492 BBQ items from ~325 templates). Therefore:
- nulls reassign whole templates;
- confidence intervals resample templates (cluster bootstrap);
- permutation p-values are $p = (1 + \#\{T^{(b)} \ge T\})/(1+B)$, with $B = 999$;
- tests across categories or clusters are corrected with Benjamini–Hochberg.

### RQ1: internal coherence
- **H0:** $P(\kappa \mid c,\tau) = P(\kappa \mid \tau)$.
- **Statistics:** $T_1 = \mathrm{AMI}(c,\kappa)$ on non-noise items; $T_2 = \Delta$, overall and per category.
- **Null:** template-level reassignment of categories. An item-level shuffle is reported for contrast; it overstates significance when clusters follow templates (shown in `tests/test_stats.py`).
- **Effect size:** $\Delta$ with a 95% template-bootstrap CI; AMI against the 95th percentile of the null.

### RQ2: convergence
- **H0:** benchmark B's category labels are exchangeable with respect to A's.
- **Statistic:** distance matrix $D_{ab}$ (centroid cosine distance; MMD² as a check); $T = \frac{1}{|\mathcal{C}_{A\cap B}|}\sum_c \mathrm{rank}_c(D_{cc})$.
- **Null:** permute B's labels. This is robust to a global language or format offset.
- **Per category:** $\mathrm{MMD}^2_u = \frac{1}{m(m-1)}\sum_{i\ne j}k(x_i,x_j) + \frac{1}{n(n-1)}\sum_{i\ne j}k(y_i,y_j) - \frac{2}{mn}\sum_{i,j}k(x_i,y_j)$, with a permutation test (Gretton et al., 2012). The C2ST balanced accuracy serves as effect size (Lopez-Paz & Oquab, 2017).
- **Control:** the same tests for BBQ ↔ EsBBQ (same origin, different language).

### RQ3: origin
- **H0:** coverage is symmetric.
- **Statistic:** $\mathrm{Cov}(I\to H) = \frac{1}{|H|}\sum_{h\in H}\mathbb{1}[\exists i\in I: \lVert e_h - e_i\rVert \le r_k(i)]$ (Kynkäänniemi et al., 2019). Asymmetry $A = \mathrm{Cov}(H\to I) - \mathrm{Cov}(I\to H)$, with a bootstrap CI; EsBBQ serves as the language-matched $I$.
- **Orphan clusters:** $X_k \sim \mathrm{Hypergeom}(N, N_H, n_k)$, $p_k = P(X_k \ge x_k)$, BH-corrected. Orphan = $q_k < 0.05$ and share $\ge 0.9$.
