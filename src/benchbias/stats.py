"""Statistical tests for the three sub-questions.

The unit of independent information in BBQ is the template, not the item: name substitutions of one
template are near-copies. Every null distribution and confidence interval below therefore resamples or
permutes at template (group) level unless stated otherwise.

RQ1 internal coherence
  rq1_permutation_test   cluster-category agreement (AMI) vs. a template-restricted permutation null
  coherence_gap / rq1_gap_test   clustering-free check: within- vs between-category cosine similarity
  bootstrap_by_group     percentile CIs from resampling templates
RQ2 convergence
  mmd2 / mmd_test        two-sample test for one category across two benchmarks (Gretton et al., 2012)
  category_distance_matrix + label_match_test   is the same-labelled category the closest one?
  c2st                   classifier two-sample test as an interpretable effect size (Lopez-Paz & Oquab, 2017)
RQ3 origin
  manifold_coverage      share of one benchmark inside the k-NN manifold of another (Kynkäänniemi et al., 2019)
  orphan_clusters        clusters over-represented by one benchmark (hypergeometric test + BH)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_mutual_info_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestNeighbors


# ---------------------------------------------------------------- helpers
def bh(pvals) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def perm_pvalue(observed: float, null, greater: bool = True) -> float:
    """(1 + #null at least as extreme) / (1 + B): never exactly zero."""
    null = np.asarray(null)
    extreme = (null >= observed) if greater else (null <= observed)
    return float((1 + extreme.sum()) / (1 + len(null)))


def permute_within_groups_labels(labels, groups, rng) -> np.ndarray:
    """Shuffle labels between whole groups (templates), keeping every group's items together.

    Each template carries one category; the permutation reassigns categories to templates, so the
    number of templates per category is preserved and template structure is not broken."""
    labels, groups = np.asarray(labels), np.asarray(groups)
    g_ids, inv = np.unique(groups, return_inverse=True)
    g_label = pd.Series(labels).groupby(inv).agg(lambda s: s.iloc[0]).to_numpy()
    return rng.permutation(g_label)[inv]


# ---------------------------------------------------------------- RQ1
def rq1_permutation_test(categories, clusters, templates, B: int = 999, seed: int = 42) -> dict:
    """H0: given the template structure, category labels carry no information about cluster membership.

    Statistic: AMI(clusters, categories) on non-noise items. Two nulls are reported:
    item-level (labels shuffled freely; weak) and template-level (whole templates reassigned; the relevant one).
    """
    cats, cl, tm = map(np.asarray, (categories, clusters, templates))
    m = cl != -1
    cats, cl, tm = cats[m], cl[m], tm[m]
    rng = np.random.default_rng(seed)
    obs = adjusted_mutual_info_score(cats, cl)
    null_item = [adjusted_mutual_info_score(rng.permutation(cats), cl) for _ in range(B)]
    null_tmpl = [adjusted_mutual_info_score(permute_within_groups_labels(cats, tm, rng), cl) for _ in range(B)]
    return dict(AMI=obs,
                null_item_mean=float(np.mean(null_item)), p_item=perm_pvalue(obs, null_item),
                null_template_mean=float(np.mean(null_tmpl)), null_template_95=float(np.quantile(null_tmpl, .95)),
                p_template=perm_pvalue(obs, null_tmpl), B=B)


def coherence_gap(E: np.ndarray, labels) -> tuple[float, pd.Series]:
    """Mean within-category cosine similarity minus mean between-category similarity.

    Computed from category sums (rows of E are L2-normalized), so it is O(N d) and cheap to permute.
    Returns the overall gap and the per-category gap (within_c minus similarity of c to all other items).
    """
    labels = np.asarray(labels)
    cats = np.unique(labels)
    S_all = E.sum(axis=0)
    N = len(E)
    within_sum, within_pairs, per = 0.0, 0, {}
    for c in cats:
        Ec = E[labels == c]
        n = len(Ec)
        Sc = Ec.sum(axis=0)
        w = (Sc @ Sc - n) / (n * (n - 1)) if n > 1 else np.nan
        b = (Sc @ (S_all - Sc)) / (n * (N - n)) if N > n else np.nan
        per[c] = w - b
        within_sum += Sc @ Sc - n
        within_pairs += n * (n - 1)
    all_sum, all_pairs = S_all @ S_all - N, N * (N - 1)
    between = (all_sum - within_sum) / (all_pairs - within_pairs)
    return float(within_sum / within_pairs - between), pd.Series(per)


def rq1_gap_test(E: np.ndarray, categories, templates, B: int = 999, seed: int = 42) -> pd.DataFrame:
    """Clustering-free RQ1 test: is each category more similar within itself than to others,
    beyond what templates alone produce? Template-level permutation null, BH-corrected across categories."""
    rng = np.random.default_rng(seed)
    obs_all, obs = coherence_gap(E, categories)
    null_all, null = [], []
    for _ in range(B):
        perm = permute_within_groups_labels(categories, templates, rng)
        a, per = coherence_gap(E, perm)
        null_all.append(a)
        null.append(per.reindex(obs.index).to_numpy())
    null = np.vstack(null)
    p = np.array([perm_pvalue(obs.iloc[j], null[:, j]) for j in range(len(obs))])
    out = pd.DataFrame({"gap": obs, "null_mean": null.mean(axis=0), "p": p, "q_BH": bh(p)})
    out.loc["ALL"] = [obs_all, np.mean(null_all), perm_pvalue(obs_all, null_all), np.nan]
    return out.round(4)


def bootstrap_by_group(stat_fn, groups, B: int = 999, seed: int = 42, alpha: float = .05) -> dict:
    """Percentile CI for stat_fn(index_array), resampling whole groups (templates) with replacement."""
    groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    g_ids = np.unique(groups)
    idx_by_g = {g: np.flatnonzero(groups == g) for g in g_ids}
    est = stat_fn(np.arange(len(groups)))
    boots = []
    for _ in range(B):
        pick = rng.choice(g_ids, size=len(g_ids), replace=True)
        boots.append(stat_fn(np.concatenate([idx_by_g[g] for g in pick])))
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return dict(estimate=float(est), ci_low=float(lo), ci_high=float(hi), B=B)


# ---------------------------------------------------------------- RQ2
def _subsample(X, n_max, rng):
    return X if len(X) <= n_max else X[rng.choice(len(X), n_max, replace=False)]


def mmd2(X: np.ndarray, Y: np.ndarray, gamma: float | None = None) -> float:
    """Unbiased squared MMD with a Gaussian kernel k(x,y)=exp(-gamma ||x-y||^2); gamma by median heuristic."""
    Z = np.vstack([X, Y])
    sq = np.sum(Z ** 2, axis=1)
    D = np.maximum(sq[:, None] + sq[None, :] - 2 * Z @ Z.T, 0)
    if gamma is None:
        med = np.median(D[np.triu_indices_from(D, 1)])
        gamma = 1.0 / med if med > 0 else 1.0
    K = np.exp(-gamma * D)
    m, n = len(X), len(Y)
    Kxx, Kyy, Kxy = K[:m, :m], K[m:, m:], K[:m, m:]
    return float((Kxx.sum() - np.trace(Kxx)) / (m * (m - 1)) + (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
                 - 2 * Kxy.mean())


def mmd_test(X: np.ndarray, Y: np.ndarray, B: int = 499, n_max: int = 1500, seed: int = 42) -> dict:
    """H0: P_X = P_Y. Permutation test on pooled samples. With large n and any language or format
    difference this is almost always rejected, so read it together with label_match_test."""
    rng = np.random.default_rng(seed)
    X, Y = _subsample(X, n_max, rng), _subsample(Y, n_max, rng)
    obs = mmd2(X, Y)
    Z, m = np.vstack([X, Y]), len(X)
    null = []
    for _ in range(B):
        p = rng.permutation(len(Z))
        null.append(mmd2(Z[p[:m]], Z[p[m:]]))
    return dict(MMD2=obs, p=perm_pvalue(obs, null), null_mean=float(np.mean(null)), n_x=len(X), n_y=len(Y))


def category_distance_matrix(EA, catA, EB, catB, method: str = "centroid", n_max: int = 800,
                             seed: int = 42) -> pd.DataFrame:
    """Distance between every category of benchmark A (rows) and of benchmark B (columns).
    'centroid': cosine distance of category means (fast). 'mmd': squared MMD (slower, distributional)."""
    rng = np.random.default_rng(seed)
    catA, catB = np.asarray(catA), np.asarray(catB)
    ra, rb = sorted(set(catA)), sorted(set(catB))
    D = pd.DataFrame(index=ra, columns=rb, dtype=float)
    for a in ra:
        Xa = EA[catA == a]
        for b in rb:
            Xb = EB[catB == b]
            if method == "centroid":
                ca, cb = Xa.mean(0), Xb.mean(0)
                D.loc[a, b] = 1 - ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb))
            else:
                D.loc[a, b] = mmd2(_subsample(Xa, n_max, rng), _subsample(Xb, n_max, rng))
    return D


def label_match_test(D: pd.DataFrame, B: int = 9999, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    """RQ2 core test, robust to a global language/format offset between the benchmarks.

    For each category c present in both benchmarks: is B's same-labelled category the closest one to A's c?
    Statistic: mean rank (1 = closest) of the matching column within each row, over shared categories.
    H0: labels are unrelated to content across benchmarks -> column labels exchangeable.
    """
    shared = [c for c in D.index if c in D.columns]
    ranks = D.rank(axis=1, method="average")
    per = pd.DataFrame({"same_label_distance": [D.loc[c, c] for c in shared],
                        "closest_in_other": [D.loc[c].idxmin() for c in shared],
                        "rank_of_same_label": [ranks.loc[c, c] for c in shared],
                        "n_candidates": D.shape[1]}, index=shared)
    obs = per["rank_of_same_label"].mean()
    rng = np.random.default_rng(seed)
    cols = np.array(D.columns)
    null = []
    for _ in range(B):
        Dp = D.copy()
        Dp.columns = rng.permutation(cols)
        rp = Dp.rank(axis=1, method="average")
        null.append(np.mean([rp.loc[c, c] if c in rp.columns else np.nan for c in shared]))
    return per, dict(mean_rank=float(obs), null_mean=float(np.nanmean(null)),
                     p=perm_pvalue(obs, np.array(null), greater=False), n_shared=len(shared))


def c2st(XA: np.ndarray, XB: np.ndarray, groups=None, n_splits: int = 5, seed: int = 42) -> dict:
    """Classifier two-sample test: cross-validated accuracy of telling A from B.
    0.5 = indistinguishable. Grouped CV (templates) when groups are given."""
    X = np.vstack([XA, XB])
    y = np.r_[np.zeros(len(XA)), np.ones(len(XB))]
    clf = LogisticRegression(max_iter=3000, class_weight="balanced")
    if groups is not None:
        acc = cross_val_score(clf, X, y, cv=GroupKFold(n_splits), groups=np.asarray(groups), scoring="balanced_accuracy")
    else:
        acc = cross_val_score(clf, X, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed),
                              scoring="balanced_accuracy")
    return dict(balanced_accuracy=float(acc.mean()), sd=float(acc.std()), chance=0.5)


# ---------------------------------------------------------------- RQ3
def manifold_coverage(ref: np.ndarray, query: np.ndarray, k: int = 5) -> dict:
    """Share of query items that fall inside the union of k-NN balls around reference items
    (improved precision/recall, Kynkäänniemi et al., 2019). Asymmetric:
    coverage(ref=BBQ, query=HESEIA) = how much of HESEIA's content BBQ reaches."""
    nn_ref = NearestNeighbors(n_neighbors=k + 1).fit(ref)
    radii = nn_ref.kneighbors(ref)[0][:, k]          # distance to k-th neighbour (index 0 is itself)
    d, idx = NearestNeighbors(n_neighbors=min(50, len(ref))).fit(ref).kneighbors(query)
    covered = (d <= radii[idx]).any(axis=1)
    return dict(coverage=float(covered.mean()), k=k, n_ref=len(ref), n_query=len(query)), covered


def orphan_clusters(clusters, benchmark, target: str, min_share: float = .9, alpha: float = .05) -> pd.DataFrame:
    """Per cluster: share of `target` items and a one-sided hypergeometric test of over-representation
    given corpus proportions; BH across clusters. Orphan = share >= min_share and q < alpha."""
    cl, bm = np.asarray(clusters), np.asarray(benchmark)
    keep = cl != -1
    cl, bm = cl[keep], bm[keep]
    N, K = len(cl), int((bm == target).sum())
    rows = []
    for c in np.unique(cl):
        m = cl == c
        n, x = int(m.sum()), int((bm[m] == target).sum())
        rows.append(dict(cluster=c, size=n, n_target=x, share=x / n, p=float(hypergeom.sf(x - 1, N, K, n))))
    out = pd.DataFrame(rows).set_index("cluster")
    out["q_BH"] = bh(out["p"])
    out["orphan"] = (out["share"] >= min_share) & (out["q_BH"] < alpha)
    return out.sort_values("share", ascending=False).round(4)
