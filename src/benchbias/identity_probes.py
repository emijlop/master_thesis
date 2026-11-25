"""Identity-dimension probes on frozen sentence embeddings.

One binary logistic-regression probe per identity dimension (one-vs-rest, multi-label), trained on a
community-built benchmark (HESEIA) as the *reference frame*, then applied to other benchmarks' items and
aggregated per cluster: "which community-defined dimension drives this cluster?"

Safeguards built in:
  - grouped cross-validation (no near-duplicates / templates across folds)
  - control task (Hewitt & Liang, 2019): same probe on permuted labels -> selectivity = AUC - control AUC
  - transfer check: a probe's AUC on the target benchmark's own (harmonized) labels, before its
    projections are interpreted there (language and format shift)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, KFold


def _clf(C: float = 1.0):
    return LogisticRegression(max_iter=3000, class_weight="balanced", C=C)


def multi_hot(label_lists, dimensions=None) -> pd.DataFrame:
    """List-of-labels per item -> binary item x dimension matrix (supports intersectional multi-labels)."""
    dims = dimensions or sorted({d for labels in label_lists for d in labels})
    return pd.DataFrame([[int(d in set(labels)) for d in dims] for labels in label_lists], columns=dims)


def _folds(n, groups, n_splits, seed):
    if groups is not None:
        return list(GroupKFold(n_splits).split(np.zeros(n), groups=groups))
    return list(KFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(n)))


def evaluate_probes(X: np.ndarray, Y: pd.DataFrame, groups=None, n_splits: int = 5, C: float = 1.0,
                    seed: int = 42, min_pos: int = 20) -> pd.DataFrame:
    """Cross-validated AUC/F1 per dimension, plus control-task AUC and selectivity."""
    rng = np.random.default_rng(seed)
    folds = _folds(len(X), None if groups is None else np.asarray(groups), n_splits, seed)
    rows = []
    for dim in Y.columns:
        y = Y[dim].to_numpy()
        if y.sum() < min_pos:
            rows.append(dict(dimension=dim, n_pos=int(y.sum()), note=f"< {min_pos} positives, skipped"))
            continue
        y_ctrl = rng.permutation(y)  # control task: same base rate, no relation to content
        p, p_ctrl = np.zeros(len(y)), np.zeros(len(y))
        for tr, te in folds:
            if len(set(y[tr])) < 2:
                continue
            p[te] = _clf(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
            p_ctrl[te] = _clf(C).fit(X[tr], y_ctrl[tr]).predict_proba(X[te])[:, 1]
        auc, auc_c = roc_auc_score(y, p), roc_auc_score(y_ctrl, p_ctrl)
        rows.append(dict(dimension=dim, n_pos=int(y.sum()), AUC=auc, F1=f1_score(y, p > 0.5),
                         control_AUC=auc_c, selectivity=auc - auc_c))
    return pd.DataFrame(rows).set_index("dimension").round(3)


def fit_probes(X: np.ndarray, Y: pd.DataFrame, C: float = 1.0, min_pos: int = 20) -> dict:
    """Fit one probe per dimension on all data. Returns {dimension: fitted LogisticRegression}."""
    return {d: _clf(C).fit(X, Y[d].to_numpy()) for d in Y.columns if Y[d].sum() >= min_pos}


def directions(probes: dict) -> pd.DataFrame:
    """Unit-norm probe weight vectors (dimension x embedding dim)."""
    W = np.vstack([p.coef_[0] for p in probes.values()])
    return pd.DataFrame(W / np.linalg.norm(W, axis=1, keepdims=True), index=list(probes))


def direction_similarity(probes: dict) -> pd.DataFrame:
    """Cosine similarity between probe directions: do e.g. gender and sexual orientation share a direction?"""
    D = directions(probes)
    return pd.DataFrame(D.values @ D.values.T, index=D.index, columns=D.index).round(3)


def score(X: np.ndarray, probes: dict) -> pd.DataFrame:
    """Per-item probability for each dimension (item x dimension)."""
    return pd.DataFrame({d: p.predict_proba(X)[:, 1] for d, p in probes.items()})


def transfer_check(scores: pd.DataFrame, target_labels, min_pos: int = 10) -> pd.DataFrame:
    """AUC of each probe on a target benchmark whose items carry harmonized labels.

    `target_labels`: one harmonized label per item (single-label) or a list per item (multi-label);
    intersection labels like 'race_ethnicity×gender' count as positive for both parts.
    """
    def parts(lab):
        labs = lab if isinstance(lab, (list, tuple, set)) else [lab]
        return {p for l in labs if isinstance(l, str) for p in l.split("×")}

    sets = [parts(l) for l in target_labels]
    rows = []
    for dim in scores.columns:
        y = np.array([dim in s for s in sets], dtype=int)
        if y.sum() < min_pos or y.sum() == len(y):
            rows.append(dict(dimension=dim, n_pos=int(y.sum()), transfer_AUC=np.nan))
            continue
        rows.append(dict(dimension=dim, n_pos=int(y.sum()), transfer_AUC=roc_auc_score(y, scores[dim])))
    return pd.DataFrame(rows).set_index("dimension").round(3)


def cluster_profiles(scores: pd.DataFrame, clusters, native=None, usable=None) -> pd.DataFrame:
    """Mean probe score per cluster, the dimension that 'drives' each cluster, and its margin.

    usable : dimensions that passed the transfer check; others are excluded from the driver decision.
    native : the benchmark's own category per item; adds the cluster's dominant native category.
    """
    cols = [c for c in scores.columns if usable is None or c in set(usable)]
    df = scores[cols].assign(cluster=np.asarray(clusters))
    df = df[df["cluster"] != -1]
    prof = df.groupby("cluster")[cols].mean()
    ranked = np.sort(prof.values, axis=1)
    out = prof.round(3).assign(
        size=df.groupby("cluster").size(),
        driver=prof.idxmax(axis=1),
        driver_margin=(ranked[:, -1] - ranked[:, -2]).round(3) if len(cols) > 1 else np.nan,
    )
    if native is not None:
        nat = pd.Series(np.asarray(native), index=scores.index)[df.index]
        out["native_dominant"] = nat.groupby(df["cluster"]).agg(lambda s: s.value_counts().idxmax())
    return out


def co_activation(scores: pd.DataFrame, mask, threshold: float = 0.5) -> pd.Series:
    """Share of the selected items (e.g. intersectional ones) on which each pair of dimensions fires together."""
    s = scores[np.asarray(mask)] > threshold
    pairs = {f"{a} & {b}": float((s[a] & s[b]).mean())
             for i, a in enumerate(s.columns) for b in s.columns[i + 1:]}
    return pd.Series(pairs).sort_values(ascending=False).round(3)


def center_by_group(X: np.ndarray, groups) -> np.ndarray:
    """Subtract each group's mean embedding (e.g. per language) and re-normalize. Robustness check for
    cross-lingual transfer: removes the language offset that would otherwise act like a dimension."""
    X, groups = np.asarray(X, dtype=float).copy(), np.asarray(groups)
    for g in np.unique(groups):
        X[groups == g] -= X[groups == g].mean(axis=0)
    return X / np.linalg.norm(X, axis=1, keepdims=True).clip(min=1e-12)
