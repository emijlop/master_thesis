"""RQ1 internal-coherence metrics.

Proposal stage 6 -> operationalization
  (i)  category purity per cluster -> homogeneity; dominant-category share per cluster
  (ii) category dispersion         -> completeness; clusters needed to cover 80% of a category; entropy
Reference points: permutation null (floor) and template_id (ceiling / template-recovery check).
Noise points (cluster == -1) are excluded from agreement scores and reported separately.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_mutual_info_score, homogeneity_completeness_v_measure


def agreement(labels, clusters) -> dict:
    labels, clusters = np.asarray(labels), np.asarray(clusters)
    m = clusters != -1
    h, c, v = homogeneity_completeness_v_measure(labels[m], clusters[m])
    return dict(homogeneity=h, completeness=c, v_measure=v,
                AMI=adjusted_mutual_info_score(labels[m], clusters[m]))


def rq1_summary(df: pd.DataFrame, n_perm: int = 50, seed: int = 42) -> pd.DataFrame:
    cl = df["cluster"].to_numpy()
    cats = df["category"].to_numpy()
    rng = np.random.default_rng(seed)
    rows = {
        "category": agreement(cats, cl),
        "template": agreement(df["template_id"].to_numpy(), cl),
        "category (permutation null, mean)": pd.DataFrame(
            [agreement(rng.permutation(cats), cl) for _ in range(n_perm)]
        ).mean().to_dict(),
    }
    return pd.DataFrame(rows).T.round(3)


def per_category(df: pd.DataFrame, cover: float = 0.8) -> pd.DataFrame:
    rows = []
    for cat, g in df.groupby("category"):
        dist = g.loc[g["cluster"] != -1, "cluster"].value_counts(normalize=True)
        k = int((dist.cumsum() < cover).sum() + 1) if len(dist) else np.nan
        H = float(-(dist * np.log(dist)).sum() / np.log(len(dist))) if len(dist) > 1 else 0.0
        rows.append(dict(category=cat, n=len(g), noise_share=(g["cluster"] == -1).mean(),
                         top_cluster_share=dist.iloc[0] if len(dist) else np.nan,
                         clusters_for_80pct=k, n_clusters_touched=len(dist), norm_entropy=H))
    return pd.DataFrame(rows).set_index("category").round(3).sort_values("clusters_for_80pct")


def per_cluster(df: pd.DataFrame) -> pd.DataFrame:
    ct = pd.crosstab(df["cluster"], df["category"]).drop(index=-1, errors="ignore")
    return pd.DataFrame({
        "size": ct.sum(axis=1),
        "dominant_category": ct.idxmax(axis=1),
        "purity": (ct.max(axis=1) / ct.sum(axis=1)).round(3),
        "n_categories": (ct > 0).sum(axis=1),
    }).sort_values("purity")


def intersection_placement(df: pd.DataFrame, intersectional=frozenset({"Race_x_SES", "Race_x_gender"}),
                           own_threshold: float = 0.5) -> pd.DataFrame:
    """Where intersectional items land: in clusters that are mostly intersectional ('own'),
    or inside clusters dominated by a single-axis category ('inside <axis>')."""
    ct = pd.crosstab(df["cluster"], df["category"]).drop(index=-1, errors="ignore")
    inter_cols = [c for c in ct.columns if c in intersectional]
    single_cols = [c for c in ct.columns if c not in intersectional]
    dom_single = ct[single_cols].idxmax(axis=1) if single_cols else pd.Series("none", index=ct.index)
    share_inter = ct[inter_cols].sum(axis=1) / ct.sum(axis=1)
    rows = []
    for cat in inter_cols:
        g = df[(df["category"] == cat) & (df["cluster"] != -1)]
        placed = g["cluster"].map(
            lambda c: f"own (≥{own_threshold:.0%} intersectional)" if share_inter[c] >= own_threshold
            else f"inside {dom_single[c]}"
        )
        rows.append(placed.value_counts(normalize=True).rename(cat))
    return pd.concat(rows, axis=1).fillna(0).round(3) if rows else pd.DataFrame()
