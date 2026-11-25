"""UMAP -> HDBSCAN (Marchiori Manerba et al., 2024), with seeds and a sensitivity sweep."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from .metrics import agreement

UMAP_KW = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine")


def reduce_and_cluster(emb: np.ndarray, min_cluster: int = 10, seed: int = 42) -> np.ndarray:
    from hdbscan import HDBSCAN
    from umap import UMAP

    red = UMAP(random_state=seed, **UMAP_KW).fit_transform(emb)
    return HDBSCAN(min_cluster_size=min_cluster, cluster_selection_method="eom").fit_predict(red)


def make_topic_model(encoder, min_cluster: int = 10, seed: int = 42, stop_words="english"):
    """BERTopic used only as a wrapper around UMAP+HDBSCAN, for c-TF-IDF cluster keywords."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    return BERTopic(
        embedding_model=encoder,
        umap_model=UMAP(random_state=seed, **UMAP_KW),
        hdbscan_model=HDBSCAN(min_cluster_size=min_cluster, metric="euclidean",
                              cluster_selection_method="eom", prediction_data=True),
        vectorizer_model=CountVectorizer(stop_words=stop_words, min_df=2, ngram_range=(1, 2)),
        top_n_words=10,
        calculate_probabilities=False,
    )


def sweep(labels, emb: np.ndarray, sizes=(5, 10, 15, 25, 40), seeds=(0, 1, 2), cluster_fn=reduce_and_cluster):
    """RQ1 agreement metrics across min_cluster_size x seeds, plus seed stability (mean pairwise ARI)."""
    labels = np.asarray(labels)
    rows, parts = [], {}
    for mcs, s in itertools.product(sizes, seeds):
        cl = cluster_fn(emb, min_cluster=mcs, seed=s)
        parts[(mcs, s)] = cl
        rows.append(dict(min_cluster=mcs, seed=s, n_clusters=int(cl.max() + 1),
                         noise=float(np.mean(cl == -1)), **agreement(labels, cl)))
    res = pd.DataFrame(rows)
    summ = res.drop(columns="seed").groupby("min_cluster").agg(["mean", "min", "max"]).round(3)
    summ[("seed_ARI", "mean")] = pd.Series({
        mcs: np.mean([adjusted_rand_score(parts[(mcs, a)], parts[(mcs, b)])
                      for a, b in itertools.combinations(seeds, 2)]) if len(seeds) > 1 else np.nan
        for mcs in sizes
    }).round(3)
    return summ
