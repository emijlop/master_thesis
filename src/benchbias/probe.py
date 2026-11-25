"""Linear probe on frozen embeddings (proposal §4.2.1, tentative multinomial classifier).

Cross-validation is grouped by template: with random splits, instances of one template land in
train and test and the probe memorizes templates. The random-vs-grouped gap is itself reported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict


def _clf():
    return LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0)


def probe(emb: np.ndarray, labels, groups, n_splits: int = 5, seed: int = 42):
    """Returns (scores dict, row-normalized confusion matrix under grouped CV, classifier fit on all data)."""
    y, g = np.asarray(labels), np.asarray(groups)
    p_rand = cross_val_predict(_clf(), emb, y, cv=StratifiedKFold(n_splits, shuffle=True, random_state=seed))
    p_grp = cross_val_predict(_clf(), emb, y, cv=GroupKFold(n_splits), groups=g)
    scores = {"macro_f1_random_split": f1_score(y, p_rand, average="macro"),
              "macro_f1_grouped_by_template": f1_score(y, p_grp, average="macro")}
    labs = sorted(set(y))
    cm = pd.DataFrame(confusion_matrix(y, p_grp, labels=labs, normalize="true"), index=labs, columns=labs).round(2)
    return scores, cm, _clf().fit(emb, y)
