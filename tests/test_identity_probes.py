"""Identity probes on synthetic embeddings where each dimension lives on its own axis."""
import numpy as np
import pandas as pd
import pytest

from benchbias import heseia
from benchbias import identity_probes as ip

DIMS = ["age", "gender", "race_ethnicity"]


def _synthetic(n=600, seed=0):
    rng = np.random.default_rng(seed)
    labels = []
    for _ in range(n):
        k = rng.choice([1, 2], p=[0.7, 0.3])  # 30% intersectional items
        labels.append(list(rng.choice(DIMS, size=k, replace=False)))
    Y = ip.multi_hot(labels, DIMS)
    X = rng.normal(scale=0.5, size=(n, 8))
    X[:, :3] += Y.values * 2.0  # dimension d is encoded on axis d
    return X, Y, labels


def test_multi_hot_handles_intersections():
    Y = ip.multi_hot([["age"], ["gender", "race_ethnicity"]], DIMS)
    assert Y.values.tolist() == [[1, 0, 0], [0, 1, 1]]


def test_evaluate_probes_selective():
    X, Y, _ = _synthetic()
    res = ip.evaluate_probes(X, Y, groups=np.arange(len(X)) % 50)
    assert (res["AUC"] > 0.9).all()
    assert (res["control_AUC"].between(0.35, 0.65)).all()
    assert (res["selectivity"] > 0.3).all()


def test_directions_orthogonal_for_independent_dims():
    X, Y, _ = _synthetic()
    sim = ip.direction_similarity(ip.fit_probes(X, Y))
    off = sim.values[~np.eye(3, dtype=bool)]
    assert np.abs(off).max() < 0.5 and np.allclose(np.diag(sim.values), 1)


def test_transfer_check_splits_intersection_labels():
    X, Y, _ = _synthetic()
    probes = ip.fit_probes(X, Y)
    Xt, Yt, labels = _synthetic(seed=1)
    target = ["×".join(sorted(l)) for l in labels]  # e.g. 'gender×race_ethnicity'
    res = ip.transfer_check(ip.score(Xt, probes), target)
    assert (res["transfer_AUC"] > 0.9).all()


def test_cluster_profiles_find_driver_and_respect_usable():
    X, Y, labels = _synthetic()
    S = ip.score(X, ip.fit_probes(X, Y))
    clusters = np.array([DIMS.index(l[0]) for l in labels])
    prof = ip.cluster_profiles(S, clusters, native=[l[0] for l in labels])
    assert (prof["driver"] == prof["native_dominant"]).all()
    prof2 = ip.cluster_profiles(S, clusters, usable=["age", "gender"])
    assert set(prof2["driver"]) <= {"age", "gender"}


def test_co_activation_on_intersectional_items():
    X, Y, labels = _synthetic()
    S = ip.score(X, ip.fit_probes(X, Y))
    both = np.array([set(l) == {"gender", "race_ethnicity"} for l in labels])
    co = ip.co_activation(S, both)
    assert co.index[0] == "gender & race_ethnicity"


def test_center_by_group_removes_offset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)); X[:100] += 3
    Xc = ip.center_by_group(X, np.repeat(["en", "es"], 100))
    assert np.allclose(np.linalg.norm(Xc, axis=1), 1)
    assert abs(Xc[:100].mean() - Xc[100:].mean()) < 0.2


def test_heseia_label_parsing_and_mapping(tmp_path):
    p = tmp_path / "h.csv"
    pd.DataFrame({"oracion": ["Las mujeres no saben manejar.", "Los viejos no entienden la tecnología."],
                  "tipo": ["Género", "edad, género"]}).to_csv(p, index=False)
    df = heseia.load_heseia(p, columns={"text": "oracion", "labels": "tipo"},
                            label_map={"género": "gender", "edad": "age"})
    assert df["dimensions"].tolist() == [["gender"], ["age", "gender"]]
    assert df["intersectional"].tolist() == [False, True]
    with pytest.raises(ValueError):
        heseia.load_heseia(p, columns={"text": None, "labels": None})
