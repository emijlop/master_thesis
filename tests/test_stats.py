"""Statistical tests on synthetic data with known answers."""
import numpy as np
import pandas as pd
import pytest

from benchbias import stats


def _norm(X):
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def _benchmark(n_cat=4, n_tmpl=8, per_tmpl=6, cat_signal=1.0, tmpl_signal=1.0, d=16, seed=0):
    """Items = category signal + template signal + noise."""
    rng = np.random.default_rng(seed)
    cat_dirs, rows, cats, tmpls = rng.normal(size=(n_cat, d)), [], [], []
    for c in range(n_cat):
        for t in range(n_tmpl):
            t_dir = rng.normal(size=d)
            for _ in range(per_tmpl):
                rows.append(cat_signal * cat_dirs[c] + tmpl_signal * t_dir + rng.normal(scale=.3, size=d))
                cats.append(f"c{c}")
                tmpls.append(f"c{c}_t{t}")
    return _norm(np.array(rows)), np.array(cats), np.array(tmpls)


def test_bh_matches_definition():
    q = stats.bh([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)


def test_template_permutation_keeps_templates_whole():
    rng = np.random.default_rng(0)
    cats = np.array(["a"] * 4 + ["b"] * 4)
    tmpl = np.array(["t1", "t1", "t2", "t2", "t3", "t3", "t4", "t4"])
    for _ in range(20):
        p = stats.permute_within_groups_labels(cats, tmpl, rng)
        assert all(len(set(p[tmpl == t])) == 1 for t in np.unique(tmpl))
        assert sorted(p) == sorted(cats)


def test_rq1_template_null_rejects_only_with_category_signal():
    for signal, expect_sig in [(2.0, True), (0.0, False)]:
        E, cats, tmpls = _benchmark(cat_signal=signal)
        clusters = pd.factorize(tmpls)[0]            # clusters = templates (the trivial case)
        res = stats.rq1_permutation_test(cats, clusters, tmpls, B=199)
        # clusters == templates: item-level null always "significant", template-level null is not
        assert res["p_item"] < .05
        assert res["p_template"] > .05
        gap = stats.rq1_gap_test(E, cats, tmpls, B=199)
        assert (gap.loc["ALL", "p"] < .05) == expect_sig


def test_bootstrap_by_group_ci_contains_estimate():
    E, cats, tmpls = _benchmark(cat_signal=2.0)
    ci = stats.bootstrap_by_group(lambda i: stats.coherence_gap(E[i], cats[i])[0], tmpls, B=99)
    assert ci["ci_low"] <= ci["estimate"] <= ci["ci_high"] and ci["ci_low"] > 0


def test_mmd_detects_shift_not_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(150, 5))
    assert stats.mmd_test(X, rng.normal(size=(150, 5)), B=199)["p"] > .05
    assert stats.mmd_test(X, rng.normal(loc=1.0, size=(150, 5)), B=199)["p"] < .01


def test_label_match_robust_to_global_offset():
    EA, catA, _ = _benchmark(cat_signal=2.0, seed=1)
    rng = np.random.default_rng(2)
    offset = rng.normal(size=EA.shape[1]) * 1.5   # "language" shift applied to every item of B
    EB = _norm(EA + offset + rng.normal(scale=.2, size=EA.shape))
    D = stats.category_distance_matrix(EA, catA, EB, catA)
    per, res = stats.label_match_test(D, B=999)
    assert (per["rank_of_same_label"] == 1).all() and res["p"] < .05
    D_shuffled = D.copy(); D_shuffled.columns = ["c1", "c2", "c3", "c0"]
    assert stats.label_match_test(D_shuffled, B=999)[1]["p"] > .05


def test_c2st_chance_and_separable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    assert abs(stats.c2st(X[:100], X[100:])["balanced_accuracy"] - .5) < .12
    assert stats.c2st(X[:100], X[100:] + 3)["balanced_accuracy"] > .95


def test_manifold_coverage_asymmetric():
    rng = np.random.default_rng(0)
    broad = rng.normal(size=(400, 2)) * 3
    narrow = rng.normal(size=(400, 2)) * .5
    assert stats.manifold_coverage(broad, narrow)[0]["coverage"] > .9
    assert stats.manifold_coverage(narrow, broad)[0]["coverage"] < .3


def test_orphan_clusters_flags_single_benchmark_cluster():
    clusters = np.r_[np.zeros(50), np.ones(50), np.full(30, 2)].astype(int)
    bench = np.r_[["A"] * 25 + ["B"] * 25, ["A"] * 25 + ["B"] * 25, ["B"] * 30]
    out = stats.orphan_clusters(clusters, bench, target="B")
    assert out.loc[2, "orphan"] and not out.loc[0, "orphan"]


def test_runlog_writes_manifest(tmp_path):
    import json
    from benchbias.runlog import RunLog
    log = RunLog("unit", config={"seed": 1})
    with log.step("work"):
        sum(range(1000))
    log.add_result("x", 1.5)
    data = json.loads(log.save(tmp_path).read_text())
    assert data["config"]["seed"] == 1 and "work" in data["timings_s"] and data["results"]["x"] == 1.5
    assert {"platform", "cpu_count", "python"} <= set(data["hardware"])
