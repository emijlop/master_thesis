"""Fast tests on synthetic data (no downloads, no models)."""
import numpy as np
import pandas as pd
import pytest

from benchbias import bbq, metrics, schema
from benchbias.cluster import sweep
from benchbias.probe import probe


def test_mask_simple_and_names():
    out, hits = bbq.mask("I saw a grandson and their grandfather outside.", ["grandfather", "grandson"])
    assert out == "I saw a Person B and their Person A outside." and hits == 2
    out, _ = bbq.mask("Jamal Branch and Amy Yu were late. Jamal said sorry.", ["Jamal Branch", "Amy Yu"])
    assert "Jamal" not in out and "Amy" not in out


def test_entities_and_group_pair():
    ai = {"ans0": ["grandfather", "old"], "ans1": ["Unknown", "unknown"], "ans2": ["grandson", "nonOld"]}
    assert bbq.entities(ai) == ["grandfather", "grandson"]
    assert bbq.group_pair(ai) == "nonOld vs old"


def _fake_raw():
    rows = []
    for cat in ["Age", "Race_x_gender"]:
        for t in range(3):
            for name_a, name_b in [("Ann", "Bob"), ("Cy", "Di")]:
                for pol, q in [("neg", "Who was rude?"), ("nonneg", "Who was kind?")]:
                    for cond in ["ambig", "disambig"]:
                        rows.append(dict(
                            example_id=len(rows), category=cat, question_index=str(t),
                            context_condition=cond, question_polarity=pol, question=q,
                            context=f"{name_a} met {name_b} at place {t}.",
                            answer_info={"ans0": [name_a, "g1"], "ans1": ["Unknown", "unknown"], "ans2": [name_b, "g2"]},
                            additional_metadata={"stereotyped_groups": ["g1"]},
                        ))
    raw = pd.DataFrame(rows)
    raw["template_id"] = raw["category"] + "_" + raw["question_index"]
    return raw


def test_build_units_collapses_substitutions():
    D = bbq.build_units(_fake_raw(), k_per_template=5)
    # masked: name substitutions collapse -> one unit per template
    assert len(D["masked"]) == 6
    assert D["masked"]["text"].str.contains("Person A").all()
    # unmasked keeps both name variants per template
    assert len(D["unmasked"]) == 12
    assert set(D["masked"]["intersectional"]) == {False, True}
    # identical masked texts across categories are kept and reported
    shared = bbq.shared_texts(D["masked"])
    assert shared["n_texts"].sum() == 3


def _clustered(perfect=True, n=300, seed=0):
    rng = np.random.default_rng(seed)
    cats = rng.choice(["Age", "SES", "Race_ethnicity", "Race_x_gender"], n)
    tmpl = np.array([f"{c}_{i}" for c, i in zip(cats, rng.integers(0, 3, n))])
    cl = pd.factorize(cats)[0] if perfect else rng.integers(-1, 8, n)
    return pd.DataFrame(dict(category=cats, template_id=tmpl, cluster=cl))


def test_agreement_perfect_vs_random():
    assert metrics.agreement(*_clustered(True)[["category", "cluster"]].T.values)["v_measure"] == pytest.approx(1.0)
    rnd = metrics.rq1_summary(_clustered(False), n_perm=5)
    assert rnd.loc["category", "v_measure"] < 0.2


def test_per_category_and_cluster():
    pc = metrics.per_category(_clustered(True))
    assert (pc["clusters_for_80pct"] == 1).all() and (pc["top_cluster_share"] == 1).all()
    assert (metrics.per_cluster(_clustered(True))["purity"] == 1).all()


def test_intersection_placement_own_vs_inside():
    df = pd.DataFrame(dict(
        category=["Race_ethnicity"] * 8 + ["Race_x_gender"] * 2 + ["Race_x_gender"] * 5,
        cluster=[0] * 10 + [1] * 5,
    ))
    res = metrics.intersection_placement(df)
    assert res.loc["inside Race_ethnicity", "Race_x_gender"] == pytest.approx(2 / 7, abs=1e-3)
    assert res.filter(like="own", axis=0)["Race_x_gender"].iloc[0] == pytest.approx(5 / 7, abs=1e-3)


def test_sweep_with_stub_clusterer():
    labels = np.repeat(["a", "b"], 50)
    stub = lambda emb, min_cluster, seed: (labels == "b").astype(int)
    out = sweep(labels, np.zeros((100, 2)), sizes=(5, 10), seeds=(0, 1), cluster_fn=stub)
    assert (out[("v_measure", "mean")] == 1).all() and (out[("seed_ARI", "mean")] == 1).all()


def test_probe_grouped_cv_runs():
    rng = np.random.default_rng(0)
    y = np.repeat(["a", "b", "c"], 40)
    X = rng.normal(size=(120, 5)) + (pd.factorize(y)[0][:, None] * 2)
    groups = np.array([f"{c}_{i % 8}" for i, c in enumerate(y)])
    scores, cm, _ = probe(X, y, groups)
    assert scores["macro_f1_grouped_by_template"] > 0.8 and cm.shape == (3, 3)


def test_schema_export():
    D = bbq.build_units(_fake_raw())
    out = schema.to_schema(D["masked"], "BBQ")
    assert list(out.columns) == schema.SCHEMA
    assert out["category_harmonized"].notna().all()
