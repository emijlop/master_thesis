"""BBQ (Parrish et al., 2022): loading and construction of analysis units."""
from __future__ import annotations

import re

import pandas as pd

SUBSETS = [
    "Age", "Disability_status", "Gender_identity", "Nationality", "Physical_appearance",
    "Race_ethnicity", "Race_x_SES", "Race_x_gender", "Religion", "SES", "Sexual_orientation",
]
INTERSECTIONAL = {"Race_x_SES", "Race_x_gender"}
HF_PATH = "hf://datasets/heegyu/bbq@refs%2Fconvert%2Fparquet/{subset}/test/*.parquet"


def load_bbq_raw(subsets=SUBSETS) -> pd.DataFrame:
    """All BBQ items, with a `template_id` = category + question_index."""
    from datasets import load_dataset

    frames = [
        load_dataset("parquet", data_files=HF_PATH.format(subset=s), split="train").to_pandas()
        for s in subsets
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw["template_id"] = raw["category"] + "_" + raw["question_index"].astype(str)
    return raw


def framing_table(raw: pd.DataFrame) -> pd.DataFrame:
    """Items vs. distinct templates per category (substantive-validity observation)."""
    return (
        raw.groupby("category")
        .agg(items=("example_id", "size"), templates=("template_id", "nunique"))
        .assign(
            items_per_template=lambda d: (d["items"] / d["templates"]).round(1),
            share_of_items=lambda d: (d["items"] / d["items"].sum()).round(3),
            share_of_templates=lambda d: (d["templates"] / d["templates"].sum()).round(3),
        )
        .sort_values("items", ascending=False)
    )


def entities(answer_info: dict) -> list[str]:
    """Surface strings of the two non-'unknown' answer options, e.g. 'grandfather', 'Jamal Branch'."""
    return [v[0] for _, v in sorted(answer_info.items()) if v[1] != "unknown"]


def group_pair(answer_info: dict) -> str:
    """Group tags of the two targets, e.g. 'nonOld vs old'."""
    return " vs ".join(sorted(v[1] for v in answer_info.values() if v[1] != "unknown"))


def mask(context: str, ents: list[str], labels=("Person A", "Person B")) -> tuple[str, int]:
    """Replace target entities (and the first name of multi-word names) with neutral labels.

    Returns the masked text and the number of entity variants found (>= 2 means both targets masked).
    """
    out, hits = context, 0
    for label, e in zip(labels, ents):
        parts = e.split()
        variants = {e} | ({parts[0]} if len(parts) > 1 and len(parts[0]) > 2 else set())
        for v in sorted(variants, key=len, reverse=True):
            out, n = re.compile(rf"\b{re.escape(v)}\b", flags=re.I).subn(label, out)
            hits += n > 0
    return out, hits


def build_units(raw: pd.DataFrame, k_per_template: int = 6, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Unit = ambiguous context + negative question, in two views.

    masked   : target entities -> Person A/B, de-duplicated on masked text within each category
               (collapses name substitutions; identical texts in different categories are kept, see shared_texts)
    unmasked : identity terms kept, at most `k_per_template` unique instances per template
    """
    amb = raw[(raw["context_condition"] == "ambig") & (raw["question_polarity"] == "neg")].copy()
    amb["ents"] = amb["answer_info"].apply(entities)
    amb["groups"] = amb["answer_info"].apply(group_pair)
    amb["stereotyped_groups"] = amb["additional_metadata"].apply(
        lambda m: ", ".join(m.get("stereotyped_groups", []))
    )
    masked_ctx = amb.apply(lambda r: mask(r["context"], r["ents"]), axis=1)
    amb["masked_context"] = [m[0] for m in masked_ctx]
    amb["mask_hits"] = [m[1] for m in masked_ctx]
    amb["text_unmasked"] = amb["context"] + " " + amb["question"]
    amb["text_masked"] = amb["masked_context"] + " " + amb["question"]
    amb["intersectional"] = amb["category"].isin(INTERSECTIONAL)

    masked = (
        amb.drop_duplicates(["category", "text_masked"]).reset_index(drop=True)
        .assign(view="masked", text=lambda d: d["text_masked"])
    )
    unmasked = (
        amb.drop_duplicates(["category", "text_unmasked"])
        .groupby("template_id", group_keys=False)
        .apply(lambda g: g.sample(min(len(g), k_per_template), random_state=seed))
        .reset_index(drop=True)
        .assign(view="unmasked", text=lambda d: d["text_unmasked"])
    )
    return {"masked": masked, "unmasked": unmasked}


def shared_texts(units: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Texts that occur verbatim under more than one category, e.g. a Race_x_gender item whose masked
    text equals a Race_ethnicity item. Evidence that the category is defined only by the substituted label."""
    multi = units.groupby(text_col)["category"].agg(lambda c: tuple(sorted(set(c))))
    multi = multi[multi.map(len) > 1]
    return (multi.value_counts().rename_axis("categories").reset_index(name="n_texts"))
