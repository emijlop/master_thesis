"""HESEIA (Ivetta et al., 2025): loading and label harmonization.

Source: HESEIA Sentence Bias Dataset, Mozilla Data Collective (CSV, CC-BY-SA-4.0), steward Fundación
Vía Libre. Download manually into data/raw/heseia/ and record version and date in data/README.md.

The column names and label vocabulary are NOT hard-coded from the paper: inspect the CSV first
(`inspect_heseia`), then fill COLUMNS and LABEL_MAP. Labels that do not map to a shared construct
keep their own harmonized name (see docs/crosswalk.md): forcing them into BBQ categories would hide
exactly the territory RQ3 looks for.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

# Fill after inspecting the CSV. Values are column names in the HESEIA file.
COLUMNS = {
    "text": None,        # stereotype sentence
    "labels": None,      # bias type(s); possibly several per item (intersectional)
    "group": None,       # optional: creator / school / pair id, used to group CV folds
}

# Native HESEIA label (lower-cased, stripped) -> harmonized construct. Fill after inspection.
LABEL_MAP: dict[str, str] = {
    # "género": "gender",
    # "edad": "age",
    # "discapacidad": "disability",
    # "origen geográfico": "geographic_origin",   # own label: do not force into nationality/race
    # "nivel socioeconómico": "ses",
    # "profesión": "profession",
    # ...
}

SPLIT = re.compile(r"\s*[,;|/]\s*|\s+y\s+|\s*×\s*")


def inspect_heseia(path: str | Path, **read_kw) -> None:
    df = pd.read_csv(path, **read_kw)
    print(df.shape)
    print(df.dtypes)
    print(df.head(5).to_string())
    for c in df.columns:
        if df[c].dtype == object and df[c].nunique() < 200:
            print(f"\n--- {c} ({df[c].nunique()} values)")
            print(df[c].value_counts().head(30).to_string())


def parse_labels(cell) -> list[str]:
    if pd.isna(cell):
        return []
    return [p.strip().lower() for p in SPLIT.split(str(cell)) if p.strip()]


def load_heseia(path: str | Path, columns: dict | None = None, label_map: dict | None = None,
                **read_kw) -> pd.DataFrame:
    """Returns text, native_labels (list), dimensions (list of harmonized labels), group."""
    cols = columns or COLUMNS
    lmap = label_map if label_map is not None else LABEL_MAP
    if not cols.get("text") or not cols.get("labels"):
        raise ValueError("Set heseia.COLUMNS['text'] and ['labels'] after running inspect_heseia().")
    raw = pd.read_csv(path, **read_kw)
    out = pd.DataFrame({"text": raw[cols["text"]].astype(str).str.strip()})
    out["native_labels"] = raw[cols["labels"]].apply(parse_labels)
    out["dimensions"] = out["native_labels"].apply(lambda ls: sorted({lmap.get(l, l) for l in ls}))
    out["group"] = raw[cols["group"]].astype(str) if cols.get("group") else pd.NA
    out["intersectional"] = out["dimensions"].map(len) > 1
    unmapped = sorted({l for ls in out["native_labels"] for l in ls} - set(lmap))
    if unmapped:
        print(f"[heseia] {len(unmapped)} native labels kept as their own dimension (unmapped): {unmapped[:15]}")
    return out.drop_duplicates("text").reset_index(drop=True)
