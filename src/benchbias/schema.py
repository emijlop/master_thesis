"""Common cross-benchmark schema (proposal stage 2) and category crosswalks.

Every benchmark is exported into SCHEMA so the cross-benchmark corpus is a concatenation.
The crosswalk is an operationalization choice: document every mapping in docs/crosswalk.md.
"""
from __future__ import annotations

import pandas as pd

SCHEMA = [
    "benchmark", "origin", "language", "format", "view", "category_native",
    "category_harmonized", "intersectional", "template_id", "text", "stereotyped_groups",
]

CROSSWALK = {
    "BBQ": {
        "Gender_identity": "gender",
        "Sexual_orientation": "sexual_orientation",
        "Age": "age",
        "Disability_status": "disability",
        "Nationality": "nationality",
        "Physical_appearance": "physical_appearance",
        "Race_ethnicity": "race_ethnicity",
        "Religion": "religion",
        "SES": "ses",
        "Race_x_gender": "race_ethnicity×gender",
        "Race_x_SES": "race_ethnicity×ses",
    },
    # "HESEIA": {...},   # fill from docs/crosswalk.md once HESEIA labels are inspected
    # "WinoBias": {"gender": "gender"},
}

META = {
    "BBQ": dict(origin="industry-adopted", language="en", format="qa_context"),
    "HESEIA": dict(origin="community-built", language="es", format="stereotype_sentence"),
    "WinoBias": dict(origin="academic", language="en", format="coreference_sentence"),
    "EsBBQ": dict(origin="academic-adaptation", language="es", format="qa_context"),
}


def to_schema(df: pd.DataFrame, benchmark: str, category_col: str = "category") -> pd.DataFrame:
    out = df.copy()
    for k, v in META[benchmark].items():
        out[k] = v
    out["benchmark"] = benchmark
    out["category_native"] = out[category_col]
    out["category_harmonized"] = out[category_col].map(CROSSWALK.get(benchmark, {}))
    for col in SCHEMA:
        if col not in out:
            out[col] = pd.NA
    return out[SCHEMA]
