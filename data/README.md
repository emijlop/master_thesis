# Data

No benchmark data is committed. Each notebook downloads what it needs, and generated files go to `outputs/` (git-ignored).

| Benchmark | Source | Loaded by | Notes |
|---|---|---|---|
| BBQ (Parrish et al., 2022) | Hugging Face `heegyu/bbq` (parquet conversion of the official release) | `benchbias.bbq.load_bbq_raw()` | 58,492 items, 11 categories (9 single-axis + Race×Gender, Race×SES) |
| HESEIA (Ivetta et al., 2025) | _to add_ | — | Spanish, 46,499 sentences, community-built, multi-label (intersectional) |
| WinoBias (Zhao et al., 2018) | _to add_ | — | English, 3,160 sentences, binary gender × 40 occupations |
| EsBBQ (Ruiz-Fernández et al., 2025) | _to add_ | — | Spanish BBQ adaptation; **language control** for RQ3 |

If a source needs a manual download, put it in `data/raw/<benchmark>/` (git-ignored) and record the version, date and URL here.
