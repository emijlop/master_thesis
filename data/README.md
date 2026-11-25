# Data

No benchmark data is committed. Each notebook downloads what it needs, and generated files go to `outputs/` (git-ignored).

| Benchmark | Source | Loaded by | Notes |
|---|---|---|---|
| BBQ (Parrish et al., 2022) | Hugging Face `heegyu/bbq` (parquet conversion of the official release) | `benchbias.bbq.load_bbq_raw()` | 58,492 items, 11 categories (9 single-axis + Race×Gender, Race×SES) |
| HESEIA (Ivetta et al., 2025) | Mozilla Data Collective, "HESEIA Sentence Bias Dataset" (CSV, CC-BY-SA-4.0, steward Fundación Vía Libre). Download manually to `data/raw/heseia/` | `benchbias.heseia.load_heseia()` (set columns + label map first) | Spanish, 46,499 sentences, community-built, multi-label (intersectional) |
| WinoBias (Zhao et al., 2018) | _to add_ | — | English, 3,160 sentences, binary gender × 40 occupations |
| WinoQueer (Felkner et al., 2023) | github.com/katyfelkner/winoqueer (CrowS-Pairs format) | — | community-in-the-loop, LGBTQ+ only, templated; held-out check for probes |
| EsBBQ (Ruiz-Fernández et al., 2025) | _to add_ | — | Spanish BBQ adaptation; **language control** for RQ3 |

If a source needs a manual download, put it in `data/raw/<benchmark>/` (git-ignored) and record the version, date and URL here.
