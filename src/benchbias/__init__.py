"""benchbias: shared code for the thesis
"Same Category, Different Territory: Institutional and Community Framings of Social Bias in LLM Benchmarks".

Modules
-------
bbq      load BBQ, build analysis units (masked / unmasked), framing table
schema   common cross-benchmark schema and category crosswalks
embed    frozen sentence-encoder embeddings with on-disk caching
cluster  UMAP -> HDBSCAN (optionally wrapped in BERTopic), sensitivity sweep
metrics  RQ1 coherence metrics (purity / dispersion), intersectionality placement
probe    multinomial linear probe with template-grouped cross-validation
plots    category x cluster heatmap, small-multiples maps

Heavy dependencies (datasets, sentence-transformers, umap, hdbscan, bertopic) are imported
inside the functions that need them, so metrics and units can be tested with numpy/pandas/sklearn only.
"""

SEED = 42
ENCODER = "sentence-transformers/LaBSE"
ROBUST_ENCODER = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
