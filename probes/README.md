# Concept probes

Small scripts (about ten minutes each) that make one concept visible, such as
BM25's k1 and b, MinHash accuracy, or chunk-length dilution. One per phase,
named `phase-N-<concept>.py`, run with `uv run python probes/<file>.py`.

Probes read from stage outputs and config but are not part of the pipeline.
Any figure they produce goes to `docs/figures/`.
