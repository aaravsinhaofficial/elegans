# Sensorimotor-Influence Study Artifacts

These are the frozen outputs of the 30-seed confirmatory toy study. Regenerate them
from the repository root with:

```bash
uv run ./scripts/run_sensorimotor_influence.py \
  --output artifacts/sensorimotor_influence
```

The run used three 1,500-transition phases, a separate 30-seed robustness block,
and 10,000 seed-bootstrap replicates. Its configuration SHA-256 is
`52ea815a684df20772adac2a82bd7f591a61c30306afdcbe109f44c04a4bb5a6`, and its
configuration-plus-execution-protocol SHA-256 is
`5bf9a1f71e0b9da3f8ff17533cd7dad4aec3d89e7040014cb612eb0c634dbd77`.

## Result Summary

Seven of eight prespecified hypotheses were supported. Connected-minus-yoked
influence was 1.590 nats (95% CI 1.580 to 1.601) while sensory variance remained
matched. Gating saved 0.246 action-energy units per disconnected transition. All
30 probe-withdrawal seeds failed to recover after reconnection, whereas all 30 seeds
with persistent probe probabilities 0.03 and 0.05 recovered. H7, the prespecified monotonic
detection-latency prediction across gain/noise conditions, was not supported.

## Contents

- `configuration.json`: every frozen experiment configuration.
- `run_manifest.json`: seed blocks, inferential unit, and configuration hash.
- `hypothesis_results.json`: machine-readable H1-H8 outcomes.
- `inference.csv`: estimates and seed-bootstrap confidence intervals.
- `phase_metrics.csv`: seed-by-condition phase summaries.
- `latencies.csv`: fixed-threshold detection and censored recovery metrics.
- `robustness.csv`: gain-by-noise analytic and empirical summaries.
- `control_summary.csv`: compact baseline and control comparisons.
- `representative_traces/`: lossless NumPy transition traces used for figures.
- `figures/`: five visually checked figures in PNG and PDF formats.

See `docs/research/sensorimotor_influence/results.md` for the scientific reading and
claim boundary.
