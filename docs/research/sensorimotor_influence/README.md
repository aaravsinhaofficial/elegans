# Online Sensorimotor-Influence Toy Study

This directory documents an isolated proof of concept. It does not use the
connectome, taxis behavior, reinforcement learning, a reward, or a learned policy.

The scientific question is:

> Can a small online learner estimate action-outcome contingency from egocentric
> sensorimotor transitions and use that estimate directly to regulate behavioral
> engagement?

The estimator compares an action-blind Gaussian predictor with an action-aware
Gaussian predictor. Their pre-update negative-log-likelihood difference is smoothed
into an operational estimate of **sensorimotor influence**. A fixed homeostatic
controller supplies direction; the learned estimate changes only its vigor.

## Reproduce

From the repository root:

```bash
# Fast development smoke run
uv run python -m elegans.sensorimotor_influence --quick \
  --output /tmp/sensorimotor-influence-quick

# Frozen 30-seed study
uv run ./scripts/run_sensorimotor_influence.py \
  --output artifacts/sensorimotor_influence
```

The full command runs 30 held-out seeds, three 1,500-transition phases, a 25-cell
gain/noise grid, and 10,000 seed-bootstrap replicates. It writes configuration and
seed manifests, seed-level CSV tables, representative lossless NumPy traces, and PNG
and PDF figures.

## Documents

- [Methods memo](methods_memo.md)
- [Held-out results](results.md)
- [Prespecified hypotheses](hypotheses.md)
- [Validation protocol](validation_protocol.md)
- [Literature notes](literature_notes.md)
- [Limitations](limitations.md)
- [Decisions for Eli](decisions_for_eli.md)
- [Figure guide](figure_guide.md)

## Claim boundary

The loss advantage estimates on-policy, one-step action-conditioned predictive
information when both predictors are calibrated. Randomized actions support a causal
interpretation in this toy system because its state is fully observed and the random
streams are independent. The same statement would not automatically hold with hidden
confounding, partial observability, or deterministic actions. The work therefore does
not yet establish classical controllability, agency, learned helplessness, or a
biological mechanism.
