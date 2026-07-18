# Held-Out Results

The frozen 30-seed study supports the immediate proof-of-concept claim: an online,
reward-free loss-gap estimator separates the agent's own action-contingent sensory
stream from a matched yoked stream, and its direct vigor gate becomes passive and
recovers when fixed-amplitude probes preserve identification.

Seven of eight prespecified hypotheses were supported. All confidence intervals
below are 10,000-replicate percentile bootstrap intervals over seeds; transitions
were not treated as independent samples.

## Primary Findings

- Connected influence exceeded matched-yoked influence by 1.590 nats
  (95% CI 1.580 to 1.601). The paired sensory-variance difference was -0.085
  (95% CI -0.217 to 0.045), so the yoked effect was not a quiet-environment cue.
- The connected-to-disconnected drop in influence was 1.597 nats
  (95% CI 1.586 to 1.607), and the reconnection rise was 1.593 nats
  (95% CI 1.583 to 1.604).
- At the default analytic target of 1.629 nats, the unclipped prequential loss gap
  was low by 0.010 nats (95% CI -0.021 to 0.001). Across the 25-cell gain/noise
  grid, the fitted calibration slope was 0.979 and RMSE was 0.073 nats. Those global
  summaries conceal under-calibration in the smallest feature-scale cells, where
  finite-horizon state-coefficient learning is slow.
- Gating saved 0.246 action-energy units per disconnected transition versus the
  ungated fixed controller (95% CI 0.240 to 0.251). Mean vigor fell from 0.990 in
  the initial connected phase to 0.027 while disconnected, then recovered to
  0.986 after reconnection.
- All behavioral arms were exactly matched through the initial connected phase.
  When probes were withdrawn at disconnection, all 30 seeds remained passive
  through the 1,500-step reconnection phase. Persistent probes at probability 0.03
  recovered all 30 seeds with a median online-confirmed latency of 63.5 steps;
  probability 0.05 also recovered all seeds, with a 56.5-step median.

## Controls

- The real motor command produced a 1.619-nat action-aware advantage. A
  within-phase shuffled command produced -0.00003 nats
  (95% CI -0.00059 to 0.00053); dummy and wrong-lag inputs were likewise near zero.
- A deterministic action policy produced -0.00006 nats of tail influence,
  demonstrating the expected identification failure when action has no variation
  beyond current state.
- Reversing the action mapping caused a mean initial forward-model squared-error
  spike of 3.528, but tail influence remained 1.612 nats. Its paired difference
  from ordinary connected influence was -0.006 nats
  (95% CI -0.038 to 0.026).
- Raw forward error was useful as a surprise signal but was not specific: it also
  changed with irreducible noise and spiked under a still-controllable reversal.

## Prespecified Null

H7 was not supported. Coupled-versus-yoked ROC-AUC generally improved with
action-to-noise ratio (correlation 0.390), but the fixed-threshold detection latency
did not become monotonically shorter (correlation 0.555, opposite the prediction).
The result is retained as a null; no post-hoc retuning was used to reverse it. A
follow-up should distinguish
information-limited latency from threshold, feature scaling, and learner-adaptation
effects.

The exact machine-readable estimates are in
`artifacts/sensorimotor_influence/hypothesis_results.json` and `inference.csv`.
The run's configuration SHA-256 is
`52ea815a684df20772adac2a82bd7f591a61c30306afdcbe109f44c04a4bb5a6`; the full
execution-protocol SHA-256 is
`5bf9a1f71e0b9da3f8ff17533cd7dad4aec3d89e7040014cb612eb0c634dbd77`.
