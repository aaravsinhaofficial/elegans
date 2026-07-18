# Limitations

- The estimator is an on-policy predictive quantity. It supports a causal reading in
  this fully observed randomized toy world, but hidden common causes or omitted
  history would break that inference.
- A deterministic command contains no information beyond the conditioned history.
  This can make influence unidentifiable even when the actuator has a large physical
  effect. Ongoing randomized probes are part of the estimand, not just a recovery
  trick.
- During learning and immediately after a switch, the loss gap mixes environmental
  influence with model mismatch. Only steady, calibrated phases approximate
  conditional mutual information.
- The gain/noise grid is finite-horizon and therefore measures learnability as well
  as the analytic information target. Low-amplitude state features slow the online
  state-coefficient update, and the smallest-scale cells are visibly
  under-calibrated despite a near-unit global calibration slope. This likely
  contributes to the unsupported graded-latency hypothesis.
- The EMA and evidence clipping are engineered dynamics. The unclipped loss gap is
  used for calibration; the clipped EMA is only the operational gate signal.
- Adaptive Gaussian variance is not a biological implementation. A local
  fixed-variance squared-error version is easier to map to a circuit but is not
  calibrated to mutual information.
- The homeostatic controller, the assumption that futile action is costly, the vigor
  nonlinearity, and probe schedule are engineered priors. Reward-free is not
  assumption-free.
- Exact sensory-variance matching is guaranteed for the iid estimator-only yoked
  experiment. Once behavior is gated, the focal and yoked action distributions need
  not remain matched; behavioral yoking is therefore secondary.
- This scalar linear-Gaussian system has no partial observability, delayed effects,
  nonlinear dynamics, action constraints beyond clipping, or competing sensory
  causes. History-dependent models must be tested before biological interpretation.
- Multiple hypotheses and robustness cells are descriptive validation, not a powered
  biological confirmatory study. Seeds are simulation replicates, not animals.
- The comprehensive validation protocol contains additional recommended sweeps and
  acceptance checks that this proof-of-concept did not execute; its run-coverage
  section is the authoritative checklist.
- Raw forward error is not useless; in the default matched-yoke condition it can
  remain persistently elevated because irreducible conditional variance rises. Its
  weakness is lack of specificity across noise and reversal, not guaranteed
  adaptation to zero.
- No result here establishes classical control-theoretic controllability, subjective
  agency, learned helplessness, or a zebrafish, fly, or worm mechanism.
