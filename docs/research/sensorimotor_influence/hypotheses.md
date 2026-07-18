# Prespecified Hypotheses

These hypotheses and decision rules were frozen before the held-out 30-seed run.
Numerical estimates belong in the generated result artifact, not in this file.

1. **H1 -- matched contingency.** Phase-mean influence is higher when the focal action
   drives the world than during a yoked phase, while the paired sensory-variance
   difference remains compatible with zero.
2. **H2 -- reversibility.** Influence falls after disconnection and rises after
   reconnection. Both transitions use frozen thresholds and report censoring.
3. **H3 -- energy.** The learned vigor gate uses less action energy than the ungated
   controller during the disconnected phase.
4. **H4 -- probing.** After exactly matched probing throughout the initial connected
   phase, ongoing fixed-amplitude probes reduce capped recovery latency or the
   fraction of runs that fail to recover relative to withdrawing probes at
   disconnection.
5. **H5 -- surprise is not influence.** Raw action-aware forward error may identify the
   default yoked phase, but it is not specific: sufficiently noisy connected dynamics
   can have more error than a lower-noise yoked world, and a still-controllable
   reversal produces a transient error spike.
6. **H6 -- command identity.** A within-phase-shuffled command has less than 20% of the
   real command's connected-phase predictive advantage.
7. **H7 -- graded difficulty.** Coupled-versus-yoked discrimination improves and switch
   detection becomes faster as action-to-noise ratio rises. Both parts must hold; a
   partial result is reported as unsupported.
8. **H8 -- reversal.** After adaptation, reversed coupling has positive influence and
   its paired difference from positive coupling lies within 0.163 nat, despite a
   larger immediate forward-model error.

Additional identifiability control: with a deterministic action given the current
state, tail influence should lie within 0.1 nat of zero even though the action has a
physical effect. This tests the conditional-variation requirement rather than the
physics of the actuator.

The primary uncertainty statement is the paired 95% percentile-bootstrap interval
across seeds. Null equivalence is evaluated against a practical margin; failure to
reject zero is not treated as proof of equivalence.
