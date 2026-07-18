# Decisions for Eli

The toy milestone can proceed without a connectome or learned policy. The remaining
choices concern what claim to optimize next.

1. **Primary term.** Keep “sensorimotor influence” or “action-outcome contingency” in
   titles. Reserve “causal action influence” for randomized, sufficiently observed
   conditions and avoid “agency” or classical “controllability.”
2. **Operational versus calibrated signal.** Retain adaptive predictor-specific
   variances for the information-theoretic result. If circuit locality is the next
   goal, add a fixed-variance squared-error implementation as a separately calibrated
   mechanism rather than replacing the probabilistic benchmark.
3. **Response-time hypothesis.** Decide whether a fixed-rate leaky accumulator is
   expected to slow at weak coupling. The current architecture can lose
   discrimination without a monotonic switch-latency effect; forcing both properties
   may require uncertainty-adaptive evidence accumulation.
4. **Probe budget.** Choose whether 3% fixed-amplitude probes are an explanatory model
   or merely an identification device. A next experiment should trace the complete
   energy-recovery frontier and test structured rather than iid probes.
5. **Behavioral target.** The present target `x*=1` exists only to create a fixed
   directional controller. Do not interpret its homeostatic error biologically.
6. **Yoked behavior.** For a truly matched behavioral yoke, decide between replaying a
   paired master agent's action stream and sampling from a frozen reference action
   distribution. Independent Gaussian yoking is exact only in Experiment 1A.
7. **Biological comparison.** If the next step is zebrafish futility, compare a fast
   mismatch/slow inhibition model with the influence estimator. If it is Drosophila,
   prioritize master/yoked behavioral fits. Do not select *C. elegans* solely because
   a connectome is available.
8. **Circuit work gate.** Begin connectome-constrained implementation only after the
   estimator survives delayed, partially observed, and nonlinear toy systems and a
   local squared-error version reproduces the same qualitative behavior.

Recommended immediate next experiment: add a hidden-state/delayed-actuation toy model
and compare one-step state conditioning with short sensory-motor histories. This
directly tests whether the current signal survives the first realistic complication.
