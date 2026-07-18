# Validation Protocol: Online Sensorimotor-Influence Estimation

**Status:** comprehensive reference protocol and next-stage validation checklist\
**Scope:** isolated linear-Gaussian toy system; no connectome, taxis model, reward,
value function, policy learning, or biological-circuit claim

## Run coverage

The current 30-seed proof-of-concept implements the core prequential estimator,
analytic calibration, cable-off and matched-yoked schedules, direct vigor gating,
matched-initial probe withdrawal, shuffled/dummy/wrong-lag controls, raw error,
partial correlation, deterministic-policy identification, reversal, an oracle
behavioral bound, the gain/noise grid, seed bootstrap intervals, and censor-aware
latency summaries.

This document is deliberately broader than that run. The following recommended
extensions were **not** executed and must not be inferred from the current artifact:

- evidence-clipping, learning-rate, and phase-length sensitivity sweeps;
- oracle-parameter predictive scoring separate from the oracle behavioral gate;
- a conditional action-innovation proxy beyond the reported partial correlation;
- a joint-seed bootstrap of the complete gain/noise calibration surface;
- a dense probe-frequency energy-recovery frontier and separate probe-cost figure;
- a prespecified acceptance margin for connected-phase homeostatic-error cost.

These are next-stage robustness checks, not completed confirmatory analyses.

## 1. Scientific question and claim boundary

The primary question is:

> Can a small online learner estimate action-outcome contingency from egocentric
> sensorimotor transitions and use that estimate directly to regulate behavioral
> engagement, without reward, a value function, or a learned policy?

The measured quantity should initially be called **sensorimotor influence**,
**action-outcome contingency**, or **action-conditioned predictive information**.
The toy result must not be described as classical dynamical-system controllability,
agency, learned helplessness, or a model of a particular animal.

The core estimand is on-policy and predictive:

```math
I(A_t; O_{t+1}\mid H_t).
```

It asks whether the actions actually sampled by the agent improve prediction of its
next observation given its available history. It does not, by itself, measure the
maximum control the agent could exercise, prove interventional causality under hidden
confounding, or establish that a deterministic policy has no physical effect.

## 2. Analytic benchmark

### 2.1 Generative system

Use the fully observed scalar system

```math
X_{t+1}=\rho X_t+\gamma_t A_t+\epsilon_t,
\qquad
\epsilon_t\sim\mathcal N(0,\sigma_\epsilon^2),
```

with

```math
A_t\sim\mathcal N(0,\sigma_a^2),
\qquad A_t\perp (X_t,\epsilon_t),
\qquad O_t=X_t.
```

The initial benchmark uses `rho = 0.8`, `sigma_epsilon = 0.2`, and three equal-length
phases of 1,500 transitions. The action scale must be stated in every result; if
`sigma_a = 1`, the default connected target is about 1.629 nats.

The independence assumption is essential. If actions are autocorrelated or are a
deterministic function of state, the formula below is no longer the benchmark for the
data actually collected.

### 2.2 Conditional distributions

In a stationary phase with constant `\gamma`, conditioning on both state and action
gives

```math
p_1(x'\mid x,a)
=\mathcal N(\rho x+\gamma a,\sigma_\epsilon^2).
```

Marginalizing the independent Gaussian action gives

```math
p_0(x'\mid x)
=\mathcal N
\left(\rho x,\sigma_\epsilon^2+\gamma^2\sigma_a^2\right).
```

Therefore

```math
\begin{aligned}
I(A_t;X_{t+1}\mid X_t)
&=H(X_{t+1}\mid X_t)-H(X_{t+1}\mid X_t,A_t)\\
&=\frac12\log
\frac{\sigma_\epsilon^2+\gamma^2\sigma_a^2}
{\sigma_\epsilon^2}\\
&=\frac12\log\left(
1+\frac{\gamma^2\sigma_a^2}{\sigma_\epsilon^2}
\right).
\end{aligned}
```

Natural logarithms are used, so the unit is nats. The result does not depend on
`\rho` under the stated assumptions. It is zero at `\gamma=0`, positive for any
nonzero gain, decreases with sensory noise, and is identical for `+\gamma` and
`-\gamma`.

For a stochastic state-dependent policy, the relevant state-local variance is
`\operatorname{Var}(A_t\mid X_t=x)`. For a deterministic policy
`A_t=f(H_t)`, the conditional mutual information is exactly zero because the action
contains no information beyond the conditioned history, even when its physical
coefficient `\gamma` is large. This is an identifiability fact, not estimator
failure.

### 2.3 Disconnected and yoked targets

The literal cable-off condition is

```math
X_{t+1}=\rho X_t+\epsilon_t.
```

Both true predictors then have variance `\sigma_\epsilon^2`, and the target
conditional information is zero. This condition changes sensory variance relative to
the connected phase and is not sufficient on its own.

The matched yoked condition is

```math
X_{t+1}=\rho X_t+\gamma \widetilde A_t+\epsilon_t,
\qquad
\widetilde A_t\sim\mathcal N(0,\sigma_a^2),
\qquad
\widetilde A_t\perp(A_t,X_t,\epsilon_t).
```

Here the focal action `A_t` is irrelevant, so

```math
p_1(x'\mid x,a)=p_0(x'\mid x)
=\mathcal N
\left(\rho x,\sigma_\epsilon^2+\gamma^2\sigma_a^2\right)
```

and the target is again zero. Unlike cable-off, the yoked phase preserves the
distribution of exogenous drive and, in stationarity, the marginal state variance:

```math
\operatorname{Var}(X)=
\frac{\gamma^2\sigma_a^2+\sigma_\epsilon^2}{1-\rho^2}.
```

This equality is exact in distribution, not pathwise in a finite realization. It also
depends on focal and yoked actions having the same distribution. Once vigor gates the
focal action, that equality is not automatic; see Section 8.4.

The stationary-variance expression assumes `abs(rho) < 1`.

### 2.4 Reversed coupling

Switching from `\gamma=+g` to `\gamma=-g` preserves the analytic influence
target. It does, however, make a previously learned forward model wrong until it
adapts. The expected qualitative sequence is therefore:

1. a large forward-model mismatch immediately after reversal;
2. possibly negative instantaneous loss advantage while the old action coefficient
   is misleading;
3. recovery to the same positive steady-state influence as before reversal.

This condition separates persistent action influence from conformity to a previously
expected action consequence.

## 3. Online estimators and what each can claim

### 3.1 Prequential evaluation order

For every transition, perform operations in this order:

1. observe the history `h_t` and choose `a_t`;
2. compute both predictive distributions without seeing `o_{t+1}`;
3. apply the action (or yoked drive) and observe `o_{t+1}`;
4. score both frozen pre-update predictions;
5. update the smoothed influence estimate;
6. only then update either predictor on this transition.

The per-transition losses and evidence are

```math
\ell_{1,t}=-\log p_1(o_{t+1}\mid h_t,a_t),
\quad
\ell_{0,t}=-\log p_0(o_{t+1}\mid h_t),
\quad
d_t=\ell_{0,t}-\ell_{1,t}.
```

The operational signal is

```math
q_{t+1}=(1-\alpha)q_t
+\alpha\operatorname{clip}(d_t,-d_{\max},d_{\max}).
```

Plots must align `q_{t+1}` with the transition ending at `t+1`. Neither model may
update from the outcome before its loss for that outcome is recorded.

### 3.2 Probabilistic Gaussian version

For a Gaussian predictor with mean `\widehat\mu_j` and variance
`\widehat\sigma_j^2`, use

```math
\ell_j=\frac12\left[
\log(2\pi\widehat\sigma_j^2)
+\frac{(x'-\widehat\mu_j)^2}{\widehat\sigma_j^2}
\right].
```

Each model needs its own calibrated predictive variance. Variance parameters or
residual-variance estimates must be read before, and updated after, scoring the
current transition. Enforce a documented positive floor to avoid a transient
near-zero variance producing unbounded loss. Report how often the floor is active.

When both mean and variance models have converged to the correct conditional
distributions, `\mathbb E[d_t]` equals the analytic conditional information. During
initial learning or after a switch, the loss gap also contains model mismatch. Call it
**prequential influence evidence** in those transient intervals, not an exact mutual
information estimate.

### 3.3 Fixed common variance version

If both models instead use the same fixed variance `s^2`, the log-normalization
terms cancel and

```math
d_t=\frac{e_{0,t}^2-e_{1,t}^2}{2s^2}.
```

After correct mean learning in a connected phase,

```math
\mathbb E[d_t]
=\frac{\gamma^2\sigma_a^2}{2s^2}.
```

Even if `s^2=\sigma_\epsilon^2`, this is
`\tfrac12\mathrm{SNR}`, whereas the information target is
`\tfrac12\log(1+\mathrm{SNR})`. A common fixed-variance implementation is therefore
a useful squared-error-gap debugger, but its numerical output must not be reported as
conditional mutual information. The two versions should have distinct metric names
in artifacts and result tables.

### 3.4 Clipping changes the estimand

In general,

```math
\mathbb E[\operatorname{clip}(d_t)]
\ne \operatorname{clip}(\mathbb E[d_t]).
```

Consequently a finite `d_{\max}` makes the steady-state mean of `q` a robust
operational score, not an exactly calibrated information estimate. Analytic
calibration must use the unclipped prequential `d_t`, or show that clipping is so
rare that the difference is negligible. Always report the clipped fraction by phase
and include a sensitivity analysis over `d_{\max}`.

### 3.5 Comparable models

The action-aware model necessarily has one extra useful input. Rule out a generic
capacity advantage by using all of the following:

- identical state features, initialization scheme, optimizer, update timing, and
  variance-learning rule;
- standardized or otherwise comparable feature scales;
- an action-aware coefficient initialized to zero so the richer mean model begins
  nested at the blind model;
- a same-dimensional dummy-input model receiving an independent draw with the same
  marginal distribution as the true action;
- a same-dimensional shuffled-action model;
- matched learning-rate sweeps fixed using development seeds, not selected on the
  final test seeds.

For SGD, “same scalar learning rate” does not by itself guarantee comparable effective
learning when feature scales differ.

## 4. Experiments

### 4.1 Experiment 1A: estimator-only cable switch

Use externally sampled iid Gaussian actions of fixed variance. The schedule is

```math
\gamma_t: 1\longrightarrow 0\longrightarrow 1.
```

This experiment validates online tracking but does not control sensory variance. Plot
the true coupling state, both pre-update losses, instantaneous `d_t`, smoothed
`q_t`, and action magnitude. Report steady-state phase summaries separately from
switch transients.

### 4.2 Experiment 1A-yoked: matched contingency switch

Keep the physical gain fixed and switch the driver

```math
X_{t+1}=\rho X_t+\gamma
\left[c_tA_t+(1-c_t)\widetilde A_t\right]+\epsilon_t,
\qquad
c_t:1\longrightarrow0\longrightarrow1.
```

Focal and yoked actions must be independent iid draws from the same Gaussian
distribution. Verify, rather than assume, that phase-wise drive variance and state
variance are matched within Monte Carlo uncertainty. The primary H1 comparison is
connected versus yoked in this estimator-only experiment.

### 4.3 Experiment 1A-reversal

Use

```math
\gamma_t:+1\longrightarrow-1\longrightarrow+1.
```

Report raw forward-model loss, the loss gap, and the learned action coefficient. A
valid influence estimator may show a transient failure response but must return to the
same positive steady-state level for either sign.

### 4.4 Experiment 1B: vigor gate

After Experiment 1A is frozen and passes its acceptance criteria, add the fixed
controller

```math
u_t=-K(x_t-x^*),
```

the vigor mapping

```math
m_t=m_{\min}+(1-m_{\min})
\operatorname{sigmoid}[\beta(q_t-\theta)],
```

and randomized probes

```math
a_t=\begin{cases}
s_t a_{\mathrm{probe}}, & z_t=1,\\
m_tu_t, & z_t=0,
\end{cases}
```

where `z_t\sim\operatorname{Bernoulli}(p_{\mathrm{probe}})` and
`s_t\in{-1,+1}` is an independent fair sign. The predictor receives the command
actually issued, including probes.

Direction and vigor remain conceptually separate: the fixed controller selects a
direction, while `q_t` changes only the command scale. Do not describe the gate as
reward-free and assumption-free; it contains the engineered assumption that action
cost should be reduced when action-outcome contingency is low.

Match paired conditions with the same probe stream throughout the initial connected
phase, then withdraw probes from one arm at disconnection while retaining them in
the other. This prevents the recovery contrast from being confounded by pre-switch
loss of identification. An additional never-probed deterministic-policy control may
demonstrate the broader identification failure. An oracle-coupling vigor gate is an
experimenter-side upper bound, not an input to the learned agent.

## 5. Prespecified hypotheses

- **H1:** steady-state influence evidence is higher when the focal action drives the
  world than when an independent yoked action drives a statistically matched world.
- **H2:** influence evidence falls after disconnection and rises after reconnection.
- **H3:** gating reduces focal action energy during uncontrollable phases.
- **H4:** fixed-amplitude probes reduce reconnection latency and the probability of
  remaining passive through the end of the phase.
- **H5:** raw action-aware forward-model error is not a consistent indicator across
  cable-off and yoked disconnection, even if it is diagnostic in either one condition.
- **H6:** shuffled and random-dummy actions do not provide a sustained predictive
  advantage.
- **H7:** detection becomes slower or less reliable as coupling decreases or sensory
  noise increases.
- **H8:** reversing the action mapping preserves steady-state influence but causes a
  large transient mismatch.

H5 is deliberately falsifiable. In the yoked system, the action-aware model has
irreducible residual variance
`\gamma^2\sigma_a^2+\sigma_\epsilon^2`, so raw error can remain higher than in a
connected system. In cable-off, its irreducible residual variance instead decreases
to `\sigma_\epsilon^2`. Raw error is therefore confounded by environmental noise
and need not move in a consistent “low control” direction. The differential loss gap
is intended to remove that particular confound; this toy does not establish that
forward error can never be useful.

## 6. Required controls

| Control | Exact comparison | Failure ruled out |
|---|---|---|
| Cable off | `\gamma=0` | Basic failure to track a missing actuator |
| Yoked drive | Independent matched `\widetilde A_t` replaces `A_t` | Detecting a quieter environment rather than contingency |
| Random dummy | Same-dimensional iid dummy input | Extra parameters alone lowering loss |
| Shuffled action | Within-phase permutation fixed before the run | Arbitrary use of the action marginal rather than alignment |
| Incorrect lag | `A_{t-k}` for prespecified noncausal/wrong lags | Exploiting broad temporal correlations or an off-by-one bug |
| Raw forward loss | `\ell_{1,t}` and squared residual alone | General surprise being relabeled as lost influence |
| Marginal correlation | `\operatorname{corr}(A_t,X_{t+1})` | Shows why passive dynamics/state feedback require conditioning |
| Partial correlation | Linear residual association after controlling `X_t` | Simple, transparent linear alternative |
| Probe withdrawal | Match initial probes, then set `p_{\mathrm{probe}}=0` | Whether rediscovery truly needs exploration |
| Deterministic policy | Disable all exogenous action variation | Demonstrates conditional-identifiability failure |
| Reversed gain | `+g` versus `-g` | Influence versus expected-direction success |
| Gain/noise grid | Several absolute gains `\gamma` and `\sigma_\epsilon` values | Binary schedule-specific detector |
| Oracle state | Gate with the true coupling state | Upper bound on behavioral performance |

The shuffled sequence must not cross phase boundaries; doing so can leak the coupling
schedule. With autocorrelated actions, use a permutation or circular shift exceeding
the prespecified correlation length. Include several incorrect lags and correct for
selection if the “best” lag is reported.

For the partial-correlation baseline, regress both `X_{t+1}` and `A_t` on `X_t`, then
correlate their residuals. In the correctly specified scalar Gaussian system, its
batch information equivalent is `-0.5 log(1-r_partial^2)`. This is an
experimenter-side diagnostic, not an online input to the agent.

## 7. Metrics and exact definitions

### 7.1 Analysis windows

Keep transition and steady-state analyses separate.

- **Transition window:** every step from the known switch until the next phase ends;
  used for latency and response plots.
- **Steady-state window:** the final 50% of each phase by default; used for phase means,
  calibration, and distribution comparisons.
- **Initialization:** the first phase may include an explicit warm-up interval, but
  warm-up transitions and model updates must be reported. Do not silently delete
  unusually poor early runs.

If a different guard interval is selected based on the EMA time constant

```math
\tau_q=-1/\log(1-\alpha),
```

freeze that rule before final test seeds. Do not move phase boundaries or analysis
windows after viewing results.

### 7.2 Estimation metrics

For every seed, compute:

1. phase-wise mean unclipped `d_t`;
2. phase-wise mean operational `q_t`;
3. phase-wise mean and median `\ell_0` and `\ell_1`;
4. the fraction of `d_t` samples clipped;
5. analytic-target error, `\overline d-I_\mathrm{analytic}`, for the calibrated
   probabilistic estimator;
6. coupled-versus-yoked ROC-AUC;
7. disconnection and reconnection latency;
8. false-active and false-passive fractions.

Compute one ROC-AUC within each seed using equal-length steady-state samples from
connected and yoked phases, then summarize the seed-level AUCs. Time steps may
contribute to the descriptive AUC, but the seed—not a time step—is the inferential
unit.

Across the gain/noise grid, fit the prespecified calibration regression

```math
\overline d_{s,g,n}=b_0+b_1 I_{g,n}+u_s+\eta_{s,g,n},
```

or report an equivalently prespecified seed-paired slope and intercept. Also report
root-mean-square calibration error. Do not fit calibration using clipped `q`.

### 7.3 Latency

Thresholds, smoothing, and dwell time must be frozen using development seeds. A
recommended deterministic rule is:

- estimate a reference connected level `q_{\mathrm{ref}}` from development seeds;
- set `\theta_{\mathrm{off}}=0.25q_{\mathrm{ref}}` and
  `\theta_{\mathrm{on}}=0.75q_{\mathrm{ref}}`;
- use a dwell of
  `W=\max(5,\lceil0.1\tau_q\rceil)` consecutive samples.

Then define:

- **disconnection latency:** number of transitions from the switch to the online
  confirmation sample (the `W`th consecutive sample) below
  `\theta_{\mathrm{off}}`;
- **reconnection latency:** number of transitions from the switch to the online
  confirmation sample (the `W`th consecutive sample) above
  `\theta_{\mathrm{on}}`.

Thresholds must not be re-estimated per test seed or per condition. If a crossing does
not occur before the next phase, the latency is right-censored; it must not be dropped.
Report recovery/detection probability and either Kaplan-Meier summaries or restricted
mean latency capped at the common phase horizon. For a simple paired bootstrap,
additionally report capped latency with failures assigned the horizon and show the
failure fraction beside it.

The estimator latency and behavioral-vigor latency are distinct endpoints. For vigor,
apply the same frozen-threshold principle to `m_t`, and label the result explicitly.

### 7.4 State-classification errors

Use the two thresholds as a hysteretic classifier:

- above `\theta_{\mathrm{on}}`: active/contingent;
- below `\theta_{\mathrm{off}}`: passive/noncontingent;
- between thresholds: retain the previous label.

After a frozen transition-exclusion interval, define:

- **false-active rate:** fraction of uncontrollable/yoked steps classified active;
- **false-passive rate:** fraction of connected steps classified passive.

Report transition latency separately so excluding a fixed transition interval does
not hide slow responses.

### 7.5 Behavioral metrics

For each seed and phase, compute:

```math
E=\sum_t a_t^2,
\qquad
\overline E=\frac1T\sum_t a_t^2,
\qquad
R_x=\frac1T\sum_t(x_t-x^*)^2.
```

Report at minimum:

- total and per-step action energy;
- disconnected/yoked-phase energy;
- homeostatic mean-squared error during connected phases;
- estimator and vigor recovery latency;
- fraction not recovered by the end of reconnection;
- energy-recovery curves over the prespecified probe-frequency grid;
- probe and non-probe energy separately.

These are experimenter-side metrics and must never be supplied to the agent as a
reward.

## 8. Principal failure modes and diagnostic tests

### 8.1 The adaptive null can absorb driven dynamics

The blind predictor is not necessarily a model of “what happens without action.” It
is a model of `p(x'\mid h)` under the behavior policy. If action is predictable from
history, the blind model can infer its consequence indirectly. Examples include:

- a deterministic controller `a_t=f(x_t)`;
- autocorrelated motor commands whose current value is predictable from past actions;
- an overly rich history that contains the random seed or controller state;
- slowly varying actions that are almost reconstructible from `x_t`.

In those cases a small loss gap does not imply small physical actuator gain. Diagnose
this by measuring action innovation conditional on predictor history, running iid
externally randomized actions in Experiment 1A, comparing multiple history lengths,
and including probes with independently randomized sign in Experiment 1B.

Conversely, an underfit blind model can create a spurious advantage if the extra
action feature merely compensates for missing nonlinear state dynamics. Dummy-input,
shuffled-action, capacity-matched nonlinear, and state-history controls are required
before extending beyond the linear toy.

### 8.2 Gate-induced loss of identifiability

The gate creates a feedback loop:

```math
q\downarrow\Rightarrow m\downarrow\Rightarrow
\operatorname{Var}(A\mid H)\downarrow\Rightarrow
\text{less evidence about coupling}.
```

At zero action innovation, return of physical coupling is statistically invisible.
This can create permanent passivity even with a perfect learner. A nonzero
`m_{\min}` helps only if it produces conditional action variation; a deterministic
scaled controller remains redundant with history. Randomized fixed-amplitude probes
are the clean identification mechanism. Log probe occurrence and sign, and verify
that probe amplitude does not depend on `q` or `m`.

For this audit, history includes any agent-internal state that deterministically
selects the command, including `q`, `m`, and current predictor parameters. Omitting
those variables from the blind predictor can create apparent action innovation, but
it is not a substitute for experimental randomization.

Also report `\operatorname{Var}(A_t\mid H_t)`, or a transparent residual-variance
proxy, alongside `q_t`. Recovery failure with negligible action innovation should
not be interpreted as evidence that the world remained disconnected.

### 8.3 Fixed-variance NLL is miscalibrated

A fixed common variance makes the signal monotone in squared-error advantage but not
equal to conditional information. A badly chosen value also rescales `q`, changing
gate thresholds and apparent latency. Validate the fixed-variance learner only as a
debug baseline. Use predictor-specific prequential variances for the information
claim, and test variance floors, adaptation rates, and clipping sensitivity.

After a connected-to-yoked switch, the action-aware predictor initially retains a
small variance and a nonzero action coefficient. It may receive extremely large NLL
and produce a negative `q` transient before adapting. That is expected model
mismatch. The steady-state target remains zero.

### 8.4 Yoking under a vigor gate is not automatically matched

The estimator-only yoked equality assumes both action streams are iid with fixed,
equal variance. Under gating, the focal action distribution changes with `q`, state,
and probe history. Simply drawing a fixed-variance yoked action no longer matches the
focal command distribution; drawing a yoked action scaled by the focal vigor can
reintroduce state-dependent coupling and a quieter-world confound.

Therefore:

- make estimator-only iid yoking the primary contingency control;
- label cable-off gating as a behavioral feedback demonstration rather than the
  matched-yoke proof;
- if a gated yoked result is reported, use a prespecified master-replay design or an
  independently generated driver with a documented matching rule;
- verify that the replayed driver is independent of the focal command conditional on
  the analysis history; two deterministic controllers responding to the same state
  can remain highly correlated even when only one drives the world;
- report phase-wise driver variance, state variance, lag spectrum, and state-driver
  dependence for both master and yoked arms;
- do not claim exact sensory matching from equality of nominal action variance alone.

### 8.5 History, hidden confounding, and causal language

Conditional predictive information is not automatically interventional causality.
A hidden variable that drives both motor command and sensory transition can give a
positive loss gap without an actuator effect. Experiment 1A avoids this by generating
the command independently and explicitly intervening on the simulated actuator.
Future closed-loop or biological interpretations need an explicit causal graph and
randomized perturbations.

The chosen history also defines the question. With `H_t=X_t`, the toy tests a
fully observed Markov system. Adding past observations/actions changes the null model,
the action innovation, and the estimand. Treat history length as a prespecified model
choice, not an innocuous architecture detail.

### 8.6 Adaptation is part of the measured response

The two models learn online, so `d_t` mixes environmental contingency with unequal
adaptation transients. A larger model may learn more slowly, and a model trained in a
previous phase can be actively misleading after a switch. Required diagnostics are:

- both learned mean coefficients over time;
- both predictive variances over time;
- pre-update residuals and NLLs separately;
- an oracle-parameter scoring analysis to show the target absent learning dynamics;
- learning-rate and phase-length sensitivity on development seeds.

Do not tune separate phases until the desired switch trace appears.

### 8.7 Numerical and timing failures

Guard explicitly against:

- scoring after updating on the same outcome;
- using `a_{t+1}` or `a_{t-1}` for the transition labeled `a_t`;
- sharing a mutable feature array between predictors;
- variance collapse, overflow, or silent clipping;
- random-number draw counts changing across paired conditions;
- including initialization transitions in one condition but not another;
- treating thousands of autocorrelated time steps as independent replicates.

The wrong-lag controls should fail, and a unit test should verify one transition by
hand with learning disabled.

## 9. Randomization, pairing, and inference

### 9.1 Independent random streams

Derive named random streams from each seed for at least:

- initial state;
- environmental noise;
- focal exploratory action;
- yoked action;
- probe occurrence;
- probe sign;
- dummy input;
- action shuffle/permutation.

Do not use one stateful generator for all sources. Otherwise adding a diagnostic draw
can silently change the environment and break pairing.

### 9.2 Paired conditions

Use identical seed IDs and share the streams that logically represent the same
counterfactual quantity. For example, persistent-probe and probe-withdrawal arms
should share initial state, environmental-noise innovations, and the complete
initial-phase probe stream before their schedules diverge. Connected and yoked arms
can share environmental noise and the presampled focal/yoked sequences.

Trajectories will diverge once conditions differ; paired seed design does not imply
identical realized sensory histories. Pair summaries by seed, never by time step.

### 9.3 Development and final seeds

Use separate development seeds to choose learning rates, `\alpha`, clipping,
variance floors, gate thresholds, dwell time, and plotting bandwidths. Freeze all
hyperparameters and analysis rules before the final run. Use at least 30 independent
held-out test seeds. Record the complete seed list and configuration hash in the
artifact.

Use development runs for an a priori power or precision calculation on the smallest
primary paired effect of interest; increase the final seed count above 30 if needed.
Do not stop adding seeds when significance is first reached.

### 9.4 Bootstrap confidence intervals

For a paired contrast, first compute the within-seed difference

```math
\Delta_s=M_{s,\mathrm{condition\ 1}}
-M_{s,\mathrm{condition\ 2}}.
```

Then resample seed IDs with replacement, preserving all conditions and grid cells for
the selected seed. For each of at least 10,000 bootstrap replicates, recompute the
complete statistic. Report the point estimate, 95% percentile interval, number of
seeds, and bootstrap method. A BCa interval may be added if specified before the final
run. Never bootstrap individual time steps.

For an unpaired summary, resample seeds. For a gain/noise heat map, resample the same
seed index jointly across all cells so differences and calibration curves preserve
their paired structure.

For censored latency, bootstrap a prespecified survival statistic such as restricted
mean latency and report event probability. Do not compute an ordinary mean after
discarding nonrecovering seeds.

## 10. Acceptance criteria

Freeze numeric equivalence margins using development simulations. Recommended primary
criteria for held-out seeds are:

1. **Analytic calibration:** across the gain/noise grid, the probabilistic unclipped
   loss gap has a calibration slope near one, intercept near zero, and low RMSE under
   prespecified tolerances. The fixed-variance gap is compared to its squared-error
   target, not the information target.
2. **Matched contingency:** the paired connected-minus-yoked steady-state contrast is
   positive with a 95% seed-bootstrap interval excluding zero.
3. **Null equivalence:** yoked, dummy, shuffled, and cable-off steady-state means lie
   within a prespecified practical-equivalence band around zero. Failure to reject a
   difference from zero is not evidence of equivalence.
4. **Reversibility:** both disconnection and reconnection event probabilities and
   censored latencies are reported; no failed seeds are omitted.
5. **Behavior:** gating reduces uncontrollable-phase energy without an unacceptable
   prespecified increase in connected-phase homeostatic error.
6. **Probe benefit:** relative to the paired arm whose probes are withdrawn at
   disconnection, persistent probes improve recovery probability or restricted-mean
   recovery latency, with their extra energy cost reported.
7. **Reversal:** steady-state `+\gamma` and `-\gamma` influence are practically
   equivalent while the immediate forward-model mismatch increases after reversal.
8. **No capacity artifact:** dummy and shuffled inputs do not reproduce the true
   action advantage.

Suggested starting equivalence margins are 0.05 nat or 10% of the default connected
analytic target, whichever is larger, but this choice must be checked for scientific
meaning on development seeds and then frozen. Do not redefine “success” after the
held-out results are known.

## 11. Required figures and tables

The minimum result set is:

1. one time-series figure with coupling/yoke schedule, both pre-update losses,
   unclipped `d_t`, `q_t`, vigor, and action magnitude;
2. connected, cable-off, and yoked seed-level `q` distributions with paired lines
   and confidence intervals;
3. persistent-probe versus probe-withdrawal recovery probability and censored
   latency, with energy cost;
4. a gain-by-noise heat map for analytic target, estimated mean, and calibration
   error;
5. a reversal panel showing gain sign, forward loss, `q`, and learned action
   coefficient;
6. a control table containing dummy, shuffled, wrong-lag, raw-error, partial-correlation,
   and oracle results.

Each figure caption must state the number of seeds, whether traces are seed means or
single illustrative runs, the smoothing used only for display, the analysis window,
and the confidence-interval construction. A smoothed display trace must never replace
unsmoothed data in statistical calculations.

## 12. Interpretation checklist

Before making the milestone claim, verify all of the following:

- losses were scored before updates;
- the calibrated model had predictor-specific variances;
- clipping did not silently define the calibration result;
- the focal and yoked commands were independent and distribution matched;
- coupling state was never supplied to the learner;
- actions retained conditional variation during identification periods;
- all hyperparameters were frozen before held-out seeds;
- seeds, not transitions, were treated as independent;
- nonrecovering runs were retained as censored observations;
- fixed-variance output was not labeled mutual information;
- the result was described as immediate sensorimotor influence, not general agency or
  dynamical controllability.

If these checks pass, the defensible milestone claim is:

> A reward-free online estimator distinguished the agent's own action-contingent
> sensory stream from a statistically matched yoked stream, and a direct vigor gate
> used that estimate to reduce engagement and later recover through randomized probes
> in a linear-Gaussian toy system.

That is a proof of concept for the learning signal and feedback loop, not yet a
biological mechanism or a general theory of control.
