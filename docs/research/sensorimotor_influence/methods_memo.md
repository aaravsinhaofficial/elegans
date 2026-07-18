# Methods Memo: Reward-Free Online Sensorimotor Influence

## Question

Can a small online learner infer whether its own sampled actions help explain its next
observation, and use that estimate to regulate vigor without reward, value learning,
or policy optimization?

## Toy world and analytic target

The fully observed scalar environment is

```math
X_{t+1}=0.8X_t+\gamma_t A_t+\epsilon_t,
\qquad \epsilon_t\sim\mathcal N(0,0.2^2).
```

In estimator-only runs, `A_t` is independently sampled from `N(0, 1)`. The canonical
schedule has 1,500 transitions per phase and switches `gamma: 1 -> 0 -> 1`. The
strong control replaces the middle drive with an independent, distribution-matched
action. It therefore preserves environmental drive variance while removing the
relationship to the focal command. For independent Gaussian actions, the target is

```math
I(A_t;X_{t+1}\mid X_t)
=\tfrac12\log(1+\gamma^2\sigma_a^2/\sigma_\epsilon^2),
```

which is 1.629 nats at the default connected gain and zero when disconnected or
yoked. Reversing the gain leaves the target unchanged.

## Online learner

Both predictors have three scalar weights, the same learning rate, and their own
adaptive residual variance:

- blind features: `[x_t, 1, 0]`;
- aware features: `[x_t, 1, a_t]`.

Matched-capacity dummy, within-phase-shuffled-action, and wrong-lag predictors replace
the third input. Means use a normalized online delta rule with rate 0.015. Variances
use a pre-update squared-residual EMA with rate 0.01 and bounds `[1e-4, 100]`.

Every transition is predicted and scored before either model sees its outcome. The
unclipped evidence is `d_t = NLL_blind - NLL_aware`. The operational signal is

```math
q_{t+1}=0.99q_t+0.01\operatorname{clip}(d_t,-8,8).
```

Analytic calibration uses the unclipped loss gap, not `q`, because clipping changes
the estimand. The runner records the clipped fraction and variance-floor incidence.

## Vigor experiment

Direction comes from a fixed controller `u_t = 0.5(1-x_t)`. The learned module sets

```math
m_t=0.01+0.99\,\sigma(30(q_t-0.06)),
\qquad a_t=m_tu_t.
```

All behavioral arms share 300 initial randomized calibration transitions and the
same `p = 0.03` maintenance probes throughout the initial connected phase. At
disconnection, the withdrawal arm loses probes for the remainder of the run, while
the persistent-probe arms use Bernoulli rates 0.03 or 0.05 with random sign and unit
amplitude. This makes pre-switch action and vigor exactly paired and isolates whether
experimentation is required to rediscover restored coupling. Comparators are an
ungated controller and an oracle gate given the true coupling phase. Evaluation
energy and homeostatic error are never supplied to the agent.

## Inference and reproducibility

Hyperparameters were frozen on development seeds before the final run. The confirmatory
run uses seeds `751000` through `751029`; robustness uses the separate paired block
`761000` through `761029`.
The steady-state window is the final 50% of each phase. Transitions remain in latency
plots. Censored recoveries are retained at the phase horizon and accompanied by their
failure fraction. Confidence intervals use 10,000 percentile-bootstrap resamples of
seed IDs. Time points are never treated as independent inferential observations.

The runner saves all configurations, exact seed lists, a SHA-256 configuration hash,
seed-level tables, and representative raw traces. Source and tests live in
`elegans.sensorimotor_influence` and
`test_sensorimotor_influence.py`, respectively.
