# Literature notes: online sensorimotor influence

Verified against primary paper, publisher, or official proceedings pages on 2026-07-17.
The literature supports the proposed toy experiment as a test of **action-conditioned predictive
information**. It does not, by itself, justify claims about classical controllability, agency, learned
helplessness, or a biological implementation of two learned predictors.

## What the loss gap estimates

Let the two ideal predictive distributions be

\[
p_1(o'\\mid h,a), \\qquad p_0(o'\\mid h),
\]

and evaluate both before learning from the current transition. With
(d=\\ell_0-\\ell_1=\\log p_1(o'\\mid h,a)-\\log p_0(o'\\mid h)),

\[
\\mathbb E[d]=I(A;O'\\mid H).
\]

This identity requires the predictors to represent the true conditional distributions and the
expectation to use the same joint distribution that defines the conditionals. With learned,
misspecified models, the running loss advantage is a **prequential predictive-score difference**, not
an exact mutual-information estimator. It can be negative, and differences in model capacity,
regularization, optimization, or adaptation speed can bias it. Action-shuffled and random-dummy
inputs are therefore essential controls.

For the linear-Gaussian toy model

\[
X\_{t+1}=\\rho X_t+\\gamma A_t+\\epsilon_t,
\\qquad A_t\\sim\\mathcal N(0,\\sigma_a^2),
\\qquad \\epsilon_t\\sim\\mathcal N(0,\\sigma\_\\epsilon^2),
\]

with (A_t) independent of (X_t) and (\\epsilon_t), the analytic target is

\[
I(A_t;X\_{t+1}\\mid X_t)
=\\frac12\\log\\left(1+\\frac{\\gamma^2\\sigma_a^2}{\\sigma\_\\epsilon^2}\\right).
\]

The probabilistic predictors need different calibrated residual variances for their NLL gap to attain
this target: the action-aware conditional variance is (\\sigma\_\\epsilon^2), whereas the action-blind
variance is (\\sigma\_\\epsilon^2+\\gamma^2\\sigma_a^2). Giving both predictors the same fixed variance
makes the loss gap a scaled squared-error advantage, useful for debugging but generally not equal to
conditional mutual information.

Once vigor controls the action distribution, the measured information also becomes policy-dependent.
If (A_t=f(H_t)) is deterministic, then (I(A_t;O\_{t+1}\\mid H_t)=0), even when actions have a large
physical effect. Randomized, fixed-amplitude probes create the conditional action variation needed
for identification and rediscovery. In yoked experiments, matching only the marginal action variance
may not match the sensory distribution under a state-dependent controller; paired replay or explicitly
matched innovations provide a stronger control.

## Causal interpretation: justified only under explicit assumptions

Conditional predictive information is observational. It becomes evidence for action-to-observation
influence in the proposed toy structural equation because the experimenter randomizes the action,
the environmental and yoked drives are independently generated, and the relevant state is observed.
Outside that setup, an unobserved variable can drive both action and later sensation, or an incomplete
history can leave common causes unblocked. Then (I(A;O'\\mid H)>0) need not be an interventional
effect.

Seitzer, Schölkopf, and Martius (2021) are the closest direct mathematical precedent. They define
state-local causal action influence as

\[
C_j(s)=I(S'\_j;A\\mid S=s),
\]

but embed it in an explicit causal graphical model and define control under interventions
(\\mathrm{do}(A\\sim\\pi(\\cdot\\mid s))) with a full-support policy. They then estimate a learned
transition model and use the influence signal for exploration and replay prioritization in RL. The
present two-predictor, online loss gap is a smaller prequential approximation under the behavior or
probe distribution; it is not automatically equivalent to their interventional, state-local quantity.

Recommended initial terminology is therefore **sensorimotor influence**, **action-outcome
contingency**, or **action-conditioned predictive information**. Reserve **causal influence** for the
randomized toy intervention and clearly state the assumptions.

## Neighboring concepts that should remain distinct

### Transfer entropy

Schreiber (2000) introduced transfer entropy as directed predictive information from one process's
past to another process's future, conditioned on the target's own past. The proposed one-step signal
is transfer-entropy-like when (H_t) contains the relevant sensory history. Transfer entropy is still
a property of a joint time-series distribution; it is not automatically an interventional causal effect
under hidden common causes, contemporaneous coupling, selection, or insufficient history.

### Empowerment

Klyubin, Polani, and Nehaniv (2005) define empowerment as the channel capacity of an agent's
actuation channel. In a one-step setting this is, schematically,

\[
\\max\_{\\pi(a\\mid s)} I(A;S'\\mid S=s).
\]

Mohamed and Rezende (2015) developed a scalable variational lower bound for empowerment-based
reasoning. Empowerment asks how much influence the agent **could** exercise after optimizing an
action distribution, often over multiple steps. The proposed estimator asks how informative the
actions **currently sampled** are about the next observation. It is one-step, realized,
distribution-dependent influence, not empowerment unless the maximization is added.

### Classical dynamical-system controllability

Classical controllability concerns which states are reachable under admissible input sequences (for a
linear system, via the controllability matrix). A one-step conditional-information score depends on
the action distribution, observation map, and noise. It should not initially be called classical
controllability. The toy model's invariance to (\\gamma\\mapsto-\\gamma) is useful here: both signs
carry equal conditional information, even though a previously learned forward model will show a
large transient mismatch after reversal.

### Raw forward-model prediction error

Raw action-aware prediction error measures surprise relative to the current model. It can spike when
coupling is removed or reversed and then shrink as the model adapts. The two-model loss advantage
asks the different question of whether the action continues to improve prediction after accounting
for history. Reversal should therefore produce high initial mismatch but positive influence after
relearning; disconnection should eventually produce little loss advantage.

## Biological evidence and its limits

### Matched master/yoked contingency in *Drosophila*

Yang et al. (2013) paired a "master" fly with a yoked fly. Heat began when the master rested for
more than one second; resuming locomotion immediately turned it off. The yoked fly received the
same heat sequence, but that sequence was unrelated to its own behavior. Yoked flies reduced walking
more than masters during training and subsequently walked more slowly and rested more. This is a
strong contingency control because thermal exposure is pair-matched.

The analogy to an independently driven toy world is conceptual, not exact. The biological protocol
uses an aversive heat stressor generated by another animal's behavior, not a Gaussian action with
matched state innovations, and the paper studies a richer learned-helplessness phenotype. A toy
loss-gap/vigor result should not itself be labeled learned helplessness.

### Motor-copy signals are biologically available

Kim, Fitzgerald, and Maimon (2015) recorded motor-related inputs in *Drosophila* optic-flow neurons
during voluntary flight turns. Their sign and latency were appropriate to suppress the expected visual
response, supporting an internal prediction of self-generated optic flow. Ji et al. (2021) showed that
the *C. elegans* AIY sensory-processing interneuron carries temperature and motor-state information;
RIM-dependent corollary discharge from the motor circuit helps sustain forward locomotion during
thermotaxis.

These papers establish motor-to-sensory or motor-to-sensory-processing signals. Neither demonstrates
online learning of competing action-aware/action-blind predictors, estimation of conditional mutual
information, or vigor gating from such an estimate.

### Zebrafish futility-induced passivity

Mu et al. (2019) withheld expected visual flow from fictively swimming larval zebrafish. Fish first
increased motor vigor, then became passive after tens of seconds. Hindbrain noradrenergic neurons
responded to failed swims; radial astrocyte calcium accumulated with repeated failures and causal
perturbations supported a role for those astrocytes in suppressing swimming. This strongly motivates
the architecture "fast mismatch -> slow accumulation -> behavioral suppression," but it does not
show that the nervous system compares two probabilistic predictors or computes mutual information.

Chen et al. (2025) resolved a delayed inhibitory pathway. Norepinephrine drives fast excitation and
delayed inhibition; the inhibitory arm involves alpha-1-adrenergic activation of astroglia,
calcium-dependent ATP release, extracellular ATP-to-adenosine conversion, and activation of
swim-suppressing hindbrain neurons. Pharmacological blockade implicated A2B adenosine receptors:
blocking A2B, but not A1 or A2A receptors, reduced futility-induced passivity. Thus "primarily
A2B-mediated" is supported, while "exclusively A2B-mediated" would overstate the evidence. The
published article carries a correction dated 21 May 2025.

### Closest computational comparison: 3M-Progress

The final NeurIPS 2025 paper by Keller et al. calls the method Model-Memory-Mismatch Progress
(3M-Progress). It compares an online forward model with a frozen prior learned in an ethological
environment, filters the mismatch over time, and uses the resulting scalar as an intrinsic reward for
a PPO actor-critic policy. The authors report stable active/passive cycling and strong behavioral and
neural-glial alignment with zebrafish data.

Calling it the "closest competitor" is an assessment, not a claim established by the paper. The key
technical distinction is real: 3M-Progress trains a policy with intrinsic reinforcement and relies on a
pretrained ecological prior (whose construction used a shaped task reward), whereas the proposed
toy system uses an online action-conditioned-versus-blind score to modulate a fixed controller
directly. "No extrinsic reward" should not be conflated with "no scalar reward or policy learning."

The final proceedings metadata also corrects the prompt's shorthand: the final title is *Intrinsic
Goals for Autonomous Agents: Model-Based Exploration in Virtual Zebrafish Predicts Ethological
Behavior and Whole-Brain Dynamics*, and the proceedings list Alyn **Kirsch**. The early arXiv
version used a different title and listed Alyn **Tornell**.

## Verified references

01. Seitzer, M., Schölkopf, B., & Martius, G. (2021). Causal Influence Detection for Improving
    Efficiency in Reinforcement Learning. *Advances in Neural Information Processing Systems*,
    **34**, 22905-22918. [Official proceedings](https://proceedings.neurips.cc/paper/2021/hash/c1722a7941d61aad6e651a35b65a9c3e-Abstract.html);
    [arXiv:2106.03443](https://arxiv.org/abs/2106.03443).
02. Schreiber, T. (2000). Measuring Information Transfer. *Physical Review Letters*, **85**(2),
    461-464. [doi:10.1103/PhysRevLett.85.461](https://doi.org/10.1103/PhysRevLett.85.461).
03. Klyubin, A. S., Polani, D., & Nehaniv, C. L. (2005). Empowerment: A Universal Agent-Centric
    Measure of Control. In *Proceedings of the 2005 IEEE Congress on Evolutionary Computation*,
    vol. 1, 128-135. [doi:10.1109/CEC.2005.1554676](https://doi.org/10.1109/CEC.2005.1554676).
04. Klyubin, A. S., Polani, D., & Nehaniv, C. L. (2005). All Else Being Equal Be Empowered. In
    *Advances in Artificial Life (ECAL 2005)*, LNCS 3630, 744-753.
    [doi:10.1007/11553090_75](https://doi.org/10.1007/11553090_75).
05. Mohamed, S., & Rezende, D. J. (2015). Variational Information Maximisation for Intrinsically
    Motivated Reinforcement Learning. *Advances in Neural Information Processing Systems*, **28**,
    2125-2133. [Official proceedings](https://proceedings.neurips.cc/paper/2015/hash/e00406144c1e7e35240afed70f34166a-Abstract.html);
    [arXiv:1509.08731](https://arxiv.org/abs/1509.08731).
06. Yang, Z., Bertolucci, F., Wolf, R., & Heisenberg, M. (2013). Flies Cope with Uncontrollable
    Stress by Learned Helplessness. *Current Biology*, **23**(9), 799-803.
    [doi:10.1016/j.cub.2013.03.054](https://doi.org/10.1016/j.cub.2013.03.054).
07. Kim, A. J., Fitzgerald, J. K., & Maimon, G. (2015). Cellular Evidence for Efference Copy in
    Drosophila Visuomotor Processing. *Nature Neuroscience*, **18**, 1247-1255.
    [doi:10.1038/nn.4083](https://doi.org/10.1038/nn.4083).
08. Ji, N., Venkatachalam, V., Rodgers, H. D., Hung, W., Kawano, T., Clark, C. M., Lim, M.,
    Alkema, M. J., Zhen, M., & Samuel, A. D. T. (2021). Corollary Discharge Promotes a Sustained
    Motor State in a Neural Circuit for Navigation. *eLife*, **10**, e68848.
    [doi:10.7554/eLife.68848](https://doi.org/10.7554/eLife.68848).
09. Mu, Y., Bennett, D. V., Rubinov, M., Narayan, S., Yang, C.-T., Tanimoto, M., Mensh, B. D.,
    Looger, L. L., & Ahrens, M. B. (2019). Glia Accumulate Evidence that Actions Are Futile and
    Suppress Unsuccessful Behavior. *Cell*, **178**(1), 27-43.e19.
    [doi:10.1016/j.cell.2019.05.050](https://doi.org/10.1016/j.cell.2019.05.050).
10. Chen, A. B., Duque, M., Rymbek, A., Dhanasekar, M., Wang, V. M., Mi, X., Tocquer, L.,
    Narayan, S., Marquez Legorreta, E., Eddison, M., Yu, G., Wyart, C., Prober, D. A., Engert, F.,
    & Ahrens, M. B. (2025). Norepinephrine Changes Behavioral State Through Astroglial Purinergic
    Signaling. *Science*, **388**(6748), 769-775.
    [doi:10.1126/science.adq5233](https://doi.org/10.1126/science.adq5233).
11. Keller, R., Kirsch, A., Pei, F., Pitkow, X., Kozachkov, L., & Nayebi, A. (2025). Intrinsic Goals
    for Autonomous Agents: Model-Based Exploration in Virtual Zebrafish Predicts Ethological
    Behavior and Whole-Brain Dynamics. *Advances in Neural Information Processing Systems*,
    **38**. [Official proceedings](https://papers.nips.cc/paper_files/paper/2025/hash/334507d67dd7a310e563eecafbd5fee6-Abstract-Conference.html);
    [arXiv:2506.00138](https://arxiv.org/abs/2506.00138).

## Bottom line for the toy project

The most defensible milestone claim is:

> A small online learner distinguishes the agent's action-contingent sensory stream from a
> statistically matched yoked stream and uses that predictive advantage to regulate vigor, with
> randomized probes enabling recovery after reconnection.

That claim is information-theoretically principled and biologically motivated. It remains a toy
demonstration of sensorimotor contingency, not evidence that an animal computes the same quantity
or that the mechanism constitutes classical controllability, agency, or learned helplessness.
