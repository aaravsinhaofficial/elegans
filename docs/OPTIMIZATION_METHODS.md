# Optimization Methods for Brain Architectures

This document provides guidance on which optimization methods work best for each brain architecture in the elegans project.

## Summary Table

| Architecture | Primary Method | Secondary Method | Success Rate | Notes |
|-------------|----------------|------------------|--------------|-------|
| MLPReinforceBrain | REINFORCE | Adam + LR schedule | 92% | Classic policy gradient |
| MLPPPOBrain | Clipped PPO | Adam | 97% | Actor-critic with GAE |
| MLPDQNBrain | DQN | Adam | 75% | Value-based, off-policy |
| SpikingReinforceBrain | Surrogate + REINFORCE | Adam | 63-78% | Surrogate gradients |
| HybridClassicalBrain | REINFORCE (per-stage) | Adam | — | Curriculum-driven |

## Detailed Findings

### Classical Neural Networks (MLPReinforceBrain)

#### Recommended: REINFORCE with baseline

The standard REINFORCE algorithm with a learned baseline works well for classical networks:

| Method | Success Rate |
|--------|-------------|
| PPO | 97% |
| REINFORCE + baseline | 92% |
| Raw REINFORCE | 78% |

#### Why REINFORCE works

1. **Simplicity** - Fewer hyperparameters than actor-critic methods
2. **Stability** - The baseline reduces variance effectively
3. **Efficiency** - Single network, no value function needed

#### Configuration

```yaml
brain:
  name: mlpreinforce
  config:
    hidden_dim: 64
    num_hidden_layers: 2
    learning_rate: 0.001
    baseline: 0.0
    baseline_alpha: 0.05  # Exponential moving average
    entropy_beta: 0.01    # Entropy regularization
    gamma: 0.99

gradient:
  method: clip
```

### PPO Brain (MLPPPOBrain)

#### Recommended: Clipped surrogate objective

PPO uses the clipped surrogate objective with GAE for advantage estimation. December 2025 benchmarks achieved **97.1% ± 1.2% success rate** on foraging small (20x20) with fast convergence (~14 episodes to 80% success).

#### Why PPO excels

1. **Stable updates** - Clipped objective prevents destructive policy updates
2. **Sample efficiency** - Multiple epochs per rollout improve data utilization
3. **Variance reduction** - GAE provides low-variance advantage estimates
4. **Fast convergence** - Learns effective policies in ~14 episodes

#### Configuration

```yaml
brain:
  name: mlpppo
  config:
    actor_hidden_dim: 64
    critic_hidden_dim: 64
    clip_epsilon: 0.2      # Clipping parameter
    gae_lambda: 0.95       # GAE lambda
    value_loss_coef: 0.5   # Value function weight
    entropy_coef: 0.01     # Entropy bonus
    learning_rate: 0.0003
    num_epochs: 4          # Epochs per update
    num_minibatches: 4     # Minibatches per epoch
    rollout_buffer_size: 2048
    max_grad_norm: 0.5
```

### Spiking Neural Networks (SpikingReinforceBrain)

#### Recommended: Surrogate gradients + REINFORCE

Spiking networks require special handling due to non-differentiable spike functions:

| Task | Method | Success Rate |
|------|--------|-------------|
| Foraging | Surrogate + REINFORCE | 78% |
| Predator evasion | Surrogate + REINFORCE | 63% |

#### Configuration

```yaml
brain:
  name: spikingreinforce
  config:
    hidden_size: 64
    num_steps: 10
    tau_mem: 10.0
    tau_syn: 5.0
    threshold: 1.0
    learning_rate: 0.001
    surrogate_gradient: fast_sigmoid  # Options: fast_sigmoid, atan, piece_wise
    beta: 5.0  # Surrogate gradient sharpness
```

## Selection Guidance

Use this decision tree to choose an optimization method:

```text
Is it a spiking network?
├── YES → Use Surrogate Gradients + REINFORCE
│         - Surrogate enables backprop through spikes
│         - REINFORCE handles non-differentiable reward
│
└── NO → Classical MLP/PPO
         ├── Want best performance? → PPO (Recommended)
         │   - 97% success rate on foraging
         │   - Fast convergence (~14 episodes)
         │   - Stable training with clipped objective
         │
         └── Want simplicity? → REINFORCE with baseline
             - Single network, fewer hyperparameters
             - 92% success on foraging
```

## Hyperparameter Recommendations

### Learning Rates

| Architecture | Recommended LR | Range |
|-------------|----------------|-------|
| MLPReinforceBrain | 0.001 | 0.0001 - 0.01 |
| MLPPPOBrain | 0.0003 | 0.0001 - 0.001 |
| SpikingReinforceBrain | 0.001 | 0.0001 - 0.01 |
| MLPDQNBrain | 0.001 | 0.0001 - 0.01 |

### CMA-ES Parameters (evolutionary optimization)

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| Population size | 20 | 4 + 3\*log(n_params) |
| Initial sigma | 0.5 | Start exploring broadly |
| Max generations | 500 | Increase for complex tasks |

### Entropy Coefficients

| Architecture | Entropy Coef | Notes |
|-------------|--------------|-------|
| MLPReinforceBrain | 0.01 | Encourages exploration |
| MLPPPOBrain | 0.01 | Standard value |
| SpikingReinforceBrain | 0.005 | Less needed with spikes |

## Common Pitfalls

### Classical Networks

1. **No baseline** - Raw REINFORCE has high variance
2. **Learning rate too high** - Causes policy collapse
3. **No entropy regularization** - Premature convergence

### Spiking Networks

1. **Wrong surrogate choice** - Fast sigmoid works best empirically
2. **Tau values too small** - Information doesn't propagate
3. **Threshold too high** - Neurons never fire

## Experimental Results

### Foraging Task (Small, 20x20)

| Method | Architecture | Success | Learning Speed |
|--------|-------------|---------|----------------|
| Clipped PPO | MLPPPOBrain | 97% | 14 episodes |
| REINFORCE | MLPReinforceBrain | 92% | 85 episodes |
| Surrogate | SpikingReinforceBrain | 78% | 150 episodes |

### Predator Evasion Task (Small, 20x20)

| Method | Architecture | Success | Survival |
|--------|-------------|---------|----------|
| REINFORCE | MLPReinforceBrain | 84% | 88% |
| Surrogate | SpikingReinforceBrain | 63% | 71% |
