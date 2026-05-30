# Contributing to elegans

Thank you for your interest in contributing to the elegans simulation project! This guide will help you get started with development and contributing to the project.

## 🚀 Development Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) for dependency management

### 1. Install uv

```bash
brew install uv
```

### 2. Clone and Setup

```bash
git clone https://github.com/SyntheticBrains/nematode.git
cd nematode
```

### 3. Install Dependencies

```bash
uv sync --extra torch --extra pixel
```

Available extras:

- `torch` — PyTorch for classical and spiking neural network brains
- `pixel` — Pygame-based pixel renderer

### 4. Environment Configuration

Copy the environment template:

```bash
cp .env.template .env
```

Edit `.env` as needed for your local configuration.

## 🛠️ Development Tools

### Pre-commit Hooks

We use [pre-commit](https://pre-commit.com/) to maintain code quality. Install the hooks:

```bash
uv run pre-commit install
```

Run checks manually:

```bash
uv run pre-commit run -a
```

### Code Quality Tools

The project uses several tools for code quality:

- **Ruff**: Fast Python linter and formatter
- **Pyright**: Static type checker
- **Pytest**: Testing framework

Configuration is in `pyproject.toml`:

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ['ALL']
ignore = ['TD002', 'TD003']

[tool.ruff.lint.pydocstyle]
convention = "numpy"
```

## 🧠 Architecture Overview

### Brain Architectures

The project supports the following brain architectures:

**Classical (MLP):**

1. **MLPReinforceBrain** (`mlpreinforce`): MLP with policy gradients (REINFORCE)
2. **MLPDQNBrain** (`mlpdqn`): MLP with Deep Q-Network
3. **MLPPPOBrain** (`mlpppo`): MLP actor-critic with PPO — best classical architecture

**Biologically-Inspired:**

4. **SpikingReinforceBrain** (`spikingreinforce`): LIF spiking neural network with surrogate gradients

**Hybrid:**

5. **HybridClassicalBrain** (`hybridclassical`): Classical MLP reflex + classical cortex MLP + classical critic with mode-gated fusion and 3-stage curriculum

Each brain architecture follows a common interface defined in `elegans.brain.arch`.

## 🔧 Development Workflows

### Running Tests

The project has three tiers of tests:

#### Unit & Integration Tests (default)

Run automatically on every commit (pre-commit hook) and PR:

```bash
uv run pytest
```

#### Smoke Tests

End-to-end tests that run the actual entry-point scripts (`run_simulation.py`, `run_evolution.py`) with minimal episodes to verify they don't crash. Run on every PR via CI but excluded from pre-commit hooks to keep commits fast:

```bash
uv run pytest -m smoke -v
```

#### Nightly E2E Tests

Full training sessions that assert success rates fall within established benchmark ranges (derived from experiment logbooks). Run automatically at 3 AM UTC daily via GitHub Actions and can be triggered manually from the Actions tab:

```bash
# Run all nightly tests (slow — full training sessions)
uv run pytest -m nightly -v

# Run a single config
uv run pytest -m nightly -k "foraging_small" -v
```

Benchmark ranges are defined in `packages/elegans/tests/elegans_tests/e2e_benchmarks.json`. When updating configs or training parameters, you may need to update these ranges based on new experiment results.

### Running Simulations

#### Development Testing

##### Testing with Dynamic Foraging Environment

```bash
# Best classical brain (MLP PPO)
uv run ./scripts/run_simulation.py --runs 5 --config ./configs/examples/mlpppo_foraging_small.yml --theme emoji

# Spiking brain on foraging
uv run ./scripts/run_simulation.py --runs 5 --config ./configs/examples/spikingreinforce_foraging_small.yml --theme emoji
```

##### Testing with Predators in Dynamic Foraging Environment

```bash
# MLP PPO on predator evasion
uv run ./scripts/run_simulation.py --runs 5 --config ./configs/examples/mlpppo_predators_small.yml --theme emoji

# Spiking brain on predator evasion
uv run ./scripts/run_simulation.py --runs 5 --config ./configs/examples/spikingreinforce_predators_small.yml --theme emoji
```

### Experiment Tracking and Benchmarks

The project includes a comprehensive experiment tracking and benchmark management system to facilitate reproducibility and performance comparison.

#### Experiment Tracking

Track any simulation run automatically with the `--track-experiment` flag:

```bash
# Run with experiment tracking
uv run ./scripts/run_simulation.py \
  --config configs/examples/mlpppo_foraging_medium.yml \
  --runs 50 \
  --track-experiment
```

This saves complete metadata to `experiments/<experiment-id>.json` including:

- Configuration file and hash
- Git commit, branch, and dirty state
- System information and dependency versions
- Brain and environment parameters
- Complete results and performance metrics
- Export paths for plots and CSV files

#### Querying Experiments

Use the experiment query CLI to explore your tracked experiments:

```bash
# List recent experiments
uv run scripts/experiment_query.py list

# Filter by environment or brain type
uv run scripts/experiment_query.py list --env-type dynamic --brain-type modular

# Show detailed experiment info
uv run scripts/experiment_query.py show <experiment-id>

# Compare two experiments
uv run scripts/experiment_query.py compare <exp-id-1> <exp-id-2>

# Export as JSON for analysis
uv run scripts/experiment_query.py show <experiment-id> --format json > results.json
```

#### Submitting Benchmarks

When you achieve noteworthy results, submit them as benchmarks:

**Method 1: From existing experiment**

```bash
# Submit a tracked experiment as benchmark
uv run scripts/benchmark_submit.py submit <experiment-id> \
  --contributor "Your Name" \
  --github "your-username" \
  --notes "Brief description of optimization approach"
```

**Method 2: Direct from simulation**

```bash
# Run and submit as benchmark in one step
uv run scripts/run_simulation.py \
  --config configs/examples/mlpppo_foraging_medium.yml \
  --runs 50 \
  --save-benchmark \
  --benchmark-notes "Your optimization approach"
```

The CLI will interactively prompt for contributor information if not provided via flags.

#### Benchmark Quality Standards

To ensure benchmark quality and reproducibility, submissions must meet these criteria:

**Required**:

- Minimum 50 simulation runs for statistical significance
- Clean git state (no uncommitted changes)
- Complete contributor information
- Valid configuration file

**Recommended**:

- High success rate (category-dependent)
- Meaningful optimization notes explaining your approach
- Standard environment configurations
- Documented novel techniques or insights

#### Benchmark Workflow

NematodeBench requires multiple independent training sessions for scientific rigor:

1. **Develop and Test**: Experiment with different configurations, learning rates, and brain architectures
2. **Run Multiple Sessions**: Run 10+ independent training sessions with `--track-experiment`
3. **Submit Benchmark**: Use `benchmark_submit.py` to aggregate sessions into a benchmark
4. **Create PR**: Add the generated benchmark JSON and artifacts to git
5. **Verification**: Maintainers will verify reproducibility and merge

Example workflow:

```bash
# 1. Run 10+ independent training sessions
for session in {1..10}; do
    uv run scripts/run_simulation.py \
        --config configs/my_config.yml \
        --runs 50 \
        --track-experiment
done

# 2. Submit all sessions together
uv run scripts/benchmark_submit.py \
    --experiments experiments/* \
    --category foraging_medium/classical \
    --contributor "Jane Doe" \
    --github "janedoe" \
    --notes "Tuned learning rate schedule with adaptive exploration"

# 3. Regenerate leaderboards
uv run scripts/benchmark_submit.py regenerate

# 4. Create PR
git add benchmarks/foraging_medium/classical/*.json
git add artifacts/experiments/
git add README.md docs/nematodebench/LEADERBOARD.md
git commit -m "Add benchmark: Adaptive exploration for foraging medium"
git push origin feature/my-benchmark
```

#### Viewing Leaderboards

Check current benchmark standings:

```bash
# View summary of all categories
uv run scripts/benchmark_submit.py leaderboard

# View specific category
uv run scripts/benchmark_submit.py leaderboard --category foraging_medium/classical

# Regenerate leaderboard documentation
uv run scripts/benchmark_submit.py regenerate
```

See [BENCHMARKS.md](BENCHMARKS.md) for complete leaderboards and detailed submission guidelines.

#### Experiment Logbooks

For documenting analysis and insights from experiment series, use the logbook system in `docs/experiments/`:

```text
docs/experiments/
├── README.md                    # Index and workflow guide
├── templates/
│   └── experiment.md            # Template for new logbooks
└── logbooks/
    ├── 001-quantum-predator-optimization.md  (historical)
    ├── 002-evolutionary-parameter-search.md
    ├── 003-spiking-brain-optimization.md
    ├── ...
    └── supporting/              # Detailed appendix data per logbook
```

Logbooks numbered 001 and 008 document earlier quantum-architecture experimentation
and are preserved as historical records.

**Key distinction from auto-tracking:**

| System | Location | Git Tracked | Purpose |
|--------|----------|-------------|---------|
| Auto-tracking | `experiments/*.json` | No | Raw metadata from every run |
| Evolution results | `evolution_results/` | No | All evolution run outputs |
| Artifacts | `artifacts/` | Yes | Curated outputs referenced in logbooks |
| Logbooks | `docs/experiments/logbooks/` | Yes | Human analysis and narrative |
| Benchmarks | `benchmarks/` | Yes | Top-performing submissions |

To create a new logbook:

1. Copy `docs/experiments/templates/experiment.md` to `docs/experiments/logbooks/NNN-name.md`
2. Use the next sequential number
3. Update the index in `docs/experiments/README.md`
4. Reference session IDs from `artifacts/experiments/` or `artifacts/evolutions/` for reproducibility

### Evolutionary Optimization

For parameter optimization without gradient-based learning, use the evolution script:

```bash
# CMA-ES optimization
uv run python scripts/run_evolution.py \
  --config configs/examples/evolution_mlpreinforce_foraging_small.yml \
  --algorithm cmaes \
  --generations 50 \
  --population 20 \
  --episodes 15 \
  --parallel 4

# Genetic Algorithm
uv run python scripts/run_evolution.py \
  --config configs/examples/evolution_mlpreinforce_foraging_small.yml \
  --algorithm ga \
  --generations 50 \
  --population 30 \
  --episodes 15 \
  --parallel 4
```

Results are saved to `evolution_results/<timestamp>/`:

- `best_params_<timestamp>.json` - Best parameters found
- `history_<timestamp>.csv` - Fitness history per generation
- `checkpoint_gen<N>.pkl` - Checkpoints every 10 generations

Resume from checkpoint:

```bash
uv run python scripts/run_evolution.py \
  --config configs/examples/evolution_mlpreinforce_foraging_small.yml \
  --resume evolution_results/20251209_123456/checkpoint_gen20.pkl \
  --generations 50
```

### Adding New Features

#### Adding a New Brain Architecture

1. Create a new file in `packages/elegans/elegans/brain/arch/`
2. Inherit from the appropriate base class (e.g. `ClassicalBrain`)
3. Implement required methods:
   - `run_brain()`: Execute brain and return actions
   - `learn()`: Update parameters based on rewards
   - `update_parameters()`: Low-level parameter updates

Example structure:

```python
from elegans.brain.arch import ClassicalBrain

class MyNewBrain(ClassicalBrain):
    def run_brain(self, params, reward=None, **kwargs):
        # Implement brain execution logic
        pass

    def learn(self, params, reward, **kwargs):
        # Implement learning logic
        pass
```

4. Add configuration class:

```python
from elegans.brain.arch.dtypes import BrainConfig

class MyNewBrainConfig(BrainConfig):
    # Define configuration parameters
    pass
```

5. Update `__init__.py` files to export new classes
6. Add tests in the appropriate test directory

#### Adding New Environment Features

1. Extend `elegans.env` classes (base class: `BaseEnvironment`, main implementation: `DynamicForagingEnvironment`)
2. Ensure compatibility with `BrainParams` interface
3. Add visualization support for new features
4. Update environment state encoding for brain input
5. Add tracking for new metrics in `EpisodeTracker`
6. Create corresponding plots and CSV exports

Example for adding a new foraging feature:

```python
# In elegans/env/dynamic_foraging.py
class ExtendedForagingEnvironment(DynamicForagingEnvironment):
    def __init__(self, temperature_variation: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.temperature_variation = temperature_variation

    def get_state_vector(self) -> list[float]:
        state = super().get_state_vector()
        if self.temperature_variation:
            state.append(self.get_temperature_at_position())
        return state
```

### Code Style Guidelines

1. **Type Hints**: Use comprehensive type hints

   ```python
   def my_function(param: int, optional: str | None = None) -> list[float]:
       return [1.0, 2.0]
   ```

2. **Docstrings**: Use NumPy-style docstrings

   ```python
   def compute_gradient(params: dict[str, float]) -> list[float]:
       """
       Compute parameter gradients.

       Parameters
       ----------
       params : dict[str, float]
           Parameter values.

       Returns
       -------
       list[float]
           Computed gradients for each parameter.
       """
   ```

3. **Error Handling**: Use descriptive error messages

   ```python
   if not isinstance(params, dict):
       error_message = f"Expected dict for params, got {type(params)}"
       logger.error(error_message)
       raise TypeError(error_message)
   ```

4. **Logging**: Use structured logging

   ```python
   from elegans.logging_config import logger

   logger.info(f"Training episode {episode} completed with reward {reward}")
   ```

## 🧪 Testing Guidelines

### Unit Tests

- Test individual functions and methods
- Use meaningful test names: `test_policy_gradient_with_valid_params`
- Mock external dependencies

### Integration Tests

- Test complete workflows
- Use small configurations for faster execution

### Performance Tests

- Benchmark critical paths
- Monitor memory usage
- Test scalability with network size

## 🚀 Contribution Areas

### High Priority

1. **Dynamic Foraging Enhancements**

   - Temperature gradient integration
   - Food quality variations and preferences
   - Social foraging behaviors (multi-agent)
   - Realistic chemotaxis modeling

2. **Advanced Learning Algorithms**

   - SOTA RL baselines (SAC, TD3)
   - Meta-learning across environment variations
   - Transfer learning from simple to complex foraging

3. **Visualization and Analysis**

   - Real-time foraging trajectory visualization
   - Satiety and efficiency heatmaps
   - Comparative analysis tools across architectures

### Medium Priority

1. **Environment Extensions**

   - Multi-agent cooperative foraging
   - Competitive foraging scenarios
   - Continuous action spaces

2. **Monitoring and Logging**

   - Advanced foraging strategy detection
   - Performance metrics tracking per food type
   - Experiment management and reproducibility

### Documentation

1. **API Documentation**

   - Complete docstring coverage
   - Usage examples
   - Best practices guides

2. **Tutorials**

   - Getting started tutorials
   - Advanced usage patterns
   - Research applications

## 📝 Pull Request Process

1. **Fork and Branch**

   ```bash
   git checkout -b feature/my-new-feature
   ```

2. **Develop and Test**

   - Write tests for new functionality
   - Ensure all tests pass
   - Run pre-commit checks

3. **Documentation**

   - Update docstrings
   - Add usage examples
   - Update README if needed

4. **Submit PR**

   - Clear description of changes
   - Link to related issues
   - Include testing instructions

### PR Checklist

- [ ] All tests pass
- [ ] Pre-commit checks pass
- [ ] Documentation updated
- [ ] Type hints added
- [ ] Backward compatibility maintained
- [ ] Performance impact considered

## 🔬 Research Directions

### Bio-inspired Reinforcement Learning

- **Multi-objective Foraging**: Balancing food intake, survival, and exploration
- **Hybrid Architectures**: Reflex/cortex/critic decompositions and modular policies
- **Spiking Networks**: Surrogate gradient learning in biologically realistic neurons

### Biological Modeling

- **Neural Modeling**: More accurate modeling of C. elegans chemosensory neurons and interneurons
- **Behavioral Patterns**: Realistic nematode foraging behaviors (area-restricted search, klinokinesis)
- **Multi-scale Modeling**: From molecular signaling to behavioral strategies
- **Satiety and Homeostasis**: Realistic modeling of internal state management

### Algorithm Development

- **Optimization Techniques**: Advanced parameter optimization methods for foraging
- **Scalability**: Methods for larger networks and complex environments
- **Transfer Learning**: Strategies for adapting from simple to complex foraging scenarios

## 🤝 Community

### Getting Help

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and general discussion
- **Email**: Direct contact with maintainers

### Code of Conduct

We are committed to providing a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## 📄 License

This project is licensed under the Apache License 2.0. See [LICENSE](LICENSE) for details.

______________________________________________________________________

Thank you for contributing to the elegans simulation project! Your contributions help advance the field of bio-inspired reinforcement learning and computational biology.
