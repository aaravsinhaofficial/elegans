# Dynamic Foraging Environment Proposal

## Why

The current simulation environment supports only single-food-source navigation in fixed-size grids, limiting the training scenarios to simple goal-finding behavior. To enable more complex behavioral training (e.g., continuous foraging, predator avoidance, multi-objective decision-making), we need persistent, large-scale environments with multiple food sources that better simulate realistic C. elegans foraging ecology.

## What Changes

- **Add new `DynamicForagingEnvironment` class** that supports multiple simultaneous food sources with configurable spawning strategies
- **Implement gradient field superposition** where multiple food sources emit overlapping chemotaxis gradients that sum at each grid position
- **Add satiety-based termination** where agents have a hunger level that decays over time and can be replenished by consuming food
- **Implement viewport-based rendering** with a follow-camera system that centers on the agent instead of rendering the entire large environment
- **Add spatial distribution algorithms** (Poisson disk sampling) to ensure food sources are well-distributed across the environment
- **Introduce new efficiency metrics** for multi-food foraging including foraging efficiency rate (foods/steps) and distance efficiency per food item
- **Add exploration reward** for visiting previously unvisited grid cells to encourage search behavior
- **Create three preset configurations** (small, medium, large) for curriculum learning scenarios
- **Extend configuration system** with new environment parameters for food count, spawn intervals, satiety levels, and viewport size
- **Maintain backward compatibility** with existing single-goal `MazeEnvironment` and all brain architectures

## Impact

### Affected Specs

- **NEW**: `environment-simulation` - Creating new specification for environment system requirements
- **MODIFIED**: `configuration-system` - Adding environment configuration schema for dynamic foraging parameters

### Affected Code

- **Core Implementation**:

  - [`elegans/env.py`](../../../packages/elegans/elegans/env.py) - New `DynamicForagingEnvironment` class and `BaseEnvironment` refactoring
  - [`elegans/agent.py`](../../../packages/elegans/elegans/agent.py) - Extended reward calculation for multi-food scenarios and exploration bonus
  - [`elegans/utils/config_loader.py`](../../../packages/elegans/elegans/utils/config_loader.py) - New `DynamicEnvironmentConfig` and `SatietyConfig` classes

- **Supporting Systems**:

  - [`elegans/report/dtypes.py`](../../../packages/elegans/elegans/report/dtypes.py) - New metrics for foraging efficiency and distance efficiency
  - [`configs/examples/`](../../../configs/examples/) - Three new preset configuration files (dynamic_small.yml, dynamic_medium.yml, dynamic_large.yml)
  - [`scripts/run_simulation.py`](../../../scripts/run_simulation.py) - Environment initialization logic for dynamic vs static mode

- **Rendering**:

  - [`elegans/env.py`](../../../packages/elegans/elegans/env.py) - Viewport rendering methods with agent-centered camera

### Migration Path

- Existing configurations continue to work unchanged with `MazeEnvironment`
- Users opt into dynamic environments via new `environment_type: "dynamic"` config field
- All existing brain architectures compatible without modification
- Default behavior remains single-goal navigation

### Breaking Changes

None - fully backward compatible with existing simulations.

### Performance Considerations

- Gradient field computation scales with O(num_foods × grid_size) per step
- Large environments (100×100) with many foods (50+) may have ~10-20% performance overhead
- Viewport rendering reduces visual rendering cost for large grids
- Poisson disk sampling occurs only at food spawn time (minimal impact)

### Testing Strategy

- Unit tests for gradient superposition and distance calculations
- Integration tests with preset configurations (small, medium, large)
- Validation that existing configurations still work unchanged
- Performance benchmarks comparing single-food vs multi-food scenarios
