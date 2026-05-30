"""Tests for benchmark categorization."""

from datetime import UTC, datetime

import pytest
from elegans.benchmark.categorization import (
    determine_benchmark_category,
    get_category_directory,
)
from elegans.experiment.metadata import (
    BrainMetadata,
    EnvironmentMetadata,
    ExperimentMetadata,
    GradientMetadata,
    LearningRateMetadata,
    ResultsMetadata,
    RewardMetadata,
    SystemMetadata,
)

# Centralized list of all valid benchmark categories
VALID_CATEGORIES = [
    "foraging_small",
    "foraging_medium",
    "foraging_large",
    "predator_small",
    "predator_medium",
    "predator_large",
]


def create_test_experiment(env: EnvironmentMetadata, brain: BrainMetadata) -> ExperimentMetadata:
    """Create test experiment metadata."""
    return ExperimentMetadata(
        experiment_id="test_id",
        timestamp=datetime.now(UTC),
        config_file="test.yml",
        config_hash="abc123",
        environment=env,
        brain=brain,
        reward=RewardMetadata(
            reward_goal=0.2,
            reward_distance_scale=0.1,
            reward_exploration=0.05,
            penalty_step=0.01,
            penalty_anti_dithering=0.05,
            penalty_stuck_position=0.02,
            stuck_position_threshold=2,
            penalty_starvation=10.0,
            penalty_predator_death=10.0,
            penalty_predator_proximity=0.1,
        ),
        learning_rate=LearningRateMetadata(
            method="dynamic",
            initial_learning_rate=0.01,
        ),
        gradient=GradientMetadata(method="raw"),
        results=ResultsMetadata(
            total_runs=50,
            success_rate=0.9,
            avg_steps=40.0,
            avg_reward=120.0,
        ),
        system=SystemMetadata(
            python_version="3.12.0",
            device_type="cpu",
        ),
    )


class TestDetermineBenchmarkCategory:
    """Test benchmark category determination."""

    def test_foraging_small(self):
        """Test categorizing small foraging environment."""
        env = EnvironmentMetadata(grid_size=15, num_foods=10)
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.02)
        experiment = create_test_experiment(env, brain)

        assert determine_benchmark_category(experiment) == "foraging_small"

    def test_foraging_medium(self):
        """Test categorizing medium foraging environment."""
        env = EnvironmentMetadata(grid_size=50, num_foods=20)
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        assert determine_benchmark_category(experiment) == "foraging_medium"

    def test_foraging_large(self):
        """Test categorizing large foraging environment."""
        env = EnvironmentMetadata(grid_size=100, num_foods=50)
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.005)
        experiment = create_test_experiment(env, brain)

        assert determine_benchmark_category(experiment) == "foraging_large"

    def test_boundary_case_small_medium(self):
        """Test boundary between small and medium (grid_size=20)."""
        env = EnvironmentMetadata(grid_size=20, num_foods=10)
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        assert determine_benchmark_category(experiment) == "foraging_small"

    def test_boundary_case_medium_large(self):
        """Test boundary between medium and large (grid_size=50)."""
        env = EnvironmentMetadata(grid_size=50, num_foods=20)
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.001)
        experiment = create_test_experiment(env, brain)

        assert determine_benchmark_category(experiment) == "foraging_medium"

    def test_predator_small(self):
        """Test categorizing small predator environment."""
        env = EnvironmentMetadata(
            grid_size=20,
            num_foods=10,
            predators_enabled=True,
            num_predators=2,
        )
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        assert determine_benchmark_category(experiment) == "predator_small"

    def test_predator_medium(self):
        """Test categorizing medium predator environment."""
        env = EnvironmentMetadata(
            grid_size=50,
            num_foods=20,
            predators_enabled=True,
            num_predators=3,
        )
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        assert determine_benchmark_category(experiment) == "predator_medium"

    def test_predator_large(self):
        """Test categorizing large predator environment."""
        env = EnvironmentMetadata(
            grid_size=100,
            num_foods=50,
            predators_enabled=True,
            num_predators=5,
        )
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.005)
        experiment = create_test_experiment(env, brain)

        assert determine_benchmark_category(experiment) == "predator_large"

    def test_predators_disabled_uses_foraging_category(self):
        """Test that predators_enabled=False uses foraging categories."""
        env = EnvironmentMetadata(
            grid_size=30,
            num_foods=15,
            predators_enabled=False,
        )
        brain = BrainMetadata(type="mlpreinforce", learning_rate=0.01)
        experiment = create_test_experiment(env, brain)

        category = determine_benchmark_category(experiment)
        assert category == "foraging_medium"
        assert "predator" not in category


class TestGetCategoryDirectory:
    """Test category directory path generation."""

    @pytest.mark.parametrize("category", VALID_CATEGORIES)
    def test_category_directory_passthrough(self, category):
        """Directory for a category is just the category name."""
        assert get_category_directory(category) == category
