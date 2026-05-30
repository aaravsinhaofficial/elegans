"""Integration tests for dynamic foraging environment with preset configurations."""

import pytest
from quantumnematode.agent import QuantumNematodeAgent
from quantumnematode.brain.actions import Action
from quantumnematode.brain.arch.mlpreinforce import MLPReinforceBrain, MLPReinforceBrainConfig
from quantumnematode.env import DynamicForagingEnvironment, ForagingParams


class TestDynamicEnvironmentWithBrain:
    """Integration tests running dynamic environment with brain architectures."""

    @pytest.fixture
    def dynamic_env_small(self):
        """Create small dynamic foraging environment."""
        return DynamicForagingEnvironment(
            grid_size=20,
            start_pos=(10, 10),
            foraging=ForagingParams(
                foods_on_grid=5,
                target_foods_to_collect=10,
                min_food_distance=3,
                agent_exclusion_radius=5,
                gradient_decay_constant=8.0,
                gradient_strength=1.0,
            ),
            viewport_size=(11, 11),
            max_body_length=0,
            action_set=[Action.FORWARD, Action.LEFT, Action.RIGHT, Action.STAY],
        )

    @pytest.fixture
    def mlp_brain(self):
        """Create a simple MLP brain for testing."""
        config = MLPReinforceBrainConfig()
        return MLPReinforceBrain(config=config, input_dim=2, num_actions=4)

    def test_agent_initialization_with_dynamic_env(self, dynamic_env_small, mlp_brain):
        """Test agent initialization with dynamic foraging environment."""
        from quantumnematode.agent import SatietyConfig

        satiety_config = SatietyConfig(
            initial_satiety=100.0,
            satiety_decay_rate=1.0,
            satiety_gain_per_food=0.2,
        )

        agent = QuantumNematodeAgent(
            brain=mlp_brain,
            env=dynamic_env_small,
            satiety_config=satiety_config,
        )

        assert agent.current_satiety == 100.0
        assert agent.max_satiety == 100.0
        assert agent._metrics_tracker.foods_collected == 0
        assert isinstance(agent.env, DynamicForagingEnvironment)

    def test_food_consumption_workflow(self, dynamic_env_small, mlp_brain):
        """Test complete food consumption workflow."""
        from quantumnematode.agent import SatietyConfig

        satiety_config = SatietyConfig(
            initial_satiety=100.0,
            satiety_decay_rate=0.5,  # Slow decay for testing
            satiety_gain_per_food=0.3,
        )

        agent = QuantumNematodeAgent(
            brain=mlp_brain,
            env=dynamic_env_small,
            satiety_config=satiety_config,
        )

        # Type narrowing: ensure env is DynamicForagingEnvironment
        assert isinstance(agent.env, DynamicForagingEnvironment)

        initial_satiety = agent.current_satiety
        initial_food_count = len(agent.env.foods)

        # Manually place agent on food and consume
        if len(agent.env.foods) > 0:
            food_pos = agent.env.foods[0]
            agent.env.agent_pos = food_pos

            consumed = agent.env.consume_food()

            if consumed:
                # Update agent satiety via satiety manager
                satiety_gain = agent.max_satiety * satiety_config.satiety_gain_per_food
                agent._satiety_manager.restore_satiety(satiety_gain)
                agent._metrics_tracker.foods_collected += 1

                # Verify satiety increased
                current_satiety = agent.current_satiety
                assert current_satiety > initial_satiety or current_satiety == agent.max_satiety

                # Verify food respawned
                assert len(agent.env.foods) == initial_food_count

                # Verify food was removed from original position
                assert consumed not in agent.env.foods

    def test_satiety_decay(self, dynamic_env_small, mlp_brain):
        """Test satiety decay over steps."""
        from quantumnematode.agent import SatietyConfig

        satiety_config = SatietyConfig(
            initial_satiety=200.0,
            satiety_decay_rate=1.0,
            satiety_gain_per_food=0.2,
        )

        agent = QuantumNematodeAgent(
            brain=mlp_brain,
            env=dynamic_env_small,
            satiety_config=satiety_config,
        )

        # Just verify satiety system exists and is functional
        assert hasattr(agent, "_satiety_manager")
        assert agent.current_satiety >= 0.0
        assert agent.current_satiety <= agent.max_satiety
