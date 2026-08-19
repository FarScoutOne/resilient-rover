import pytest

from resilient_rover.envs.line_world_env import LineWorldEnv


@pytest.fixture
def env():
    environment = LineWorldEnv()
    environment.reset(seed=42)
    return environment

def test_reset_returns_valid_observation(env):
    observation, info = env.reset(seed=42)

    assert 0 <= observation["rover"] < env.size
    assert 0 <= observation["target"] < env.size
    assert observation["rover"] != observation["target"]
    assert observation["battery"] == env.max_battery
    assert info["step_count"] == 0

def test_wait_action(env):
    env._rover_location = 4
    env._target_location = 8
    env._battery_level = 20
    env._step_count = 0

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation["rover"] == 4
    assert observation["battery"] == 19
    assert info["step_count"] == 1
    assert reward == -1
    assert terminated is False
    assert truncated is False

def test_move_right(env):
    env._rover_location = 4
    env._target_location = 8
    env._battery_level = 20
    env._step_count = 0

    observation, reward, terminated, truncated, info = env.step(1)

    assert observation["rover"] == 5
    assert observation["battery"] == 19
    assert info["step_count"] == 1
    assert reward == -1
    assert terminated is False
    assert truncated is False

def test_move_left(env):
    env._rover_location = 4
    env._target_location = 8
    env._battery_level = 20
    env._step_count = 0

    observation, reward, terminated, truncated, info = env.step(2)

    assert observation["rover"] == 3
    assert observation["battery"] == 19
    assert info["step_count"] == 1
    assert reward == -1
    assert terminated is False
    assert truncated is False

def test_rover_cannot_move_below_zero(env):
    env._rover_location = 0
    env._target_location = 8
    env._battery_level = 20
    env._step_count = 0

    observation, reward, terminated, truncated, info = env.step(2)

    assert observation["rover"] == 0
    assert observation["battery"] == 19
    assert info["step_count"] == 1
    assert reward == -1
    assert terminated is False
    assert truncated is False

def test_rover_cannot_move_above_world_limit(env):
    env._rover_location = env.size - 1
    env._target_location = 4
    env._battery_level = 20
    env._step_count = 0

    observation, reward, terminated, truncated, info = env.step(1)

    assert observation["rover"] == env.size - 1
    assert observation["battery"] == 19
    assert info["step_count"] == 1
    assert reward == -1
    assert terminated is False
    assert truncated is False

def test_reaching_target_terminates_episode(env):
    env._rover_location = 3
    env._target_location = 4
    env._battery_level = 20
    env._step_count = 0

    observation, reward, terminated, truncated, info = env.step(1)

    assert observation["rover"] == 4
    assert observation["battery"] == 19
    assert info["step_count"] == 1
    assert reward == 100
    assert terminated is True
    assert truncated is False

def test_rover_battery_depleted_termination(env):
    env._rover_location = 4
    env._target_location = 8
    env._battery_level = 1
    env._step_count = 0

    observation, reward, terminated, truncated, info = env.step(0)

    assert observation["rover"] == 4
    assert observation["battery"] == 0
    assert info["step_count"] == 1
    assert reward == -100
    assert terminated is True
    assert truncated is False

def test_max_steps_reached_truncation(env):
    env._rover_location = 4
    env._target_location = 8
    env._battery_level = 20
    env._step_count = env.max_steps - 1

    observation, reward, terminated, truncated, info = env.step(1)

    assert observation["rover"] == 5
    assert observation["battery"] == 19
    assert info["step_count"] == env.max_steps
    assert reward == -1
    assert terminated is False
    assert truncated is True

def test_reaching_target_on_final_battery_unit_succeeds(env):
    env._rover_location = 7
    env._target_location = 8
    env._battery_level = 1
    env._step_count = 0

    observation, reward, terminated, truncated, info = env.step(1)

    assert observation["rover"] == 8
    assert observation["battery"] == 0
    assert info["step_count"] == 1
    assert reward == 100
    assert terminated is True
    assert truncated is False

def test_seeded_reset_reproducibility(env):
    obs1, info1 = env.reset(seed=42)
    obs2, info2 = env.reset(seed=42)

    assert obs1 == obs2
    assert info1 == info2
