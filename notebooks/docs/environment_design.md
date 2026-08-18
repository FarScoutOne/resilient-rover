# Resilient Rover — Environment Design

## 1. Purpose

Resilient Rover is a custom reinforcement learning environment built with Python and Gymnasium.

The initial goal is to create a simple, understandable environment for learning how reinforcement learning systems interact with state, observations, actions, rewards, episode termination, and evaluation.

The project will begin with a small deterministic rover-navigation problem and gradually introduce additional complexity such as resource constraints, stochastic failures, communication issues, delayed observations, and changing objectives.

The environment should remain simple enough at each stage that unexpected agent behavior can be inspected and understood.

---

## 2. Initial Learning Objectives

The first version of the environment should provide hands-on experience with:

* defining a Gymnasium environment
* defining observation and action spaces
* implementing `reset()`
* implementing `step()`
* maintaining internal environment state
* designing reward functions
* distinguishing episode termination from truncation
* validating environment behavior before training
* establishing baseline agent performance
* training and evaluating a simple Q-learning agent

The first version should prioritize clarity over realism.

---

# 3. Resilient Rover v0

## 3.1 Environment Objective

The rover's objective is to navigate from an initial position to a target position before exhausting its available battery.

A successful episode occurs when the rover reaches the target.

A failed episode occurs when the rover exhausts its battery before reaching the target.

An episode may also be truncated if the rover exceeds a configured maximum number of steps.

---

## 3.2 World Representation

Resilient Rover v0 will use a small two-dimensional grid.

Initial proposed size:

```text
5 × 5
```

Example:

```text
. . . . .
. . . T .
. . . . .
. R . . .
. . . . .
```

Where:

```text
R = Rover
T = Target
. = Empty cell
```

The rover and target will occupy valid positions within the grid.

The target should not initially occupy the same location as the rover.

The initial rover and target locations may be randomized during `reset()`.

---

# 4. Internal State

The environment will initially maintain the following internal state:

```text
rover position
target position
battery level
current step count
```

Possible internal representation:

```python
self._rover_location
self._target_location
self._battery
self._step_count
```

Additional internal state may be introduced in later versions.

---

# 5. Observation Space

The agent should initially be able to observe:

```text
rover position
target position
battery level
```

A possible Gymnasium observation structure is:

```python
{
    "rover": [x, y],
    "target": [x, y],
    "battery": battery_level
}
```

The exact Gymnasium spaces will be determined during implementation.

Possible representations include:

```text
rover position:
Box or MultiDiscrete

target position:
Box or MultiDiscrete

battery:
Discrete or Box
```

The initial environment will provide complete state information to the agent.

In other words:

```text
observation ≈ environment state
```

Partial observability will be introduced only in later versions.

---

# 6. Action Space

Resilient Rover v0 will use a discrete action space.

Initial actions:

```text
0 = move right
1 = move up
2 = move left
3 = move down
4 = wait
```

Possible Gymnasium declaration:

```python
self.action_space = gym.spaces.Discrete(5)
```

Movement actions will modify the rover's position by one grid cell.

The rover must remain inside the grid boundaries.

Attempts to move outside the grid should leave the rover in its current position.

The behavior of the `WAIT` action should initially leave the rover's position unchanged.

---

# 7. Battery Model

The rover will have a finite battery.

Initial proposed behavior:

```text
movement action → battery decreases by 1
WAIT action     → battery decreases by 1
```

This keeps the initial resource model simple.

Initial battery capacity should be large enough that successful episodes are possible from all valid starting positions.

A possible starting value is:

```text
battery = 20
```

The exact value should be treated as a tunable environment parameter.

Battery behavior can become more realistic in later versions.

---

# 8. State Transitions

For each call to:

```python
env.step(action)
```

the environment should:

1. Interpret the selected action.
2. Determine the proposed rover movement.
3. Prevent movement outside the grid.
4. Update the rover position.
5. Reduce the battery level.
6. Increment the step count.
7. Determine whether the target has been reached.
8. Determine whether the battery has been exhausted.
9. Calculate the reward.
10. Determine `terminated`.
11. Determine `truncated`.
12. Return the new observation and auxiliary information.

Conceptually:

```text
current state
     ↓
agent chooses action
     ↓
environment applies action
     ↓
environment updates state
     ↓
environment calculates reward
     ↓
environment checks termination/truncation
     ↓
new observation returned
```

---

# 9. Reward Function

The initial reward function should remain deliberately simple.

Proposed v0 reward structure:

```text
+100    reach target
-100    battery exhausted before reaching target
-1      every other timestep
```

The step penalty provides an incentive for the rover to reach the target efficiently rather than wander indefinitely.

This reward function should be treated as an initial hypothesis rather than a final design.

Unexpected behavior should be documented rather than immediately hidden by changing reward values.

---

# 10. Episode Termination

An episode is **terminated** when the environment reaches a natural terminal state.

For v0:

```text
terminated = True
```

when either:

```text
rover reaches target
```

or:

```text
battery is exhausted
```

Otherwise:

```text
terminated = False
```

---

# 11. Episode Truncation

An episode is **truncated** when it is stopped for an external constraint rather than because the environment reached a natural terminal state.

For example:

```text
maximum episode length = 100 steps
```

If the rover has not reached the target or exhausted its battery after the maximum number of steps:

```text
truncated = True
```

This prevents unexpectedly long episodes during testing and training.

---

# 12. Information Returned by the Environment

The `info` dictionary may initially provide debugging and evaluation information that is not necessary for the agent's policy.

Possible values:

```python
{
    "distance_to_target": ...,
    "battery_remaining": ...,
    "steps": ...
}
```

For distance, Manhattan distance may be useful:

```text
|x_rover - x_target| + |y_rover - y_target|
```

This information can later support evaluation and debugging.

---

# 13. Reset Behavior

Calling:

```python
env.reset()
```

should:

1. reset the step counter
2. restore the battery
3. select a rover starting position
4. select a different target position
5. return the initial observation
6. return the initial `info` dictionary

The environment should support deterministic testing through a random seed.

Example:

```python
obs, info = env.reset(seed=42)
```

Repeated resets using the same seed should produce reproducible behavior where appropriate.

---

# 14. Environment Validation

Before training an RL agent, the environment should be validated independently.

Manual tests should include:

* moving in all four directions
* movement at each grid boundary
* waiting
* reaching the target
* exhausting the battery
* episode truncation
* resetting the environment
* reproducibility with seeds

Gymnasium's environment checker should also be used.

Example:

```python
from gymnasium.utils.env_checker import check_env

check_env(env)
```

---

# 15. Automated Tests

The project should include automated unit tests for core environment behavior.

Initial planned tests:

```text
test_reset_returns_valid_observation
test_rover_and_target_do_not_start_together
test_move_right
test_move_left
test_move_up
test_move_down
test_rover_cannot_leave_grid
test_action_reduces_battery
test_wait_reduces_battery
test_goal_terminates_episode
test_battery_exhaustion_terminates_episode
test_goal_reward
test_failure_reward
test_step_penalty
test_episode_step_limit
test_observation_is_valid
test_seeded_reset_is_reproducible
```

The learning algorithm should not be introduced until the environment passes its basic tests.

---

# 16. Baseline Evaluation

Before training a Q-learning agent, a random-action agent should be evaluated.

The random agent will choose:

```python
action = env.action_space.sample()
```

over multiple episodes.

Initial metrics should include:

```text
success rate
failure rate
average episode reward
average episode length
average remaining battery on success
```

These results will provide a baseline against which trained agents can later be compared.

---

# 17. Q-Learning Phase

Once the environment is validated, a simple Q-learning agent will be implemented.

The first agent should include:

```text
Q-table
epsilon-greedy exploration
epsilon decay
learning rate
discount factor
Q-value updates
```

The purpose of this phase is to understand reinforcement learning mechanics directly rather than relying immediately on a prebuilt deep-RL framework.

The trained policy should be compared against the random-agent baseline.

---

# 18. Planned Future Versions

## v1 — Battery and Charging

Possible additions:

```text
charging station
RECHARGE action
variable energy costs
route planning around battery constraints
```

The rover may need to decide whether continuing toward the target or detouring to recharge offers the better expected outcome.

---

## v2 — Communication Failures

Possible additions:

```text
communications status
stochastic command failures
retry behavior
communication recovery actions
```

Example:

```text
95% probability movement command succeeds
5% probability movement command fails
```

This introduces stochastic state transitions.

---

## v3 — Imperfect Observations

Possible additions:

```text
delayed telemetry
missing telemetry
sensor uncertainty
stale position information
```

At this point:

```text
observation ≠ complete environment state
```

The rover will have to act without perfect knowledge of the world.

---

## v4 — Mission Interruptions

The rover's objective may change during an episode.

Example:

```text
Initial objective:
Reach waypoint A.

New instruction:
Abort current objective and return to waypoint B.
```

This phase will explore how learned policies behave when objectives change during execution.

---

## v5 — Evaluation Suite

A structured evaluation suite may introduce controlled scenarios such as:

```text
baseline operation
low starting battery
intermittent communication
delayed telemetry
sensor failure
mission change
```

Each trained policy can be evaluated against the same scenarios.

Possible metrics:

```text
task success rate
episode duration
energy use
unsafe or invalid actions
recovery success
```

---

# 19. Deep Reinforcement Learning

Deep reinforcement learning should not be introduced until the simpler Q-learning implementation is understood and the environment complexity justifies a larger state representation.

Possible future technologies include:

```text
Stable-Baselines3
PPO
PyTorch
neural-network policies
```

The purpose of this progression is to understand why more advanced techniques become necessary rather than treating them as black-box solutions.

---

# 20. Experiment Log

Unexpected behavior should be recorded in:

```text
docs/experiment_log.md
```

For each interesting result, record:

```text
Date:

Environment version:

Hypothesis:

Configuration:

Expected behavior:

Observed behavior:

Possible explanation:

Next experiment:
```

Particular attention should be paid to behavior caused by reward design.

Examples of questions to investigate:

* Did the rover discover an unintended strategy?
* Did a reward encourage behavior that looked irrational?
* Did the agent exploit an environment assumption?
* Did learning become unstable?
* Did epsilon decay too quickly?
* Did a change improve training performance but reduce robustness?

The goal is not simply to produce a successful policy.

The goal is to understand how environment design, reward structure, and learning behavior interact.

---

# 21. Design Principle

Resilient Rover should evolve incrementally.

Only one major source of complexity should be introduced at a time whenever practical.

The preferred development pattern is:

```text
design
  ↓
implement
  ↓
validate
  ↓
establish baseline
  ↓
train
  ↓
evaluate
  ↓
analyze unexpected behavior
  ↓
add one new complexity
```

This should make it easier to identify why agent behavior changes as the environment becomes more difficult.
