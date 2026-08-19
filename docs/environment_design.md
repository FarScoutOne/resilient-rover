# Resilient Rover — Environment Design

## 1. Purpose

Resilient Rover is a custom reinforcement learning environment built with Python and Gymnasium.

The initial goal is to create a simple, understandable environment for learning how reinforcement learning systems interact with state, observations, actions, rewards, episode termination, and evaluation.

The project will begin with a **one-dimensional deterministic rover-navigation problem** so that the full state space, Q-values, and learned policy can be inspected and understood directly. Later versions will introduce two-dimensional navigation, resource management, stochastic failures, communication issues, delayed observations, and changing objectives.

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
* implementing Q-learning
* inspecting how reward propagates backward through Q-values
* understanding epsilon-greedy exploration
* understanding learning rate and discount factor
* evaluating a learned policy against a baseline

The first version should prioritize **clarity and inspectability over realism**.

---

# 3. Resilient Rover v0 — One-Dimensional Learning Environment

## 3.1 Environment Objective

The rover's objective is to navigate from an initial location to a target location before exhausting its available battery.

A successful episode occurs when the rover reaches the target.

A failed episode occurs when the rover exhausts its battery before reaching the target.

An episode may also be truncated if the rover exceeds a configured maximum number of steps.

---

## 3.2 World Representation

Resilient Rover v0 will use a one-dimensional sequence of discrete locations.

Initial proposed world:

```text
0 -- 1 -- 2 -- 3 -- 4 -- 5 -- 6 -- 7 -- 8 -- 9
```

A possible episode might begin as:

```text
0 -- 1 -- R -- 3 -- 4 -- 5 -- 6 -- 7 -- T -- 9
          ↑                         ↑
        rover                     target
```

Where:

```text
R = Rover
T = Target
```

The rover and target must occupy valid locations within the world.

The target should not initially occupy the same location as the rover.

The initial rover and target locations may be randomized during `reset()`.

The small one-dimensional world is intentional. It allows the learned values for every state and action to be inspected directly before adding more complex spatial behavior.

---

# 4. Internal State

The environment will initially maintain:

```text
rover location
target location
battery level
current step count
```

Possible internal representation:

```python
self._rover_location
self._target_location
self._battery_level
self._step_count
```

Additional internal state may be introduced in later versions.

---

# 5. Observation Space

The agent should initially observe:

```text
rover location
target location
battery level
```

A possible observation structure is:

```python
{
    "rover": rover_location,
    "target": target_location,
    "battery": battery_level,
}
```

Because each location is a single integer rather than a coordinate pair, the observation space can remain simple.

Possible representations include:

```text
rover location:
Discrete(world_size)

target location:
Discrete(world_size)

battery:
Discrete(max_battery + 1)
```

The initial environment will provide complete state information to the agent.

In other words:

```text
observation ≈ environment state
```

Partial observability will be introduced only in later versions.

---

# 6. Action Space

Resilient Rover v0 will use three discrete actions:

```text
0 = wait
1 = move right
2 = move left
```

Possible Gymnasium declaration:

```python
self.action_space = gym.spaces.Discrete(3)
```

Movement actions change the rover's location by one position.

For example:

```text
WAIT  → no location change
RIGHT → location + 1
LEFT  → location - 1
```

The rover must remain inside the valid range:

```text
0 through world_size - 1
```

Attempts to move beyond either boundary should leave the rover in its current location.

---

# 7. Battery Model

The rover will have a finite battery level.

Initial proposed behavior:

```text
MOVE RIGHT → battery level decreases by 1
MOVE LEFT  → battery level decreases by 1
WAIT       → battery level decreases by 1
```

This keeps the initial resource model deliberately simple.

A possible initial battery capacity is:

```text
battery_level = 20
```

The value should be large enough that successful navigation is possible from all valid starting states.

Battery capacity should be treated as a configurable environment parameter.

---

# 8. State Transitions

For each call to:

```python
env.step(action)
```

the environment should:

1. Interpret the selected action.
2. Determine the proposed new rover location.
3. Prevent movement beyond the world boundaries.
4. Update the rover location.
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
agent chooses LEFT / RIGHT / WAIT
     ↓
environment applies action
     ↓
environment updates location and battery level
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

The step penalty encourages the rover to reach the target efficiently.

This reward structure is an initial hypothesis rather than a final design.

Unexpected behavior should be recorded and investigated rather than immediately hidden by changing the reward values.

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

An episode is **truncated** when it is stopped because of an external limit rather than because the rover naturally succeeded or failed.

For example:

```text
maximum episode length = 50 steps
```

If the rover has not reached the target or exhausted its battery after the maximum number of steps:

```text
truncated = True
```

This prevents unexpectedly long episodes during testing and training.

---

# 12. Information Returned by the Environment

The `info` dictionary may provide debugging and evaluation information that is not necessary for the agent's policy.

Possible values:

```python
{
    "distance_to_target": ...,
    "battery_level": ...,
    "steps": ...,
}
```

In the one-dimensional world, distance can simply be:

```text
abs(rover_location - target_location)
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
3. select a rover starting location
4. select a different target location
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

* moving left
* moving right
* waiting
* attempting to move left at location `0`
* attempting to move right at the maximum location
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

Initial planned tests:

```text
test_reset_returns_valid_observation
test_rover_and_target_do_not_start_together
test_move_right
test_move_left
test_rover_cannot_move_below_zero
test_rover_cannot_move_above_world_limit
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

# 16. Random-Agent Baseline

Before training a Q-learning agent, a random-action agent should be evaluated.

The agent will choose:

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
average remaining battery level on success
```

These results will provide a baseline against which trained agents can later be compared.

---

# 17. Q-Learning Phase

Once the environment is validated, a simple Q-learning agent will be implemented.

The agent should include:

```text
Q-table
epsilon-greedy exploration
epsilon decay
learning rate
discount factor
Q-value updates
```

The one-dimensional environment should make it possible to inspect the learned values directly.

For example:

```text
Location 0:
LEFT  = ...
RIGHT = ...
WAIT  = ...

Location 1:
LEFT  = ...
RIGHT = ...
WAIT  = ...
```

This should make it easier to observe how rewards propagate backward through earlier state-action pairs.

The trained policy should then be compared against the random-agent baseline.

---

# 18. Policy Inspection

After training, the learned policy should be inspected rather than judged solely by cumulative reward.

Questions should include:

* Does the rover generally move toward the target?
* Does it learn that actions at boundaries may be unproductive?
* Does `WAIT` acquire meaningful value?
* How does behavior change as the battery level decreases?
* Do states closer to the target receive higher Q-values?
* Can reward propagation through the Q-table be observed directly?
* Are there states the agent has rarely visited?

A simple policy report might look like:

```text
Rover   Target   Battery   Preferred Action
0       8        15        RIGHT
1       8        14        RIGHT
2       8        13        RIGHT
...
```

---

# 19. Resilient Rover v1 — Two-Dimensional Navigation

Once the one-dimensional environment and Q-learning agent are understood, the rover will move to a two-dimensional grid.

Possible initial grid:

```text
5 × 5
```

The observation will then include:

```text
rover [x, y]
target [x, y]
battery_level
```

The action space will expand to:

```text
UP
DOWN
LEFT
RIGHT
WAIT
```

This phase introduces spatial decision-making while retaining the same fundamental RL concepts learned in v0.

---

# 20. Resilient Rover v2 — Battery and Charging

Possible additions:

```text
charging station
RECHARGE action
variable energy costs
route planning around battery constraints
```

The rover may need to decide whether continuing toward the target or detouring to recharge offers the better expected outcome.

---

# 21. Resilient Rover v3 — Communication Failures

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

# 22. Resilient Rover v4 — Imperfect Observations

Possible additions:

```text
delayed telemetry
missing telemetry
sensor uncertainty
stale location information
```

At this point:

```text
observation ≠ complete environment state
```

The rover will have to act without perfect knowledge of the world.

---

# 23. Resilient Rover v5 — Mission Interruptions

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

# 24. Evaluation Suite

A structured evaluation suite may introduce controlled scenarios such as:

```text
baseline operation
low starting battery level
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
invalid actions
recovery success
```

---

# 25. Deep Reinforcement Learning

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

# 26. Experiment Log

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

# 27. Design Principle

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
inspect learned behavior
  ↓
analyze unexpected behavior
  ↓
add one new complexity
```

Resilient Rover v0 is intentionally simple. Its purpose is not to model a realistic rover, but to make reinforcement learning behavior visible and understandable before more realistic complexity is introduced.
