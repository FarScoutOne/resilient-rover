# Resilient Rover

Resilient Rover is a reinforcement learning learning project built with Python and Gymnasium.

The project is intended to provide hands-on experience designing custom reinforcement learning environments, defining observation and action spaces, creating reward functions, training agents, and evaluating learned behavior.

The initial version will model a simplified rover tasked with reaching a goal while operating under basic resource constraints. Later versions may introduce additional challenges such as battery management, communication failures, delayed observations, and changing mission objectives.

## Project Goals

* Learn the fundamentals of reinforcement learning through implementation.
* Build a custom Gymnasium environment from scratch.
* Implement and evaluate a simple Q-learning agent.
* Compare learned policies against baseline behavior.
* Explore how environment and reward design affect agent behavior.
* Gradually introduce robustness challenges and evaluate how learned policies respond.

## Planned Development

The project will be developed incrementally.

### Version 0 — Basic Rover Environment

Initial functionality will include:

* rover position
* target position
* discrete movement actions
* battery or resource constraints
* episode termination and truncation conditions
* simple reward structure

### Future Extensions

Possible later additions include:

* charging behavior
* stochastic action failures
* communication loss
* delayed or incomplete observations
* changing objectives and mission interruptions
* structured evaluation scenarios
* comparison of multiple reinforcement learning approaches

## Technology

* Python
* Gymnasium
* NumPy
* Jupyter
* pytest
* Matplotlib

## Current Status

Project setup and environment design are in progress.

The initial focus is on defining and validating the Resilient Rover environment before implementing a learning agent.
