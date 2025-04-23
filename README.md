# RL Snake AI

A Jupyter Notebook comparing DQN, PPO, and A2C on the Snake game.

## What’s here

- **SnakeEnv**: Pygame-based environment with full/limited vision (`reset`, `step`, `render`).
- **Notebook**: `Snake_RL.ipynb` contains all imports, training loops, evaluation, and plots.

## Install

```bash
pip install pygame numpy torch matplotlib
```

## Experiments

Tested DQN, PPO, and A2C with a hyperparameter grid, in both full and limited vision, for 300 000 steps each.

## Results & Reproducibility

- **Rewards**: +10 food, –10 crash, –0.1 per step.
- **Seed**: Fixed at 42.
- **Saving**: Checkpoints & plots are generated inline—export as needed.

Run the notebook end-to-end to reproduce side-by-side learning curves for DQN, PPO, and A2C.
