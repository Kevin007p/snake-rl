# scripts/train_dqn.py

import argparse
import os
import numpy as np
import torch
from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent
import matplotlib.pyplot as plt


def train(args):
    env = SnakeEnv(grid_size=args.grid_size)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    agent = DQNAgent(
        obs_shape,
        n_actions,
        lr=args.lr,
        gamma=args.gamma,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        target_update=args.target_update
    )

    best_avg = -float('inf')
    rewards_history = []

    epsilon = args.eps_start
    eps_decay = (args.eps_start - args.eps_end) / args.eps_decay

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        # decay epsilon
        if epsilon > args.eps_end:
            epsilon -= eps_decay

        # logging
        if ep % args.log_interval == 0:
            avg_reward = np.mean(rewards_history[-args.log_interval:])
            print(f"Episode {ep:4d} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.2f}")
            # save best
            if avg_reward > best_avg:
                best_avg = avg_reward
                os.makedirs(args.save_path, exist_ok=True)
                save_file = os.path.join(args.save_path, 'best_model.pt')
                torch.save(agent.policy_net.state_dict(), save_file)
                print(f"  → New best avg. Saving model to {save_file}")

    # final save
    final_path = os.path.join(args.save_path, 'final_model.pt')
    torch.save(agent.policy_net.state_dict(), final_path)
    print("Training complete. Final model saved to", final_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid-size',    type=int,   default=10)
    parser.add_argument('--episodes',     type=int,   default=1000)
    parser.add_argument('--batch-size',   type=int,   default=32)
    parser.add_argument('--lr',           type=float, default=1e-3)
    parser.add_argument('--gamma',        type=float, default=0.99)
    parser.add_argument('--buffer-size',  type=int,   default=10000)
    parser.add_argument('--target-update',type=int,   default=1000)
    parser.add_argument('--eps-start',    type=float, default=1.0)
    parser.add_argument('--eps-end',      type=float, default=0.05)
    parser.add_argument('--eps-decay',    type=int,   default=500)   # episodes over which to decay
    parser.add_argument('--log-interval', type=int,   default=50)
    parser.add_argument('--save-path',    type=str,   default='models')
    args = parser.parse_args()
    train(args)
