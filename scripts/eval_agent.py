# scripts/eval_agent.py

import argparse
import torch
from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--episodes", type=int, default=5)
    args = parser.parse_args()

    env = SnakeEnv(grid_size=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create agent and load weights
    agent = DQNAgent(env.observation_space.shape, env.action_space.n)
    checkpoint = torch.load(args.model, map_location=device)
    agent.policy_net.load_state_dict(checkpoint)
    agent.policy_net.to(device)
    agent.policy_net.eval()

    for ep in range(1, args.episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                q_vals = agent.policy_net(state_tensor)
            action = q_vals.argmax().item()
            state, reward, done, _ = env.step(action)
            total_reward += reward
            env.render()
        print(f"Episode {ep}: total reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    main()
