import gym, torch, numpy as np
from env.snake_env import SnakeEnv
from agents.dqn_agent import DQNAgent

def main():
    env = SnakeEnv(grid_size=10)
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, lr=1e-3, batch_size=32, ...)
    for episode in range(1, N_EP+1):
        state = env.reset()
        done = False
        while not done:
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        # log metrics, decay epsilon, save checkpoint
