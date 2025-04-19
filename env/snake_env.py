# env/snake_env.py

import gym
from gym import spaces
import numpy as np
import random
import pygame


class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, grid_size=10, cell_size=20):
        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.screen_size = grid_size * cell_size

        # Pygame setup
        pygame.init()
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Snake RL")
        self.clock = pygame.time.Clock()

        # Gym spaces
        self.action_space = spaces.Discrete(4)  # 0=up,1=right,2=down,3=left
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(grid_size, grid_size), dtype=np.int8
        )

        # initialize game state
        self.reset()

    def reset(self):
        mid = self.grid_size // 2
        self.snake = [(mid, mid)]
        self.direction = (0, 1)
        self._place_food()
        self.done = False
        return self._get_obs()

    def step(self, action):
        dirs = {
            0: (-1, 0),
            1: (0, 1),
            2: (1, 0),
            3: (0, -1),
        }
        new_dir = dirs[action]
        # prevent immediate reverse
        if (new_dir[0] == -self.direction[0] and new_dir[1] == -self.direction[1]):
            new_dir = self.direction
        self.direction = new_dir

        head = self.snake[0]
        new_head = (head[0] + new_dir[0], head[1] + new_dir[1])
        reward = 0

        # check crash
        if (
            not 0 <= new_head[0] < self.grid_size
            or not 0 <= new_head[1] < self.grid_size
            or new_head in self.snake
        ):
            self.done = True
            return self._get_obs(), -10, True, {}

        # move snake
        self.snake.insert(0, new_head)
        if new_head == self.food_pos:
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        return self._get_obs(), reward, self.done, {}

    def _place_food(self):
        empty = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in self.snake
        ]
        self.food_pos = random.choice(empty)

    def _get_obs(self):
        obs = np.zeros((self.grid_size, self.grid_size), dtype=np.int8)
        for x, y in self.snake:
            obs[x, y] = 1
        obs[self.food_pos] = 2
        return obs

    def render(self, mode="human"):
        # clear
        self.screen.fill((0, 0, 0))
        # draw food
        fx, fy = self.food_pos
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            pygame.Rect(fy * self.cell_size, fx * self.cell_size, self.cell_size, self.cell_size),
        )
        # draw snake
        for x, y in self.snake:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                pygame.Rect(y * self.cell_size, x * self.cell_size, self.cell_size, self.cell_size),
            )
        # update display and cap framerate
        pygame.display.flip()
        self.clock.tick(10)  # FPS

    def close(self):
        pygame.quit()
