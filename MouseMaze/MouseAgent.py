import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
import time

class Maze:
    def __init__(self, grid):
        self.grid = grid
        self.start = np.argwhere(self.grid == 2)[0]
        self.current_position = self.start
        self.goal = np.argwhere(self.grid == 3)[0]
        self.actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    def step(self, action):
        new_position = self.current_position + self.actions[action]
        if self.is_valid(new_position):
            self.current_position = new_position

        reward = -1
        done = False
        if np.array_equal(self.current_position, self.goal):
            reward = 100
            done = True

        return self.current_position, reward, done

    def is_valid(self, position):
        if position[0] < 0 or position[0] >= self.grid.shape[0]:
            return False
        if position[1] < 0 or position[1] >= self.grid.shape[1]:
            return False
        if self.grid[position[0], position[1]] == 1:
            return False
        return True

    def reset(self):
        self.current_position = self.start
        return self.current_position


class MouseAgent:
    def __init__(self, input_size, output_size, lr=0.001):
        self.q_net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def get_action(self, state, epsilon=0.1):
        if np.random.random() < epsilon:
            return np.random.randint(4)
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return torch.argmax(q_values).item()

    def update(self, state, action, next_state, reward, done, discount_factor=0.99):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
        q_values = self.q_net(state_tensor)
        next_q_values = self.q_net(next_state_tensor)

        target_q_values = q_values.clone().detach()
        target_q_values[0, action] = reward + discount_factor * torch.max(next_q_values) * (1 - done)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train(agent, env, episodes, maze_grid, epsilon_start=1, epsilon_end=0.1, epsilon_decay=0.995, save_path=None, visualize=True):
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            epsilon = max(epsilon_end, epsilon_start * epsilon_decay ** episode)
            action = agent.get_action(state, epsilon)
            next_state, reward, done = env.step(action)
            agent.update(state, action, next_state, reward, done)
            state = next_state
            total_reward += reward

        if episode % 10 == 0:
            print(f'Episode {episode}, total reward: {total_reward}')
            if visualize:
                visualize_maze(agent, maze_grid)

        if save_path is not None and episode % 100 == 0:
            torch.save(agent.q_net.state_dict(), save_path)

    if save_path is not None:
        torch.save(agent.q_net.state_dict(), save_path)

def visualize_maze(agent, maze_grid, fps=1):
    print(maze_grid)
    env = Maze(maze_grid)
    cell_size = 50
    width, height = env.grid.shape[1] * cell_size, env.grid.shape[0] * cell_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mouse Maze")
    clock = pygame.time.Clock()

    colors = {
        0: (255, 255, 255),  # empty cell
        1: (0, 0, 0),        # wall
        2: (0, 255, 0),      # start
        3: (255, 0, 0),      # goal
        4: (0, 0, 255)       # agent
    }

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(colors[0])

        for i in range(env.grid.shape[0]):
            for j in range(env.grid.shape[1]):
                color = colors[env.grid[i, j]]
                pygame.draw.rect(screen, color, (j * cell_size, i * cell_size, cell_size, cell_size), 0)

        pygame.draw.rect(screen, colors[4], (env.current_position[1] * cell_size, env.current_position[0] * cell_size, cell_size, cell_size), 0)

        # pygame.display.update()
        pygame.display.flip()

        # Wait for the appropriate amount of time before drawing the next frame
        clock.tick(fps)
        time.sleep(1/fps)

        print(env.current_position, env.goal)
        if np.array_equal(env.current_position, env.goal):
            break

        action = agent.get_action(env.current_position, 0)
        env.step(action)
        print("Running!")

    print("Done!")

    pygame.quit()

if __name__ == "__main__":
    # Set parameters
    train_flag = False
    agent_weights_save_path = 'agent_weights.pth' 

    # Example maze grid:
    # 0 = empty cell, 1 = wall, 2 = start, 3 = goal
    maze_grid = np.array([
        [2, 0, 0, 0, 1],
        [1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 0, 1],
        [0, 0, 0, 0, 3]
    ])
   
    maze_env = Maze(maze_grid.copy())
    mouse_agent = MouseAgent(input_size=2, output_size=4)

    # Train the agent
    if train_flag:
        train(mouse_agent, maze_env, episodes=10000, maze_grid=maze_grid, save_path=agent_weights_save_path, visualize=False)
    else:
        mouse_agent.q_net.load_state_dict(torch.load('agent_weights.pth'))

    # Test the trained agent in the graphical environment
    visualize_maze(mouse_agent, maze_grid.copy())