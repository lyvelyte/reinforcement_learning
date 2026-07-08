import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
import time
from collections import deque
from gen_maze import generate_random_maze

# ---------------------------------------------------------------------------
# Hyper-parameters (tweak here)
# ---------------------------------------------------------------------------
VIEW_SIZE = 7               # NxN local grid patch fed to the network
BUFFER_SIZE = 100_000       # experience replay capacity
BATCH_SIZE = 64             # mini-batch sampled each update
TARGET_UPDATE_FREQ = 100    # soft-sync target-net every N gradient steps
LEARNING_RATE = 5e-4        # Adam LR
GAMMA = 0.99                # discount factor
EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY_STEPS = 4000  # linear decay; match or exceed total training episodes
SHAPING_K = 0.4             # potential-based shaping coefficient (per step)
MAX_EPISODE_STEPS = 400     # safety cap per maze episode
TRAIN_EVERY_N_STEPS = 4     # only run gradient step every N environment steps
TRANSITIONS_PER_EPISODE_CAP = MAX_EPISODE_STEPS  # drop stale goal-specific transitions from older mazes


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size=BATCH_SIZE):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions).astype(np.int64),
            np.array(rewards).astype(np.float32),
            np.array(next_states),
            np.array(dones).astype(np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Maze environment - gym-style with local-view observation
# ---------------------------------------------------------------------------
class Maze:
    """Each reset generates a new random maze.

    Observation is a VIEW_SIZE x VIEW_SIZE float array centred on the agent;
    cell values are 0 (empty), 1 (wall), 2 (goal).
    """

    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
    ACTION_NAMES = [("Right", "\u2192"), ("Left", "\u2190"), ("Down", "\u2193"), ("Up", "\u2191")]

    def __init__(self, grid):
        self.grid = grid
        self.start = tuple(np.argwhere(self.grid == 2)[0])
        self.goal = tuple(np.argwhere(self.grid == 3)[0])
        self.current_position = self.start

    def _local_view(self, position, size=VIEW_SIZE):
        """Return a local patch centred on *position*, encoding walls/goal."""
        half = size // 2
        view = np.zeros((size, size), dtype=np.float32)
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                r, c = position[0] + di, position[1] + dj
                if 0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]:
                    cell = self.grid[r, c]
                    view[di + half, dj + half] = 2.0 if cell == 3 else float(cell == 1)
        return view

    def step(self, action):
        dr, dc = self.ACTIONS[action]
        nr, nc = self.current_position[0] + dr, self.current_position[1] + dc
        old_pos = self.current_position
        moved = False
        if self._is_valid((nr, nc)):
            self.current_position = (nr, nc)
            moved = True

        reached_goal = np.array_equal(self.current_position, self.goal)
        reward = -1.0
        if reached_goal:
            reward += 100.0

        # Potential-based shaping: guide towards goal w/o changing optimal policy
        old_dist = abs(old_pos[0] - self.goal[0]) + abs(old_pos[1] - self.goal[1])
        new_dist = abs(self.current_position[0] - self.goal[0]) + abs(
            self.current_position[1] - self.goal[1]
        )
        reward += SHAPING_K * (old_dist - new_dist)

        obs = self._local_view(self.current_position)
        return obs, reward, reached_goal, moved

    def reset(self):
        self.current_position = self.start
        return self._local_view(self.current_position)

    def _is_valid(self, position):
        r, c = position
        if r < 0 or r >= self.grid.shape[0]:
            return False
        if c < 0 or c >= self.grid.shape[1]:
            return False
        if self.grid[r, c] == 1:
            return False
        return True

    def manhattan_to_goal(self):
        return abs(self.current_position[0] - self.goal[0]) + abs(
            self.current_position[1] - self.goal[1]
        )


# ---------------------------------------------------------------------------
# CNN-based Q-network
# ---------------------------------------------------------------------------
class QNetwork(nn.Module):
    def __init__(self, input_size=VIEW_SIZE, output_size=4):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
        )
        flat = 32 * input_size * input_size
        self.head = nn.Sequential(
            nn.Linear(flat, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        # x: (B, V*V) -> (B, 1, V, V)
        x = x.view(-1, 1, VIEW_SIZE, VIEW_SIZE)
        x = self.conv(x)
        x = x.flatten(1)
        return self.head(x)


# ---------------------------------------------------------------------------
# DQN agent
# ---------------------------------------------------------------------------
class MouseAgent:
    def __init__(self):
        self.online_net = QNetwork()
        self.target_net = QNetwork()
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=LEARNING_RATE
        )
        self.buffer = ReplayBuffer()
        self.update_count = 0
        self._step_count = 0          # monotonically increasing step counter
        self._episode_transitions = 0 # transitions stored in current episode

    def new_episode(self):
        """Call at the start of each episode to reset per-episode counters."""
        self._episode_transitions = 0

    def get_action(self, state, epsilon=0.0):
        if np.random.random() < epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q = self.online_net(torch.FloatTensor(state).unsqueeze(0))
            return q.argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        if self._episode_transitions < TRANSITIONS_PER_EPISODE_CAP:
            self.buffer.push(state, action, reward, next_state, done)
            self._episode_transitions += 1

    def train_step(self):
        self._step_count += 1
        if self._step_count % TRAIN_EVERY_N_STEPS != 0:
            return None
        if len(self.buffer) < BATCH_SIZE:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample()

        s = torch.FloatTensor(states)
        a = torch.LongTensor(actions).unsqueeze(1)
        r = torch.FloatTensor(rewards)
        ns = torch.FloatTensor(next_states)
        d = torch.FloatTensor(dones)

        q_values = self.online_net(s)
        q_a = q_values.gather(1, a).squeeze(1)

        # Double DQN: online selects, target evaluates
        with torch.no_grad():
            next_q_online = self.online_net(ns).argmax(dim=1)
            next_q_target = self.target_net(ns).gather(
                1, next_q_online.unsqueeze(1)
            ).squeeze(1)
            target_q = r + GAMMA * next_q_target * (1.0 - d)

        loss = nn.MSELoss()(q_a, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self.update_count += 1

        if self.update_count % TARGET_UPDATE_FREQ == 0:
            tau = 0.05
            for p_tgt, p_online in zip(
                self.target_net.parameters(), self.online_net.parameters()
            ):
                p_tgt.data = tau * p_online.data + (1 - tau) * p_tgt.data

        return loss.item()

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        self.online_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.online_net.state_dict())


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self, window=100):
        self.rewards = deque(maxlen=window)
        self.losses = deque(maxlen=500)
        self.solve_lengths = deque(maxlen=window)
        self.solved = deque(maxlen=window)

    def record(self, total_reward, loss, steps, solved):
        self.rewards.append(total_reward)
        if loss is not None:
            self.losses.append(loss)
        self.solve_lengths.append(steps)
        self.solved.append(1 if solved else 0)

    @property
    def solve_rate(self):
        return sum(self.solved) / len(self.solved) if self.solved else 0.0


def _linear_epsilon(episode):
    progress = min(episode / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + (EPSILON_END - EPSILON_START) * progress


# ---------------------------------------------------------------------------
# Live dashboard pygame window
# ---------------------------------------------------------------------------
class Dashboard:
    def __init__(self, width=560, height=340):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Training Dashboard")
        self.running = True

    def draw(self, tracker, episode):
        if not self.running or self.screen is None:
            return
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False
                self.screen = None
                break
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                self.running = False
                self.screen = None

        W, H = self.screen.get_size()
        self.screen.fill((30, 30, 40))
        font_sm = pygame.font.SysFont("mono", 14)

        avg_r = sum(tracker.rewards) / len(tracker.rewards) if tracker.rewards else 0
        avg_len = (
            sum(tracker.solve_lengths) / len(tracker.solve_lengths)
            if tracker.solve_lengths
            else 0
        )
        lines = [
            f"Episode    {episode}",
            f"Solve rate  {tracker.solve_rate:.1%}",
            f"Avg reward   {avg_r:+.1f}   Avg steps  {avg_len:.0f}",
        ]
        y = 18
        for line in lines:
            self.screen.blit(
                font_sm.render(line, True, (200, 200, 210)), (12, y)
            )
            y += 20

        # Reward curve
        _draw_curve(
            self.screen, list(tracker.rewards),
            (60, 80, W // 2 - 40, H - 100),
            (64, 224, 208), "Reward"
        )
        # Loss curve
        _draw_curve(
            self.screen, list(tracker.losses),
            (W // 2 + 10, 80, W - W // 2 - 30, H - 100),
            (208, 64, 64), "Loss"
        )

        pygame.display.flip()

    def close(self):
        self.running = False
        if self.screen is not None:
            pygame.display.quit()


def _draw_curve(screen, data, rect, color, label):
    """Draw a simple line curve inside a rect (x, y, w, h)."""
    if not data:
        return
    x0, y0, w, h = rect
    font = pygame.font.SysFont("mono", 12)
    screen.blit(font.render(label, True, color), (x0 + 4, y0 - 2))

    # Border
    pygame.draw.rect(screen, (80, 80, 90), rect, 1)

    mn = min(data)
    mx = max(data)
    rng = mx - mn if mx != mn else 1.0

    pts = []
    for i, v in enumerate(data):
        px = x0 + int(i * (w - 4) / max(len(data) - 1, 1))
        py = y0 + h - 4 - int((v - mn) / rng * (h - 8))
        pts.append((px, py))

    if len(pts) > 1:
        pygame.draw.lines(screen, color, False, pts, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    agent, maze_size, episodes=2000, save_path=None,
    dashboard_flag=True, eval_every=50
):
    """Train *agent* on random mazes of size *maze_size*.

    If *dashboard_flag* is True, a pygame window shows live reward/loss curves.
    Every *eval_every* episodes the trained policy runs a fresh maze and the
    result is printed (solved or not, steps taken).
    """
    tracker = MetricsTracker()
    dashboard = None if not dashboard_flag else Dashboard()

    best_solve_rate = 0.0
    best_weights = None

    for ep in range(episodes):
        maze_grid = generate_random_maze(maze_size[0], maze_size[1])
        env = Maze(maze_grid.copy())
        agent.new_episode()
        state = env.reset()
        done = False
        total_reward = 0.0
        last_loss = None

        for step in range(MAX_EPISODE_STEPS):
            eps = _linear_epsilon(ep)
            action = agent.get_action(state, eps)
            next_state, reward, done, _moved = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            state = next_state
            total_reward += reward
            if done:
                break

        steps_taken = step + 1
        tracker.record(total_reward, last_loss, steps_taken, done)

        # Console progress
        if ep % eval_every == 0 or ep == episodes - 1:
            rate = f"{tracker.solve_rate:.0%}"
            avg = (
                sum(tracker.rewards) / len(tracker.rewards)
                if tracker.rewards else 0
            )
            print(
                f"Ep {ep:5d} | solve {rate} | "
                f"avg_reward {avg:+6.1f} | buf {len(agent.buffer):>6d}"
            )

        # Dashboard (throttled)
        if dashboard and ep % 3 == 0:
            dashboard.draw(tracker, ep)

        # Periodic eval / best-snapshot (check every episode, so we don't miss the peak)
        if len(tracker.solved) >= 20 and tracker.solve_rate > best_solve_rate:
            best_solve_rate = tracker.solve_rate
            best_weights = {k: v.clone() for k, v in agent.online_net.state_dict().items()}

    if best_weights is not None:
        agent.online_net.load_state_dict(best_weights)
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print(f"Loaded best weights (solve rate {best_solve_rate:.0%}).")

    if save_path:
        agent.save(save_path)
        print(f"Weights saved to {save_path}")

    if dashboard:
        dashboard.close()


# ---------------------------------------------------------------------------
# Icon helpers (mouse / cheese) - kept from original
# ---------------------------------------------------------------------------
def _draw_mouse_icon(screen, center_x, center_y, cell_size):
    """Draw a simple mouse icon using pygame primitives."""
    cx, cy = center_x + cell_size // 2, center_y + cell_size // 2
    body_r = cell_size // 4
    head_r = cell_size // 5
    pygame.draw.circle(screen, (160, 160, 160), (cx, cy + 2), body_r)
    pygame.draw.circle(screen, (160, 160, 160), (cx, cy - body_r + 2), head_r)
    ear_r = cell_size // 8
    pygame.draw.circle(
        screen, (220, 180, 180), (cx - head_r // 2, cy - body_r - ear_r // 2), ear_r
    )
    pygame.draw.circle(
        screen, (220, 180, 180), (cx + head_r // 2, cy - body_r - ear_r // 2), ear_r
    )
    eye_r = max(1, cell_size // 16)
    pygame.draw.circle(screen, (0, 0, 0), (cx - 3, cy - body_r), eye_r)
    pygame.draw.circle(screen, (0, 0, 0), (cx + 3, cy - body_r), eye_r)
    tail_points = [
        (cx + body_r, cy + 2),
        (cx + body_r + cell_size // 4, cy - cell_size // 6),
        (cx + body_r + cell_size // 3, cy - cell_size // 3),
    ]
    pygame.draw.lines(screen, (220, 180, 180), False, tail_points, 2)


def _draw_cheese_icon(screen, center_x, center_y, cell_size):
    """Draw a simple cheese wedge with holes."""
    cx, cy = center_x + cell_size // 2, center_y + cell_size // 2
    half = cell_size // 3
    points = [
        (cx, cy - half),
        (cx - half, cy + half // 2),
        (cx + half, cy + half // 2),
    ]
    pygame.draw.polygon(screen, (255, 200, 0), points)
    hole_r = cell_size // 10
    pygame.draw.circle(screen, (200, 150, 0), (cx - 4, cy - 2), hole_r)
    pygame.draw.circle(screen, (200, 150, 0), (cx + 4, cy + 2), hole_r)
    pygame.draw.circle(screen, (200, 150, 0), (cx, cy + 5), hole_r // 2)


# ---------------------------------------------------------------------------
# Inference visualization - rich mouse/cheese rendering with trail
# ---------------------------------------------------------------------------
def visualize_inference(agent, maze_grid, fps=15):
    env = Maze(maze_grid.copy())
    cell_size = 50
    width = env.grid.shape[1] * cell_size
    height = env.grid.shape[0] * cell_size
    pygame.init()
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Mouse Maze - Inference")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont(None, 28)
    start_text = font.render(
        "Start", True, (0, 0, 0),
    )
    text_rect = start_text.get_rect(
        center=(env.start[1] * cell_size + cell_size // 2, env.start[0] * cell_size + cell_size // 2)
    )

    trail = [env.start]

    running = True
    solved = False
    steps = 0
    last_action_label = "—"
    last_blocked = False

    state = env._local_view(env.current_position)
    with torch.no_grad():
        q_vals = agent.online_net(torch.FloatTensor(state).unsqueeze(0)).numpy()[0]
    action = int(np.argmax(q_vals))

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        screen.fill((255, 255, 255))

        # Walls
        for i in range(env.grid.shape[0]):
            for j in range(env.grid.shape[1]):
                if env.grid[i, j] == 1:
                    pygame.draw.rect(
                        screen, (0, 0, 0),
                        (j * cell_size, i * cell_size, cell_size, cell_size), 0,
                    )

        # Start marker
        pygame.draw.rect(
            screen, (0, 255, 0),
            (env.start[1] * cell_size, env.start[0] * cell_size, cell_size, cell_size), 0,
        )
        screen.blit(start_text, text_rect)

        # Cheese at goal
        _draw_cheese_icon(
            screen, env.goal[1] * cell_size, env.goal[0] * cell_size, cell_size
        )

        # Trail
        for r, c in trail[:-1]:
            pygame.draw.rect(
                screen, (80, 80, 220),
                (c * cell_size + 5, r * cell_size + 5,
                 cell_size - 10, cell_size - 10),
                1,
            )

        # Mouse icon
        _draw_mouse_icon(
            screen, env.current_position[1] * cell_size,
            env.current_position[0] * cell_size, cell_size,
        )

        # --- HUD: step count + last action ---
        hud_h = 28
        pygame.draw.rect(screen, (200, 200, 210), (0, height - hud_h, width, hud_h))
        hud_font = pygame.font.SysFont("mono", 16)

        action_name, arrow = Maze.ACTION_NAMES[action]
        blocked_tag = " (blocked)" if last_blocked else ""
        best_action = int(np.argmax(q_vals))
        best_name, best_arrow = Maze.ACTION_NAMES[best_action]

        hud_text = (
            f"Steps: {steps:>4}  |  Last: [{action_name} {arrow}]{blocked_tag}"
            f"  |  Q-max: [{best_name} {best_arrow}] ({q_vals[best_action]:.2f})"
        )
        hud_surf = hud_font.render(hud_text, True, (30, 30, 40))
        screen.blit(hud_surf, (8, height - hud_h + 2))

        pygame.display.flip()
        clock.tick(fps)
        time.sleep(1 / fps)

        if solved or steps >= MAX_EPISODE_STEPS:
            break

        trail.append(env.current_position)
        _, _, done, moved = env.step(action)
        last_blocked = not moved
        if done:
            solved = True
        steps += 1

        state = env._local_view(env.current_position)
        with torch.no_grad():
            q_vals = agent.online_net(torch.FloatTensor(state).unsqueeze(0)).numpy()[0]
        action = int(np.argmax(q_vals))

    label = "SOLVED!" if solved else "Gave up (timeout)"
    print(f"Steps: {steps}  --  {label}")
    pygame.quit()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    maze_size = (11, 11)
    agent_weights_path = "agent_weights.pth"

    # --- Training --------------------------------------------------------
    train_flag = False
    if train_flag:
        mouse_agent = MouseAgent()
        print("Training...")
        train(mouse_agent, maze_size, episodes=2000, save_path=agent_weights_path)

    # --- Inference -------------------------------------------------------
    mouse_agent = MouseAgent()
    mouse_agent.load(agent_weights_path)

    maze_grid = generate_random_maze(
        maze_size[0], maze_size[1], visualize_maze_flag=False
    )
    print("Inference on a fresh maze:")
    visualize_inference(mouse_agent, maze_grid.copy())