import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import pygame
import time
from collections import deque
from gen_maze import generate_random_maze

# ---------------------------------------------------------------------------
# Hyper-parameters (tweak here)
# ---------------------------------------------------------------------------
VIEW_SIZE = 7               # NxN local grid patch fed to the network
BUFFER_SIZE = 200_000       # experience replay capacity
BATCH_SIZE = 512            # mini-batch sampled each update (GPU loves big batches)
TARGET_UPDATE_FREQ = 200    # hard-sync target-net every N gradient steps
LEARNING_RATE = 3e-4        # Adam LR
GAMMA = 0.99                # discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 2000  # linear decay; reaches min epsilon by ep 2000
TEMP_START = 1.0            # initial softmax temperature for policy noise
TEMP_END = 0.05             # final softmax temperature (near-deterministic)
TEMP_DECAY_STEPS = 1500     # cooling schedule length
CYCLE_WINDOW = 10           # recent steps inspected for cycles
CYCLE_THRESHOLD = 3         # same-pos repetitions before forcing escape
SHAPING_K = 0.4             # potential-based shaping coefficient (per step)
EFFICIENCY_BONUS = 5.0      # extra reward if episode finishes under target
EFFICIENCY_TARGET_FLOOR = 60   # don't tighten the efficiency bar below this
EFFICIENCY_TARGET_FRAC = 0.6   # target = 60% of recent avg success length
MAX_EPISODE_STEPS = 400     # safety cap per maze episode
TRAIN_EVERY_N_STEPS = 1     # train every step (GPU can keep up)
TRANSITIONS_PER_EPISODE_CAP = MAX_EPISODE_STEPS  # drop stale goal-specific transitions from older mazes
NUM_EVAL_EPISODES = 20      # fresh greedy evals per checkpoint
EVAL_PERIOD = 50            # eval every this many training episodes
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# Experience replay buffer (pre-allocated numpy — faster than deque)
# ---------------------------------------------------------------------------
class ReplayBuffer:
    __slots__ = ("states", "actions", "rewards", "next_states", "dones", "capacity", "_pos", "_size")

    def __init__(self, capacity=BUFFER_SIZE):
        flat_size = VIEW_SIZE * VIEW_SIZE
        self.capacity = capacity
        self.states = np.empty((capacity, flat_size), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, flat_size), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0

    def push(self, state, action, reward, next_state, done):
        idx = self._pos % self.capacity
        self.states[idx] = state.ravel()
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state.ravel()
        self.dones[idx] = done
        self._pos += 1
        if self._size < self.capacity:
            self._size += 1

    def sample(self, batch_size=BATCH_SIZE):
        n = min(batch_size, self._size)
        idx = np.random.randint(0, self._size, size=n)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    def __len__(self):
        return self._size


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
        self.steps = 0
        self.efficiency_target = None  # set by training loop; None = no bonus
        self._compute_bfs_distances()

    def _compute_bfs_distances(self):
        """BFS from goal → shortest-path distance for every reachable cell."""
        from collections import deque as bfs_deque
        dist = np.full(self.grid.shape, -1, dtype=np.float32)
        gr, gc = self.goal
        dist[gr, gc] = 0
        queue = bfs_deque([(gr, gc)])
        while queue:
            r, c = queue.popleft()
            d = dist[r, c] + 1
            for dr, dc in self.ACTIONS:
                nr, nc = r + dr, c + dc
                if (0 <= nr < self.grid.shape[0]
                    and 0 <= nc < self.grid.shape[1]
                    and self.grid[nr, nc] != 1
                    and dist[nr, nc] < 0):
                    dist[nr, nc] = d
                    queue.append((nr, nc))
        self.bfs_distances = dist

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

        # Potential-based shaping using BFS shortest-path distance
        old_potential = self.bfs_distances[old_pos]
        new_potential = self.bfs_distances[self.current_position]
        reward += SHAPING_K * (old_potential - new_potential)

        # Stepping-efficiency bonus: extra reward for fast solves
        if reached_goal and self.efficiency_target is not None:
            if self.steps <= self.efficiency_target:
                reward += EFFICIENCY_BONUS

        self.steps += 1

        obs = self._local_view(self.current_position)
        return obs, reward, reached_goal, moved

    def reset(self):
        self.steps = 0
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
        self.to(DEVICE)

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

    def get_action(self, state, epsilon=0.0, temperature=None):
        if np.random.random() < epsilon:
            return random.randint(0, 3)
        with torch.no_grad():
            q = self.online_net(
                torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            )
            q_vals = q.cpu().numpy()[0]
        if temperature is not None:
            return _softmax_action(q_vals, temperature)
        return int(np.argmax(q_vals))

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

        s = torch.as_tensor(states, dtype=torch.float32, device=DEVICE)
        a = torch.as_tensor(actions, dtype=torch.long, device=DEVICE).unsqueeze(1)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=DEVICE)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=DEVICE)
        d = torch.as_tensor(dones, dtype=torch.float32, device=DEVICE)

        q_values = self.online_net(s)
        q_a = q_values.gather(1, a).squeeze(1)

        # Double DQN: online selects, target evaluates
        with torch.no_grad():
            next_q_online = self.online_net(ns).argmax(dim=1)
            next_q_target = self.target_net(ns).gather(
                1, next_q_online.unsqueeze(1)
            ).squeeze(1)
            target_q = r + GAMMA * next_q_target * (1.0 - d)

        loss = nn.SmoothL1Loss()(q_a, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        self.update_count += 1

        # Hard target sync (stable for DQN on discrete envs)
        if self.update_count % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

        return loss.item()

    def save(self, path):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path):
        state = torch.load(path, map_location=DEVICE)
        self.online_net.load_state_dict(state)
        self.target_net.load_state_dict(state)


# ---------------------------------------------------------------------------
# Metrics tracker
# ---------------------------------------------------------------------------
class MetricsTracker:
    def __init__(self, window=100):
        self.rewards = deque(maxlen=window)
        self.losses = deque(maxlen=500)
        self.solve_lengths = deque(maxlen=window)
        self.solved = deque(maxlen=window)           # 1 if reached goal, 0 otherwise
        self.solve_lengths_success = deque(maxlen=window)   # steps for successful runs only
        self.solve_rate_history = deque(maxlen=400)  # rolling solve rate snapshots
        self.avg_steps_to_solve_history = deque(maxlen=400)  # avg steps/success snapshots

    def record(self, total_reward, loss, steps, solved):
        self.rewards.append(total_reward)
        if loss is not None:
            self.losses.append(loss)
        self.solve_lengths.append(steps)
        self.solved.append(1 if solved else 0)
        if solved:
            self.solve_lengths_success.append(steps)

    def snapshot(self):
        """Record current rolling solve-rate and avg-steps for plotting."""
        self.solve_rate_history.append(self.solve_rate)
        self.avg_steps_to_solve_history.append(self.avg_steps_to_solve)

    @property
    def solve_rate(self):
        return sum(self.solved) / len(self.solved) if self.solved else 0.0

    @property
    def avg_steps_to_solve(self):
        """Average length of successful episodes (goal reached) in the window."""
        return (
            sum(self.solve_lengths_success) / len(self.solve_lengths_success)
            if self.solve_lengths_success
            else 0
        )


def _linear_epsilon(episode):
    progress = min(episode / EPSILON_DECAY_STEPS, 1.0)
    return EPSILON_START + (EPSILON_END - EPSILON_START) * progress


def _linear_temperature(episode):
    progress = min(episode / TEMP_DECAY_STEPS, 1.0)
    return TEMP_START + (TEMP_END - TEMP_START) * progress


def _softmax_action(q_vals, temperature=1.0):
    """Sample an action by softmax over Q-values scaled by *temperature*.

    High temperature ≈ uniform random; low temperature ≈ argmax.
    """
    logits = np.asarray(q_vals, dtype=np.float64) / max(temperature, 1e-4)
    probs = np.exp(logits - np.max(logits))
    probs /= probs.sum()
    return int(np.random.choice(len(q_vals), p=probs))


def _try_escape_cycle(env, history):
    """Return an action index to break out of a cycle, or None.

    Checks the last CYCLE_WINDOW positions; if current position appears
    >= CYCLE_THRESHOLD times, pick a random valid neighbor not seen recently.
    """
    pos = env.current_position
    recent = history[-CYCLE_WINDOW:]
    from collections import Counter
    counts = Counter(recent)
    if counts.get(pos, 0) < CYCLE_THRESHOLD:
        return None
    visited = set(recent)
    for candidate in random.sample(range(4), 4):
        dr, dc = Maze.ACTIONS[candidate]
        nxt = (pos[0] + dr, pos[1] + dc)
        if env._is_valid(nxt) and nxt not in visited:
            return candidate
    return None


def _eval_greedy(agent, maze_size, n_episodes=NUM_EVAL_EPISODES):
    """Run epsilon=0 greedy eval on fresh mazes; returns solve fraction."""
    wins = 0
    for _ in range(n_episodes):
        env = Maze(generate_random_maze(maze_size[0], maze_size[1]).copy())
        state = env.reset()
        done = False
        for _ in range(MAX_EPISODE_STEPS):
            action = agent.get_action(state, epsilon=0.0)  # pure greedy
            state, reward, done, _ = env.step(action)
            if done:
                wins += 1
                break
    return wins / n_episodes


# ---------------------------------------------------------------------------
# Live dashboard pygame window
# ---------------------------------------------------------------------------
class Dashboard:
    def __init__(self, width=720, height=480):
        self.running = True
        self.disabled = False
        self.screen = None
        # Headless check — pygame display will silently fail without a
        # running X11/Wayland server, which is common on GPU servers.
        import os
        display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if not display:
            print("[dashboard] No DISPLAY/WAYLAND_DISPLAY — headless mode, disabling GUI.")
            self.disabled = True
            return
        pygame.init()
        try:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("Training Dashboard")
            self.disabled = False
        except pygame.error:
            print("[dashboard] pygame.display.set_mode failed — falling back to console only.")
            self.disabled = True
            self.screen = None

    def draw(self, tracker, episode):
        if self.disabled or not self.running or self.screen is None:
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
        solve_pct = tracker.solve_rate
        avg_to_solve = tracker.avg_steps_to_solve

        lines = [
            f"Episode      {episode}",
            f"Solve rate    {solve_pct:.1%}",
            f"Avg steps/success  {avg_to_solve:.0f}" if avg_to_solve else "Avg steps/success  \u2014",
            f"Avg reward    {avg_r:+.1f}     Avg len   {avg_len:.0f}",
        ]
        y = 18
        for line in lines:
            surf = font_sm.render(line, True, (200, 200, 210))
            self.screen.blit(surf, (12, y))
            y += 22

        # Solve-rate highlight bar
        bar_x, bar_y, bar_w, bar_h = 12, y + 6, W - 24, 14
        bar_color = (50, 205, 50) if solve_pct > 0.5 else (208, 168, 64) if solve_pct > 0.2 else (208, 64, 64)
        pygame.draw.rect(self.screen, (60, 60, 70), (bar_x, bar_y, bar_w, bar_h), border_radius=3)
        fill_w = int(bar_w * solve_pct)
        pygame.draw.rect(self.screen, bar_color, (bar_x, bar_y, fill_w, bar_h), border_radius=3)

        # --- Three curves in a row: Reward | Loss | Solve Rate ---
        curve_top = y + 34
        plot_margin = 58   # left margin per for y-axis labels + ticks
        gap = 14           # horizontal gap between plots
        avail = W - plot_margin * 3 - gap * 2 - 12  # subtract left margins, gaps, right pad
        plot_w = avail // 3
        plot_h = H - curve_top - 50  # leave room below for x-axis labels

        ox = plot_margin
        rects = []
        for _ in range(3):
            rects.append((ox, curve_top, plot_w, plot_h))
            ox += plot_w + gap

        _draw_curve_with_axes(
            self.screen, list(tracker.rewards),
            rects[0], (64, 224, 208), "Reward", "Reward"
        )
        # Loss values are non-negative; high initial loss is normal for DQN —
        # the target network starts with random weights, so early targets are garbage.
        _draw_curve_with_axes(
            self.screen, list(tracker.losses),
            rects[1], (208, 64, 64), "Loss", "Loss"
        )
        _draw_bounded_curve(
            self.screen, list(tracker.solve_rate_history),
            rects[2], (50, 205, 50), "Solve %",
        )

        pygame.display.flip()

    def close(self):
        self.running = False
        if self.screen is not None:
            self.screen = None


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


def _draw_y_axes(screen, rect, y_min, y_max, ylabel):
    """Draw y-axis ticks with labels and a rotated ylabel."""
    x0, y0, w, h = rect
    font = pygame.font.SysFont("mono", 10)
    mid_val = (y_min + y_max) / 2.0

    for val in [y_min, mid_val, y_max]:
        py = y0 + h - 4 - int((val - y_min) / max(y_max - y_min, 1e-9) * (h - 8))
        # Tick line
        pygame.draw.line(screen, (120, 120, 130), (x0 - 2, py), (x0 + 4, py))
        # Label right-aligned left of the plot
        txt = font.render(f"{val:.1f}", True, (160, 160, 170))
        screen.blit(txt, (x0 - 36 - txt.get_width(), py - 5))

    # Y label (rotated)
    yfont = pygame.font.SysFont("mono", 11)
    rotated = yfont.render(ylabel, True, (170, 170, 180))
    screen.blit(rotated, (x0 - 48 - rotated.get_width(), y0 + h // 2 - 6))


def _draw_x_axes(screen, rect, x_label):
    """Draw x-axis ticks and label centred below the plot."""
    x0, y0, w, h = rect
    font = pygame.font.SysFont("mono", 10)

    # Short horizontal ticks at bottom
    for ratio in (0.0, 0.5, 1.0):
        px = x0 + int(ratio * (w - 4))
        pygame.draw.line(screen, (120, 120, 130), (px, y0 + h - 4), (px, y0 + h + 2))

    # X label centred below
    surf = font.render(x_label, True, (160, 160, 170))
    screen.blit(surf, (x0 + w // 2 - surf.get_width() // 2, y0 + h + 4))


def _draw_bounded_y_axes(screen, rect):
    """Draw y-axis ticks for a [0, 1]-bounded value: 0%, 50%, 100%."""
    x0, y0, w, h = rect
    font = pygame.font.SysFont("mono", 10)
    labels = [(0.0, "0%"), (0.5, "50%"), (1.0, "100%")]
    for val, txt in labels:
        py = y0 + h - 4 - int((val - 0.0) / 1.0 * (h - 8))
        pygame.draw.line(screen, (120, 120, 130), (x0 - 2, py), (x0 + 4, py))
        surf = font.render(txt, True, (160, 160, 170))
        screen.blit(surf, (x0 - 36 - surf.get_width(), py - 5))
    # Y label
    yfont = pygame.font.SysFont("mono", 11)
    rotated = yfont.render("Solve", True, (170, 170, 180))
    screen.blit(rotated, (x0 - 48 - rotated.get_width(), y0 + h // 2 - 6))


def _draw_curve_with_axes(screen, data, rect, color, label, y_label):
    """Draw a line curve inside *rect*, complete with axis labels and ticks.

    Replaces the older split of _draw_curve + _draw_y_axes + _draw_x_axes so
    each plot is self-contained and laid out consistently.
    """
    if not data:
        return
    x0, y0, w, h = rect
    font_title = pygame.font.SysFont("mono", 12)
    screen.blit(font_title.render(label, True, color), (x0 + 4, y0 - 2))

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

    # Axes
    _draw_y_axes(screen, rect, mn, mx, y_label)
    _draw_x_axes(screen, rect, "Episode")


def _draw_bounded_curve(screen, data, rect, color, label):
    """Draw a curve clamped to [0, 1] y-range with 0%/50%/100% ticks.

    Used for the solve-rate plot so the y-axis is always in the same frame
    of reference regardless of actual values.
    """
    if not data:
        return
    x0, y0, w, h = rect
    font_title = pygame.font.SysFont("mono", 12)
    screen.blit(font_title.render(label, True, color), (x0 + 4, y0 - 2))

    # Border
    pygame.draw.rect(screen, (80, 80, 90), rect, 1)

    pts = []
    for i, v in enumerate(data):
        px = x0 + int(i * (w - 4) / max(len(data) - 1, 1))
        # Clamp to [0, 1] even if data goes outside
        clamped = max(0.0, min(1.0, v))
        py = y0 + h - 4 - int(clamped * (h - 8))
        pts.append((px, py))

    if len(pts) > 1:
        pygame.draw.lines(screen, color, False, pts, 1)

    _draw_bounded_y_axes(screen, rect)
    _draw_x_axes(screen, rect, "Episode")
    


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(
    agent, maze_size, episodes=5000, save_path=None,
    dashboard_flag=True, eval_every=50
):
    """Train *agent* on random mazes of size *maze_size*.

    Every EVAL_PERIOD episodes runs greedy (epsilon=0) evals and saves best
    weights based on actual inference-quality performance.
    """
    print(f"[train] device = {DEVICE}  |  CUDA available = {torch.cuda.is_available()}")
    tracker = MetricsTracker()
    dashboard = None if not dashboard_flag else Dashboard()

    best_eval_rate = -1.0
    best_weights = None

    for ep in range(episodes):
        maze_grid = generate_random_maze(maze_size[0], maze_size[1])
        env = Maze(maze_grid.copy())

        # Decay efficiency target based on recent success lengths
        if len(tracker.solve_lengths_success) >= 10:
            avg_succ = tracker.avg_steps_to_solve
            env.efficiency_target = max(
                int(avg_succ * EFFICIENCY_TARGET_FRAC), EFFICIENCY_TARGET_FLOOR
            )

        agent.new_episode()
        state = env.reset()
        done = False
        total_reward = 0.0
        last_loss = None
        position_history = [env.current_position]

        for step in range(MAX_EPISODE_STEPS):
            eps = _linear_epsilon(ep)
            temperature = _linear_temperature(ep)
            action = agent.get_action(state, eps, temperature=temperature)

            # Cycle detection — override with escape if stuck
            escape = _try_escape_cycle(env, position_history)
            if escape is not None:
                action = escape

            next_state, reward, done, _moved = env.step(action)
            position_history.append(env.current_position)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step()
            if loss is not None:
                last_loss = loss

            state = next_state
            total_reward += reward

            # Keep WSLg / X11 display alive when GUI dashboard is active.
            # pygame must pump events regularly or the compositor suppresses the window.
            if dashboard and step % 50 == 0:
                pygame.event.pump()

            if done:
                break

        steps_taken = step + 1
        tracker.record(total_reward, last_loss, steps_taken, done)
        tracker.snapshot()

        # Periodic greedy eval (epsilon=0 — the real inference metric)
        eval_rate = -1.0
        if ep % EVAL_PERIOD == 0 or ep == episodes - 1:
            eval_rate = _eval_greedy(agent, maze_size)

        # Save best checkpoint based on GREEDY performance (matches inference)
        if eval_rate >= 0 and eval_rate > best_eval_rate:
            best_eval_rate = eval_rate
            best_weights = {k: v.clone() for k, v in agent.online_net.state_dict().items()}

        # Console progress
        if ep % eval_every == 0 or ep == episodes - 1:
            online_rate = tracker.solve_rate
            avg = (
                sum(tracker.rewards) / len(tracker.rewards)
                if tracker.rewards else 0
            )
            avg_succ = tracker.avg_steps_to_solve
            steps_str = f"{avg_succ:.0f}" if avg_succ > 0 else "\u2014"
            target_str = f"{env.efficiency_target}" if env.efficiency_target is not None else "\u2014"
            line = (
                f"Ep {ep:5d} "
                f"| solve_rate {online_rate:.0%} "
                f"| avg_reward {avg:+7.1f} "
                f"| avg_steps+ {steps_str} "
                f"| eff_target {target_str}"
            )
            if eval_rate >= 0:
                line += f" | greedy_eval {eval_rate:.0%}"
            if best_eval_rate >= 0:
                line += f" | best_greedy {best_eval_rate:.0%}"
            line += f" | buf {len(agent.buffer):>6d}"
            print(line)

        # Dashboard (optional, off by default for speed)
        if dashboard and ep % 3 == 0:
            dashboard.draw(tracker, ep)

    # Restore best greedy-performance weights & save
    if best_weights is not None:
        agent.online_net.load_state_dict(best_weights)
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        print(f"Loaded best weights (greedy eval {best_eval_rate:.0%}).")

    # Final summary
    final_avg = (
        sum(tracker.rewards) / len(tracker.rewards) if tracker.rewards else 0
    )
    avg_succ_final = tracker.avg_steps_to_solve
    steps_final_str = f"{avg_succ_final:.0f}" if avg_succ_final > 0 else "\u2014"
    print(
        f"[train] DONE \u2014 "
        f"final solve_rate {tracker.solve_rate:.1%} | "
        f"avg_reward {final_avg:+.1f} | "
        f"best greedy eval {max(best_eval_rate, 0):.1%} | "
        f"avg steps (success) {steps_final_str}"
    )

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

    # Centre the window on screen (must be set *before* pygame.init).
    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.init()
    info = pygame.display.Info()
    rows, cols = env.grid.shape[0], env.grid.shape[1]

    hud_h = 36                                                    # status bar at bottom
    window_h = info.current_h // 2                                 # half display height
    raw_cell_size = (window_h - hud_h) // max(rows, cols)         # fill vertical space
    cell_size = max(4, min(raw_cell_size, 80))                    # keep it readable
    # Safety net: WSLg may report a huge virtual desktop; cap the window so it
    # never overflows an actual physical monitor.
    SAFE_MAX_H = 860
    while rows * cell_size + hud_h > SAFE_MAX_H and cell_size > 4:
        cell_size -= 1
    # Tight fit around content (maze grid + HUD bar).
    window_w = cols * cell_size
    window_h_safe = rows * cell_size + hud_h

    screen = pygame.display.set_mode((window_w, window_h_safe))
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
        q_vals = agent.online_net(
            torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        ).cpu().numpy()[0]
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
        inset = max(2, cell_size // 5)
        for r, c in trail[:-1]:
            pygame.draw.rect(
                screen, (80, 80, 220),
                (c * cell_size + inset, r * cell_size + inset,
                 cell_size - 2 * inset, cell_size - 2 * inset),
                1,
            )

        # Mouse icon
        _draw_mouse_icon(
            screen, env.current_position[1] * cell_size,
            env.current_position[0] * cell_size, cell_size,
        )

        # --- HUD: step count + last action ---
        hud_y = rows * cell_size
        pygame.draw.rect(screen, (200, 200, 210), (0, hud_y, window_w, hud_h))
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
        screen.blit(hud_surf, (8, hud_y + 2))

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
            q_vals = agent.online_net(
                torch.as_tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            ).cpu().numpy()[0]
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
    train_flag = True
    if train_flag:
        mouse_agent = MouseAgent()
        print("Training...")
        train(mouse_agent, maze_size, episodes=50000, save_path=agent_weights_path)

    # --- Inference -------------------------------------------------------
    mouse_agent = MouseAgent()
    mouse_agent.load(agent_weights_path)

    maze_grid = generate_random_maze(
        maze_size[0], maze_size[1], visualize_maze_flag=False
    )
    print("Inference on a fresh maze:")
    visualize_inference(mouse_agent, maze_grid.copy())