from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import subprocess
import time
import uuid
from collections import deque
from dataclasses import dataclass, fields, replace
from datetime import datetime, timezone
from typing import Callable, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gen_maze import generate_random_maze


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
VIEW_SIZE = 7
DEFAULT_MAZE_SIZE = (11, 11)
OBSERVATION_MODES = ("full", "local")
ALGORITHMS = ("dqn", "ppo")
NETWORK_TYPES = ("spatial", "flat")
DISTANCE_SHAPING_MODES = ("potential", "fractional", "none")
DEFAULT_EPISODES = 50_000
DEFAULT_SEED = 0
DEFAULT_OBSERVATION_MODE = "full"
DEFAULT_ALGORITHM = "dqn"
DEFAULT_NETWORK_TYPE = "spatial"
DEFAULT_RESUME = False

BUFFER_SIZE = 200_000
BATCH_SIZE = 512
MIN_REPLAY_SIZE = 2_000
TARGET_UPDATE_FREQ = 2_000
LEARNING_RATE = 3e-4
GAMMA = 0.99
N_STEP_RETURNS = 3
EPSILON_START = 1.0
EPSILON_MID = 0.10
EPSILON_END = 0.02
EPSILON_DECAY_EPISODES = 5_000
EPSILON_DECAY_STEPS = 500_000
EPSILON_FINAL_STEPS = 1_500_000
DEFAULT_NUM_ENVS = 64
DEFAULT_TRAIN_UPDATES_PER_STEP = 4
DEFAULT_UPDATES_PER_TRANSITION = 0.125
DEFAULT_WARMUP_STEPS = 10_000
DEFAULT_TARGET_TAU = 0.005
DEFAULT_PRIORITIZED_REPLAY = True
DEFAULT_PRIORITY_ALPHA = 0.6
DEFAULT_PRIORITY_BETA_START = 0.4
DEFAULT_PRIORITY_BETA_STEPS = 1_500_000
MAX_EPISODE_STEPS = 200
DEFAULT_TIMEOUT_STEP_FACTOR = 4.0
DEFAULT_MIN_EPISODE_STEPS = 20
NUM_EVAL_EPISODES = 200
EVAL_PERIOD = 100
EVAL_PERIOD_STEPS = 10_000
EVAL_SEED_OFFSET = 1_000_003
DEFAULT_DASHBOARD_FLAG = True
DEFAULT_DASHBOARD_EVERY = EVAL_PERIOD
DEFAULT_SAVE_PATH = "agent_weights.pth"
DEFAULT_TRAINING_LOG_PATH = "training_log.jsonl"
DEFAULT_DEVICE = "auto"
DEFAULT_REQUIRE_CUDA = True
DEFAULT_TRAIN_FLAG = True
DEFAULT_INFER_FLAG = True

STEP_PENALTY = -0.05
INVALID_MOVE_PENALTY = -0.20
GOAL_REWARD = 10.0
TIMEOUT_PENALTY = -2.0
DISTANCE_SHAPING_SCALE = 1.0
DEFAULT_DISTANCE_SHAPING_MODE = "potential"

DEFAULT_CURRICULUM_ENABLED = True
DEFAULT_CURRICULUM_EASY_EPISODES = 5_000
DEFAULT_CURRICULUM_MEDIUM_EPISODES = 15_000
DEFAULT_CURRICULUM_EASY_RANGE = (6, 10)
DEFAULT_CURRICULUM_MEDIUM_RANGE = (8, 16)
DEFAULT_CURRICULUM_MAX_RETRIES = 200
DEFAULT_CURRICULUM_PROMOTION_RATE = 0.90
DEFAULT_CURRICULUM_PROMOTION_EVALS = 3
DEFAULT_CURRICULUM_PREVIOUS_FRACTION = 0.20
DEFAULT_CURRICULUM_UNIFORM_FRACTION = 0.10
DEFAULT_VECTORIZED_ENVS = True
DEFAULT_CHECKPOINT_REPLAY = False
DEFAULT_EXPERT_PRETRAIN_MAZES = 0
DEFAULT_EXPERT_PRETRAIN_EPOCHS = 3
DEFAULT_PLANNER_FALLBACK = False

PPO_ROLLOUT_STEPS = 128
PPO_EPOCHS = 4
PPO_CLIP_RANGE = 0.2
PPO_GAE_LAMBDA = 0.95
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 0.5

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class TrainConfig:
    """Training and evaluation settings for MouseMaze."""

    maze_size: tuple[int, int] = DEFAULT_MAZE_SIZE
    episodes: int = DEFAULT_EPISODES
    seed: int = DEFAULT_SEED
    eval_seed: int | None = None
    algorithm: str = DEFAULT_ALGORITHM
    network_type: str = DEFAULT_NETWORK_TYPE
    resume: bool = DEFAULT_RESUME
    observation_mode: str = DEFAULT_OBSERVATION_MODE
    view_size: int = VIEW_SIZE
    max_episode_steps: int = MAX_EPISODE_STEPS
    timeout_step_factor: float | None = DEFAULT_TIMEOUT_STEP_FACTOR
    min_episode_steps: int = DEFAULT_MIN_EPISODE_STEPS
    step_penalty: float = STEP_PENALTY
    invalid_move_penalty: float = INVALID_MOVE_PENALTY
    goal_reward: float = GOAL_REWARD
    timeout_penalty: float = TIMEOUT_PENALTY
    distance_shaping_scale: float = DISTANCE_SHAPING_SCALE
    distance_shaping_mode: str = DEFAULT_DISTANCE_SHAPING_MODE
    curriculum_enabled: bool = DEFAULT_CURRICULUM_ENABLED
    curriculum_easy_episodes: int = DEFAULT_CURRICULUM_EASY_EPISODES
    curriculum_medium_episodes: int = DEFAULT_CURRICULUM_MEDIUM_EPISODES
    curriculum_easy_range: tuple[int, int] = DEFAULT_CURRICULUM_EASY_RANGE
    curriculum_medium_range: tuple[int, int] = DEFAULT_CURRICULUM_MEDIUM_RANGE
    curriculum_max_retries: int = DEFAULT_CURRICULUM_MAX_RETRIES
    buffer_size: int = BUFFER_SIZE
    batch_size: int = BATCH_SIZE
    min_replay_size: int = MIN_REPLAY_SIZE
    target_update_freq: int = TARGET_UPDATE_FREQ
    learning_rate: float = LEARNING_RATE
    gamma: float = GAMMA
    n_step_returns: int = N_STEP_RETURNS
    epsilon_start: float = EPSILON_START
    epsilon_mid: float = EPSILON_MID
    epsilon_end: float = EPSILON_END
    epsilon_decay_episodes: int = EPSILON_DECAY_EPISODES
    epsilon_decay_steps: int = EPSILON_DECAY_STEPS
    epsilon_final_steps: int = EPSILON_FINAL_STEPS
    num_envs: int = DEFAULT_NUM_ENVS
    train_updates_per_step: int = DEFAULT_TRAIN_UPDATES_PER_STEP
    updates_per_transition: float = DEFAULT_UPDATES_PER_TRANSITION
    warmup_steps: int = DEFAULT_WARMUP_STEPS
    target_tau: float = DEFAULT_TARGET_TAU
    prioritized_replay: bool = DEFAULT_PRIORITIZED_REPLAY
    priority_alpha: float = DEFAULT_PRIORITY_ALPHA
    priority_beta_start: float = DEFAULT_PRIORITY_BETA_START
    priority_beta_steps: int = DEFAULT_PRIORITY_BETA_STEPS
    ppo_rollout_steps: int = PPO_ROLLOUT_STEPS
    ppo_epochs: int = PPO_EPOCHS
    ppo_clip_range: float = PPO_CLIP_RANGE
    ppo_gae_lambda: float = PPO_GAE_LAMBDA
    ppo_value_coef: float = PPO_VALUE_COEF
    ppo_entropy_coef: float = PPO_ENTROPY_COEF
    ppo_max_grad_norm: float = PPO_MAX_GRAD_NORM
    eval_every: int = EVAL_PERIOD
    eval_every_steps: int = EVAL_PERIOD_STEPS
    eval_episodes: int = NUM_EVAL_EPISODES
    curriculum_promotion_rate: float = DEFAULT_CURRICULUM_PROMOTION_RATE
    curriculum_promotion_evals: int = DEFAULT_CURRICULUM_PROMOTION_EVALS
    curriculum_previous_fraction: float = DEFAULT_CURRICULUM_PREVIOUS_FRACTION
    curriculum_uniform_fraction: float = DEFAULT_CURRICULUM_UNIFORM_FRACTION
    vectorized_envs: bool = DEFAULT_VECTORIZED_ENVS
    checkpoint_replay: bool = DEFAULT_CHECKPOINT_REPLAY
    expert_pretrain_mazes: int = DEFAULT_EXPERT_PRETRAIN_MAZES
    expert_pretrain_epochs: int = DEFAULT_EXPERT_PRETRAIN_EPOCHS
    planner_fallback: bool = DEFAULT_PLANNER_FALLBACK
    dashboard_flag: bool = DEFAULT_DASHBOARD_FLAG
    dashboard_every: int = DEFAULT_DASHBOARD_EVERY
    save_path: str | None = DEFAULT_SAVE_PATH
    training_log_path: str | None = DEFAULT_TRAINING_LOG_PATH
    device: str = DEFAULT_DEVICE
    require_cuda: bool = DEFAULT_REQUIRE_CUDA

    def __post_init__(self) -> None:
        if self.save_path == "":
            self.save_path = None
        if self.training_log_path == "":
            self.training_log_path = None
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}; got {self.algorithm!r}")
        if self.network_type not in NETWORK_TYPES:
            raise ValueError(
                f"network_type must be one of {NETWORK_TYPES}; got {self.network_type!r}"
            )
        if self.observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {OBSERVATION_MODES}; "
                f"got {self.observation_mode!r}"
            )
        if self.planner_fallback and self.observation_mode != "full":
            raise ValueError("planner_fallback requires full observations")
        if self.distance_shaping_mode not in DISTANCE_SHAPING_MODES:
            raise ValueError(
                "distance_shaping_mode must be one of "
                f"{DISTANCE_SHAPING_MODES}; got {self.distance_shaping_mode!r}"
            )
        if len(self.maze_size) != 2 or min(self.maze_size) < 3:
            raise ValueError("maze_size must contain two dimensions >= 3")
        if any(size % 2 == 0 for size in self.maze_size):
            raise ValueError("maze_size dimensions must be odd for the maze generator")
        if self.view_size < 1 or self.view_size % 2 == 0:
            raise ValueError("view_size must be odd so the agent has a center cell")
        if self.episodes < 1:
            raise ValueError("episodes must be >= 1")
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.min_replay_size < 1:
            raise ValueError("min_replay_size must be >= 1")
        if self.buffer_size < self.min_replay_size:
            raise ValueError("buffer_size must be >= min_replay_size")
        if self.target_update_freq < 1:
            raise ValueError("target_update_freq must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 < self.gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        if not 0 <= self.epsilon_end <= self.epsilon_mid <= self.epsilon_start <= 1:
            raise ValueError("epsilon values must satisfy 0 <= end <= mid <= start <= 1")
        if self.epsilon_decay_steps < 1 or self.epsilon_final_steps < self.epsilon_decay_steps:
            raise ValueError("epsilon step boundaries must satisfy 1 <= decay <= final")
        if self.updates_per_transition < 0:
            raise ValueError("updates_per_transition must be non-negative")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if not 0 <= self.target_tau <= 1:
            raise ValueError("target_tau must be in [0, 1]")
        if self.priority_alpha < 0:
            raise ValueError("priority_alpha must be non-negative")
        if not 0 <= self.priority_beta_start <= 1:
            raise ValueError("priority_beta_start must be in [0, 1]")
        if self.priority_beta_steps < 1:
            raise ValueError("priority_beta_steps must be >= 1")
        if self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1")
        if self.timeout_step_factor is not None and self.timeout_step_factor <= 0:
            raise ValueError("timeout_step_factor must be positive or None")
        if self.min_episode_steps < 1:
            raise ValueError("min_episode_steps must be >= 1")
        if self.curriculum_easy_episodes < 1:
            raise ValueError("curriculum_easy_episodes must be >= 1")
        if self.curriculum_medium_episodes < self.curriculum_easy_episodes:
            raise ValueError(
                "curriculum_medium_episodes must be >= curriculum_easy_episodes"
            )
        if self.curriculum_max_retries < 1:
            raise ValueError("curriculum_max_retries must be >= 1")
        if not 0 < self.curriculum_promotion_rate <= 1:
            raise ValueError("curriculum_promotion_rate must be in (0, 1]")
        if self.curriculum_promotion_evals < 1:
            raise ValueError("curriculum_promotion_evals must be >= 1")
        if not 0 <= self.curriculum_previous_fraction <= 1:
            raise ValueError("curriculum_previous_fraction must be in [0, 1]")
        if not 0 <= self.curriculum_uniform_fraction <= 1:
            raise ValueError("curriculum_uniform_fraction must be in [0, 1]")
        if self.curriculum_previous_fraction + self.curriculum_uniform_fraction > 1:
            raise ValueError("curriculum sampling fractions must sum to at most 1")
        self._validate_distance_range("curriculum_easy_range", self.curriculum_easy_range)
        self._validate_distance_range(
            "curriculum_medium_range",
            self.curriculum_medium_range,
        )
        if self.n_step_returns < 1:
            raise ValueError("n_step_returns must be >= 1")
        if self.ppo_rollout_steps < 1:
            raise ValueError("ppo_rollout_steps must be >= 1")
        if self.ppo_epochs < 1:
            raise ValueError("ppo_epochs must be >= 1")
        if self.ppo_clip_range <= 0:
            raise ValueError("ppo_clip_range must be positive")
        if not 0 < self.ppo_gae_lambda <= 1:
            raise ValueError("ppo_gae_lambda must be in (0, 1]")
        if self.eval_every < 1:
            raise ValueError("eval_every must be >= 1")
        if self.eval_every_steps < 1:
            raise ValueError("eval_every_steps must be >= 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be >= 1")
        if self.expert_pretrain_mazes < 0 or self.expert_pretrain_epochs < 0:
            raise ValueError("expert pretraining settings must be non-negative")
        if self.dashboard_every < 1:
            raise ValueError("dashboard_every must be >= 1")

    @staticmethod
    def _validate_distance_range(name: str, value: tuple[int, int]) -> None:
        if len(value) != 2:
            raise ValueError(f"{name} must contain exactly two integers")
        low, high = value
        if low < 1 or high < low:
            raise ValueError(f"{name} must satisfy 1 <= low <= high")


@dataclass(slots=True)
class EpisodeStats:
    total_reward: float
    steps: int
    solved: bool
    timeout: bool
    invalid_moves: int
    optimal_steps: int


@dataclass(slots=True)
class EvalMetrics:
    solve_rate: float = 0.0
    solve_rate_lower_bound: float = 0.0
    avg_steps: float = 0.0
    optimality_ratio: float = 0.0
    timeout_rate: float = 0.0
    invalid_move_rate: float = 0.0
    loop_rate: float = 0.0
    repeated_state_action_rate: float = 0.0
    failed_final_distance: float = 0.0
    difficulty_solve_rates: dict[str, float] | None = None


@dataclass(slots=True)
class DashboardState:
    episode: int
    total_steps: int
    epsilon: float
    replay_size: int
    device: str
    maze_size: tuple[int, int]
    observation_mode: str
    train_solve_rate: float
    train_avg_steps: float
    train_timeout_rate: float
    train_invalid_rate: float
    reward_avg: float
    loss_ema: float
    greedy: EvalMetrics
    best_greedy_solve_rate: float
    steps_per_second: float
    episodes_per_second: float


ChartPoint = tuple[int, float]
RawTransition = tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
ReplayTransition = tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]


class TaskSampler(Protocol):
    """Create tasks and describe their difficulty without coupling to an agent."""

    def sample(self) -> "Maze":
        """Return one new training environment."""

    def describe(self, environment: "Maze") -> dict[str, float | str]:
        """Return task metadata used for curriculum and evaluation reporting."""


class ExpertPolicy(Protocol):
    """Optional source of target action distributions for an environment."""

    def optimal_action_distribution(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
    ) -> np.ndarray:
        """Return probabilities over legal optimal actions for one state."""


class Planner(Protocol):
    """Optional exact policy used as a fallback or benchmarking oracle."""

    def get_actions(
        self,
        states: np.ndarray,
        epsilon: float = 0.0,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return one legal action per state."""


@dataclass(slots=True)
class BenchmarkResult:
    """Metrics for validation, final-test, and stress evaluation suites."""

    validation: EvalMetrics
    final_test: EvalMetrics
    stress_test: EvalMetrics


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable training runs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device: str = "auto", require_cuda: bool = False) -> torch.device:
    """Resolve a torch device and fail clearly when CUDA is required."""

    requested = device.lower()
    if requested not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    cuda_available = torch.cuda.is_available()
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" or require_cuda:
        if not cuda_available:
            raise RuntimeError(
                "CUDA was requested, but PyTorch cannot see a CUDA device. "
                "Check the NVIDIA driver, container/WSL GPU passthrough, and "
                "that nvidia-smi works in this shell."
            )
        return torch.device("cuda")
    return torch.device("cuda" if cuda_available else "cpu")


def device_diagnostics(device: torch.device) -> str:
    """Return a compact, user-facing device summary."""

    cuda_available = torch.cuda.is_available()
    cuda_devices = torch.cuda.device_count() if cuda_available else 0
    cuda_name = "none"
    if cuda_devices > 0:
        cuda_name = torch.cuda.get_device_name(0)
    return (
        f"torch={torch.__version__} | cuda_build={torch.version.cuda} | "
        f"cuda_available={cuda_available} | "
        f"cuda_devices={cuda_devices} | selected={device} | "
        f"gpu={cuda_name}"
    )


def _atomic_torch_save(payload: object, path: str) -> None:
    """Write a checkpoint atomically so interrupted saves keep the old file."""

    absolute_path = os.path.abspath(path)
    directory = os.path.dirname(absolute_path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{absolute_path}.tmp-{uuid.uuid4().hex}"
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, absolute_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def observation_shape(
    maze_size: tuple[int, int],
    observation_mode: str = "full",
    view_size: int = VIEW_SIZE,
) -> tuple[int, int, int]:
    """Return the channel-first observation shape for a maze configuration."""

    if observation_mode == "full":
        return (3, maze_size[0], maze_size[1])
    if observation_mode == "local":
        return (3, view_size, view_size)
    raise ValueError(f"unsupported observation_mode: {observation_mode!r}")


def resolved_eval_seed(config: TrainConfig) -> int:
    """Return the deterministic evaluation seed for a training config."""

    if config.eval_seed is not None:
        return config.eval_seed
    return config.seed + EVAL_SEED_OFFSET


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Pre-allocated uniform or proportional-prioritized DQN replay buffer."""

    __slots__ = (
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "next_action_masks",
        "priorities",
        "capacity",
        "_pos",
        "_size",
        "_rng",
    )

    def __init__(
        self,
        observation_shape_: tuple[int, ...],
        capacity: int = BUFFER_SIZE,
        seed: int | None = None,
    ):
        self.capacity = int(capacity)
        self.states = np.empty((capacity, *observation_shape_), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.int64)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.next_states = np.empty((capacity, *observation_shape_), dtype=np.float32)
        self.dones = np.empty(capacity, dtype=np.float32)
        self.next_action_masks = np.empty((capacity, 4), dtype=np.bool_)
        self.priorities = np.ones(capacity, dtype=np.float32)
        self._pos = 0
        self._size = 0
        self._rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
    ) -> None:
        idx = self._pos % self.capacity
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)
        self.next_action_masks[idx] = self._normalize_action_mask(next_action_mask)
        self.priorities[idx] = self.priorities[: self._size].max(initial=1.0)
        self._pos += 1
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        prioritized: bool = False,
        alpha: float = DEFAULT_PRIORITY_ALPHA,
        beta: float = 1.0,
    ) -> tuple[np.ndarray, ...]:
        """Sample transitions with normalized importance weights.

        The final two arrays are buffer indices and importance weights. They
        make uniform and prioritized replay share one training call-site.
        """

        n = min(batch_size, self._size)
        if n < 1:
            raise ValueError("cannot sample an empty replay buffer")
        if prioritized and alpha > 0:
            scaled = np.maximum(self.priorities[: self._size], 1e-6) ** alpha
            probabilities = scaled / scaled.sum()
            idx = self._rng.choice(self._size, size=n, replace=True, p=probabilities)
            weights = (self._size * probabilities[idx]) ** (-beta)
            weights /= weights.max(initial=1.0)
        else:
            idx = self._rng.integers(0, self._size, size=n)
            weights = np.ones(n, dtype=np.float32)
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
            self.next_action_masks[idx],
            idx,
            weights.astype(np.float32, copy=False),
        )

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update sampled priorities from absolute TD errors."""

        index_array = np.asarray(indices, dtype=np.int64)
        priority_array = np.asarray(priorities, dtype=np.float32)
        if index_array.shape != priority_array.shape:
            raise ValueError("indices and priorities must have matching shapes")
        self.priorities[index_array] = np.maximum(priority_array, 1e-6)

    def state_dict(self) -> dict[str, object]:
        """Return replay contents and RNG state for optional exact resume."""

        return {
            "capacity": self.capacity,
            "pos": self._pos,
            "size": self._size,
            "states": self.states.copy(),
            "actions": self.actions.copy(),
            "rewards": self.rewards.copy(),
            "next_states": self.next_states.copy(),
            "dones": self.dones.copy(),
            "next_action_masks": self.next_action_masks.copy(),
            "priorities": self.priorities.copy(),
            "rng_state": self._rng.bit_generator.state,
        }

    def load_state_dict(self, payload: dict[str, object]) -> None:
        """Restore a replay snapshot after validating its capacity."""

        if int(payload["capacity"]) != self.capacity:
            raise ValueError("replay capacity does not match the loaded checkpoint")
        for name in (
            "states",
            "actions",
            "rewards",
            "next_states",
            "dones",
            "next_action_masks",
            "priorities",
        ):
            target = getattr(self, name)
            source = np.asarray(payload[name], dtype=target.dtype)
            if source.shape != target.shape:
                raise ValueError(f"replay field {name} has shape {source.shape}, expected {target.shape}")
            target[...] = source
        self._pos = int(payload["pos"])
        self._size = int(payload["size"])
        self._rng.bit_generator.state = payload["rng_state"]

    def __len__(self) -> int:
        return self._size

    @staticmethod
    def _normalize_action_mask(action_mask: np.ndarray | None) -> np.ndarray:
        if action_mask is None:
            return np.ones(4, dtype=np.bool_)
        mask = np.asarray(action_mask, dtype=np.bool_)
        if mask.shape != (4,):
            raise ValueError(f"action mask must have shape (4,), got {mask.shape}")
        if not mask.any():
            return np.ones(4, dtype=np.bool_)
        return mask


# ---------------------------------------------------------------------------
# Maze environment
# ---------------------------------------------------------------------------
class Maze:
    """Grid maze environment with full-map and local observation modes.

    Grid values are 0 for open cells, 1 for walls, 2 for start, and 3 for goal.
    Full-map observations use three channels: walls, agent position, and goal
    position. Local observations use the same channels in a centered view.
    """

    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
    ACTION_NAMES = [("Right", ">"), ("Left", "<"), ("Down", "v"), ("Up", "^")]

    def __init__(
        self,
        grid: np.ndarray,
        observation_mode: str = "full",
        view_size: int = VIEW_SIZE,
        max_episode_steps: int = MAX_EPISODE_STEPS,
        timeout_step_factor: float | None = None,
        min_episode_steps: int = 1,
        step_penalty: float = STEP_PENALTY,
        invalid_move_penalty: float = INVALID_MOVE_PENALTY,
        goal_reward: float = GOAL_REWARD,
        timeout_penalty: float = TIMEOUT_PENALTY,
        distance_shaping_scale: float = DISTANCE_SHAPING_SCALE,
        distance_shaping_mode: str = DEFAULT_DISTANCE_SHAPING_MODE,
        gamma: float = GAMMA,
    ):
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"unsupported observation_mode: {observation_mode!r}")
        if distance_shaping_mode not in DISTANCE_SHAPING_MODES:
            raise ValueError(f"unsupported distance_shaping_mode: {distance_shaping_mode!r}")
        self.grid = grid
        self.observation_mode = observation_mode
        self.view_size = view_size
        self.max_episode_steps = int(max_episode_steps)
        self.timeout_step_factor = timeout_step_factor
        self.min_episode_steps = int(min_episode_steps)
        self.step_penalty = float(step_penalty)
        self.invalid_move_penalty = float(invalid_move_penalty)
        self.goal_reward = float(goal_reward)
        self.timeout_penalty = float(timeout_penalty)
        self.distance_shaping_scale = float(distance_shaping_scale)
        self.distance_shaping_mode = distance_shaping_mode
        self.gamma = float(gamma)
        self.start = tuple(np.argwhere(self.grid == 2)[0])
        self.goal = tuple(np.argwhere(self.grid == 3)[0])
        self.current_position = self.start
        self.steps = 0
        self.invalid_moves = 0
        self.total_reward = 0.0
        self._compute_bfs_distances()
        self.optimal_start_steps = int(self.bfs_distances[self.start])
        if self.optimal_start_steps < 0:
            raise ValueError("maze start cannot reach goal")
        self.max_episode_steps = self._resolve_max_episode_steps()

    @property
    def observation_shape(self) -> tuple[int, int, int]:
        return observation_shape(
            self.grid.shape,
            observation_mode=self.observation_mode,
            view_size=self.view_size,
        )

    def _compute_bfs_distances(self) -> None:
        """Compute shortest-path distances from every reachable cell to goal."""

        dist = np.full(self.grid.shape, -1, dtype=np.float32)
        goal_r, goal_c = self.goal
        dist[goal_r, goal_c] = 0
        queue = deque([(goal_r, goal_c)])
        while queue:
            r, c = queue.popleft()
            next_dist = dist[r, c] + 1
            for dr, dc in self.ACTIONS:
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < self.grid.shape[0]
                    and 0 <= nc < self.grid.shape[1]
                    and self.grid[nr, nc] != 1
                    and dist[nr, nc] < 0
                ):
                    dist[nr, nc] = next_dist
                    queue.append((nr, nc))
        self.bfs_distances = dist

    def _resolve_max_episode_steps(self) -> int:
        if self.timeout_step_factor is None:
            return self.max_episode_steps
        shaped_limit = max(
            self.min_episode_steps,
            int(math.ceil(self.optimal_start_steps * self.timeout_step_factor)),
        )
        return min(self.max_episode_steps, shaped_limit)

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.invalid_moves = 0
        self.total_reward = 0.0
        self.current_position = self.start
        return self.observation()

    def observation(self) -> np.ndarray:
        if self.observation_mode == "full":
            return self._full_observation()
        return self._local_observation(self.current_position)

    def _full_observation(self) -> np.ndarray:
        obs = np.zeros((3, *self.grid.shape), dtype=np.float32)
        obs[0] = self.grid == 1
        obs[1, self.current_position[0], self.current_position[1]] = 1.0
        obs[2, self.goal[0], self.goal[1]] = 1.0
        return obs

    def _local_observation(self, position: tuple[int, int]) -> np.ndarray:
        half = self.view_size // 2
        obs = np.zeros((3, self.view_size, self.view_size), dtype=np.float32)
        obs[1, half, half] = 1.0
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                r, c = position[0] + di, position[1] + dj
                vi, vj = di + half, dj + half
                if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
                    obs[0, vi, vj] = 1.0
                    continue
                cell = self.grid[r, c]
                obs[0, vi, vj] = float(cell == 1)
                obs[2, vi, vj] = float((r, c) == self.goal)
        return obs

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float | bool | int]]:
        old_position = self.current_position
        old_distance = float(self.bfs_distances[old_position])
        dr, dc = self.ACTIONS[int(action)]
        nr, nc = old_position[0] + dr, old_position[1] + dc

        moved = False
        invalid = False
        if self._is_valid((nr, nc)):
            self.current_position = (nr, nc)
            moved = True
        else:
            invalid = True
            self.invalid_moves += 1

        new_distance = float(self.bfs_distances[self.current_position])
        solved = self.current_position == self.goal

        reward = self.step_penalty
        if invalid:
            reward += self.invalid_move_penalty
        reward += self.distance_shaping_scale * self._distance_shaping(
            old_distance,
            new_distance,
        )
        if solved:
            reward += self.goal_reward

        self.steps += 1
        timeout = (not solved) and self.steps >= self.max_episode_steps
        if timeout:
            reward += self.timeout_penalty

        done = solved or timeout
        self.total_reward += reward
        info = {
            "moved": moved,
            "invalid": invalid,
            "solved": solved,
            "timeout": timeout,
            "distance": new_distance,
            "optimal_steps": self.optimal_start_steps,
            "steps": self.steps,
            "invalid_moves": self.invalid_moves,
        }
        return self.observation(), reward, done, info

    def _distance_shaping(self, old_distance: float, new_distance: float) -> float:
        if self.distance_shaping_mode == "none" or old_distance <= 0:
            return 0.0
        if self.distance_shaping_mode == "fractional":
            return (old_distance - new_distance) / old_distance
        old_potential = -old_distance
        new_potential = -new_distance
        return self.gamma * new_potential - old_potential

    def _is_valid(self, position: tuple[int, int]) -> bool:
        r, c = position
        return (
            0 <= r < self.grid.shape[0]
            and 0 <= c < self.grid.shape[1]
            and self.grid[r, c] != 1
        )

    def valid_action_mask(self, position: tuple[int, int] | None = None) -> np.ndarray:
        """Return a boolean mask of actions that move into open cells."""

        base = self.current_position if position is None else position
        return np.array(
            [
                self._is_valid((base[0] + dr, base[1] + dc))
                for dr, dc in self.ACTIONS
            ],
            dtype=np.bool_,
        )

    def expert_action_distribution(
        self,
        position: tuple[int, int] | None = None,
    ) -> np.ndarray:
        """Return a uniform distribution over shortest-path legal actions.

        This is an optional environment-provided teaching signal. It does not
        participate in ordinary RL reward or action selection.
        """

        base = self.current_position if position is None else position
        action_mask = self.valid_action_mask(base)
        distance = float(self.bfs_distances[base])
        targets = np.zeros(4, dtype=np.float32)
        if distance <= 0:
            targets[action_mask] = 1.0 / max(int(action_mask.sum()), 1)
            return targets
        for action, (dr, dc) in enumerate(self.ACTIONS):
            if not action_mask[action]:
                continue
            neighbor = (base[0] + dr, base[1] + dc)
            if self.bfs_distances[neighbor] == distance - 1:
                targets[action] = 1.0
        if targets.sum() == 0:
            targets[action_mask] = 1.0
        return targets / targets.sum()

    def manhattan_to_goal(self) -> int:
        return abs(self.current_position[0] - self.goal[0]) + abs(
            self.current_position[1] - self.goal[1]
        )


class BfsPlanner:
    """Exact MouseMaze planner implementing the generic Planner protocol."""

    def get_actions(
        self,
        states: np.ndarray,
        epsilon: float = 0.0,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a shortest-path action for each full-map observation."""

        del epsilon
        state_batch = states[np.newaxis, ...] if states.ndim == 3 else states
        masks = None if action_masks is None else np.asarray(action_masks, dtype=np.bool_)
        if masks is not None and masks.ndim == 1:
            masks = masks[np.newaxis, :]
        actions = np.empty(state_batch.shape[0], dtype=np.int64)
        for index, state in enumerate(state_batch):
            walls = state[0] > 0.5
            position = tuple(np.argwhere(state[1] > 0.5)[0])
            goal = tuple(np.argwhere(state[2] > 0.5)[0])
            distances = _bfs_distances_from_observation(walls, goal)
            mask = (
                np.ones(4, dtype=np.bool_)
                if masks is None
                else masks[index]
            )
            candidates = []
            for action, (dr, dc) in enumerate(Maze.ACTIONS):
                row, col = position[0] + dr, position[1] + dc
                if (
                    mask[action]
                    and 0 <= row < walls.shape[0]
                    and 0 <= col < walls.shape[1]
                    and distances[row, col] >= 0
                ):
                    candidates.append((distances[row, col], action))
            actions[index] = min(candidates, default=(0, 0))[1]
        return actions

    def get_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Return one exact action for compatibility with scalar evaluation."""

        del epsilon
        return int(self.get_actions(state, action_mask)[0])


def _bfs_distances_from_observation(
    walls: np.ndarray,
    goal: tuple[int, int],
) -> np.ndarray:
    """Compute a distance map for a full-map planner observation."""

    distances = np.full(walls.shape, -1, dtype=np.int32)
    distances[goal] = 0
    queue = deque([goal])
    while queue:
        row, col = queue.popleft()
        for dr, dc in Maze.ACTIONS:
            next_row, next_col = row + dr, col + dc
            if (
                0 <= next_row < walls.shape[0]
                and 0 <= next_col < walls.shape[1]
                and not walls[next_row, next_col]
                and distances[next_row, next_col] < 0
            ):
                distances[next_row, next_col] = distances[row, col] + 1
                queue.append((next_row, next_col))
    return distances


def curriculum_distance_range(
    config: TrainConfig,
    episode: int | None,
) -> tuple[int, int] | None:
    """Return a size-scaled legacy curriculum range for a training episode.

    New training uses :class:`CurriculumController`; this helper remains for
    callers that rely on a deterministic episode-indexed schedule.
    """

    if (
        not config.curriculum_enabled
        or episode is None
    ):
        return None
    scale = max(config.maze_size) / max(DEFAULT_MAZE_SIZE)
    if episode <= config.curriculum_easy_episodes:
        return _scale_distance_range(config.curriculum_easy_range, scale)
    if episode <= config.curriculum_medium_episodes:
        return _scale_distance_range(config.curriculum_medium_range, scale)
    return None


def _scale_distance_range(
    value: tuple[int, int],
    scale: float,
) -> tuple[int, int]:
    low = max(1, int(round(value[0] * scale)))
    high = max(low, int(round(value[1] * scale)))
    return low, high


class CurriculumController:
    """Promote task difficulty only after repeated validation success."""

    def __init__(self, config: TrainConfig):
        self.config = config
        self.level = 0
        self.success_streak = 0

    @property
    def complete(self) -> bool:
        return self.level >= 2

    def target_range(self) -> tuple[int, int] | None:
        """Return the current maze path-length range, or uniform sampling."""

        if not self.config.curriculum_enabled or self.complete:
            return None
        scale = max(self.config.maze_size) / max(DEFAULT_MAZE_SIZE)
        source = (
            self.config.curriculum_easy_range
            if self.level == 0
            else self.config.curriculum_medium_range
        )
        return _scale_distance_range(source, scale)

    def previous_range(self) -> tuple[int, int] | None:
        """Return the immediately previous curriculum range when available."""

        if self.level < 1:
            return None
        scale = max(self.config.maze_size) / max(DEFAULT_MAZE_SIZE)
        return _scale_distance_range(self.config.curriculum_easy_range, scale)

    def record_validation(self, metrics: EvalMetrics) -> bool:
        """Record validation quality and return whether a promotion occurred."""

        if not self.config.curriculum_enabled or self.complete:
            return False
        if metrics.solve_rate >= self.config.curriculum_promotion_rate:
            self.success_streak += 1
        else:
            self.success_streak = 0
        if self.success_streak < self.config.curriculum_promotion_evals:
            return False
        self.level += 1
        self.success_streak = 0
        return True


class MazeTaskSampler:
    """Difficulty-aware MouseMaze sampler conforming to ``TaskSampler``."""

    def __init__(self, config: TrainConfig, rng: random.Random):
        self.config = config
        self.rng = rng
        self.curriculum = CurriculumController(config)

    def sample(self) -> Maze:
        """Sample from current, previous, and uniform task distributions."""

        target_range = self.curriculum.target_range()
        if target_range is not None:
            draw = self.rng.random()
            if draw < self.config.curriculum_uniform_fraction:
                target_range = None
            elif draw < (
                self.config.curriculum_uniform_fraction
                + self.config.curriculum_previous_fraction
            ):
                target_range = self.curriculum.previous_range()
        try:
            return make_maze(self.config, rng=self.rng, target_range=target_range)
        except TypeError as exc:
            if "target_range" not in str(exc):
                raise
            return make_maze(self.config, rng=self.rng)

    def describe(self, environment: Maze) -> dict[str, float | str]:
        """Describe a task using reusable difficulty metadata."""

        distance = environment.optimal_start_steps
        return {
            "difficulty": float(distance),
            "bucket": difficulty_bucket(distance),
        }


def _maze_from_grid(config: TrainConfig, grid: np.ndarray) -> Maze:
    return Maze(
        grid.copy(),
        observation_mode=config.observation_mode,
        view_size=config.view_size,
        max_episode_steps=config.max_episode_steps,
        timeout_step_factor=config.timeout_step_factor,
        min_episode_steps=config.min_episode_steps,
        step_penalty=config.step_penalty,
        invalid_move_penalty=config.invalid_move_penalty,
        goal_reward=config.goal_reward,
        timeout_penalty=config.timeout_penalty,
        distance_shaping_scale=config.distance_shaping_scale,
        distance_shaping_mode=config.distance_shaping_mode,
        gamma=config.gamma,
    )


def make_maze(
    config: TrainConfig,
    rng: random.Random | None = None,
    episode: int | None = None,
    target_range: tuple[int, int] | None = None,
) -> Maze:
    if target_range is None:
        target_range = curriculum_distance_range(config, episode)
    if target_range is None:
        grid = generate_random_maze(config.maze_size[0], config.maze_size[1], rng=rng)
        return _maze_from_grid(config, grid)

    low, high = target_range
    for _ in range(config.curriculum_max_retries):
        grid = generate_random_maze(config.maze_size[0], config.maze_size[1], rng=rng)
        env = _maze_from_grid(config, grid)
        if low <= env.optimal_start_steps <= high:
            return env

    grid = generate_random_maze(config.maze_size[0], config.maze_size[1], rng=rng)
    return _maze_from_grid(config, grid)


def make_training_maze(
    config: TrainConfig,
    rng: random.Random,
    episode: int,
    sampler: MazeTaskSampler | None = None,
) -> Maze:
    """Create a training maze while tolerating older test monkeypatches."""

    if sampler is not None:
        return sampler.sample()
    try:
        return make_maze(config, rng=rng, episode=episode)
    except TypeError as exc:
        if "episode" not in str(exc):
            raise
        return make_maze(config, rng=rng)


@dataclass(slots=True)
class BatchStep:
    """Results of stepping a same-shaped batch of full-map maze tasks."""

    states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    action_masks: np.ndarray
    invalid: np.ndarray
    solved: np.ndarray
    timeout: np.ndarray


class MazeBatch:
    """Vectorized full-observation Maze adapter used by DQN collection.

    Individual ``Maze`` objects remain the source of task construction and
    rendering. This adapter only batches their fixed-shape transition logic.
    """

    def __init__(self, environments: list[Maze]):
        if not environments:
            raise ValueError("MazeBatch requires at least one environment")
        first = environments[0]
        if first.observation_mode != "full":
            raise ValueError("MazeBatch supports full observations only")
        grid_shape = first.grid.shape
        if any(env.observation_mode != "full" or env.grid.shape != grid_shape for env in environments):
            raise ValueError("all batched mazes must use full observations and one grid shape")
        self.size = len(environments)
        self.grid_shape = grid_shape
        self.grids = np.empty((self.size, *grid_shape), dtype=np.uint8)
        self.bfs_distances = np.empty((self.size, *grid_shape), dtype=np.float32)
        self.starts = np.empty((self.size, 2), dtype=np.int64)
        self.goals = np.empty((self.size, 2), dtype=np.int64)
        self.positions = np.empty((self.size, 2), dtype=np.int64)
        self.steps = np.empty(self.size, dtype=np.int64)
        self.invalid_moves = np.empty(self.size, dtype=np.int64)
        self.total_rewards = np.empty(self.size, dtype=np.float32)
        self.optimal_steps = np.empty(self.size, dtype=np.int64)
        self.max_episode_steps = np.empty(self.size, dtype=np.int64)
        self.step_penalty = first.step_penalty
        self.invalid_move_penalty = first.invalid_move_penalty
        self.goal_reward = first.goal_reward
        self.timeout_penalty = first.timeout_penalty
        self.distance_shaping_scale = first.distance_shaping_scale
        self.distance_shaping_mode = first.distance_shaping_mode
        self.gamma = first.gamma
        for index, environment in enumerate(environments):
            self.replace(index, environment)

    def replace(self, index: int, environment: Maze) -> None:
        """Replace a completed slot with a freshly generated equivalent task."""

        if environment.observation_mode != "full" or environment.grid.shape != self.grid_shape:
            raise ValueError("replacement maze must match the batch observation shape")
        self.grids[index] = environment.grid
        self.bfs_distances[index] = environment.bfs_distances
        self.starts[index] = environment.start
        self.goals[index] = environment.goal
        self.positions[index] = environment.current_position
        self.steps[index] = environment.steps
        self.invalid_moves[index] = environment.invalid_moves
        self.total_rewards[index] = environment.total_reward
        self.optimal_steps[index] = environment.optimal_start_steps
        self.max_episode_steps[index] = environment.max_episode_steps

    def observations(self) -> np.ndarray:
        """Build batched full-map observations without a Python env loop."""

        observations = np.zeros((self.size, 3, *self.grid_shape), dtype=np.float32)
        observations[:, 0] = self.grids == 1
        indices = np.arange(self.size)
        observations[indices, 1, self.positions[:, 0], self.positions[:, 1]] = 1.0
        observations[indices, 2, self.goals[:, 0], self.goals[:, 1]] = 1.0
        return observations

    def valid_action_masks(self) -> np.ndarray:
        """Return legal action masks for every current batch position."""

        deltas = np.asarray(Maze.ACTIONS, dtype=np.int64)
        candidates = self.positions[:, np.newaxis, :] + deltas[np.newaxis, :, :]
        rows, cols = self.grid_shape
        in_bounds = (
            (candidates[:, :, 0] >= 0)
            & (candidates[:, :, 0] < rows)
            & (candidates[:, :, 1] >= 0)
            & (candidates[:, :, 1] < cols)
        )
        clipped_rows = np.clip(candidates[:, :, 0], 0, rows - 1)
        clipped_cols = np.clip(candidates[:, :, 1], 0, cols - 1)
        walls = self.grids[
            np.arange(self.size)[:, np.newaxis],
            clipped_rows,
            clipped_cols,
        ] == 1
        return in_bounds & ~walls

    def step(self, actions: np.ndarray) -> BatchStep:
        """Apply one action per environment and return vectorized transition data."""

        action_array = np.asarray(actions, dtype=np.int64)
        if action_array.shape != (self.size,):
            raise ValueError(f"actions must have shape ({self.size},), got {action_array.shape}")
        if np.any((action_array < 0) | (action_array >= len(Maze.ACTIONS))):
            raise ValueError("actions must be in [0, 3]")

        indices = np.arange(self.size)
        deltas = np.asarray(Maze.ACTIONS, dtype=np.int64)
        old_positions = self.positions.copy()
        action_masks = self.valid_action_masks()
        valid = action_masks[indices, action_array]
        candidates = old_positions + deltas[action_array]
        self.positions[valid] = candidates[valid]
        invalid = ~valid
        self.invalid_moves += invalid.astype(np.int64)

        old_distance = self.bfs_distances[indices, old_positions[:, 0], old_positions[:, 1]]
        new_distance = self.bfs_distances[
            indices,
            self.positions[:, 0],
            self.positions[:, 1],
        ]
        solved = np.all(self.positions == self.goals, axis=1)
        rewards = np.full(self.size, self.step_penalty, dtype=np.float32)
        rewards += invalid.astype(np.float32) * self.invalid_move_penalty
        if self.distance_shaping_mode == "potential":
            shaping = self.gamma * (-new_distance) - (-old_distance)
            rewards += self.distance_shaping_scale * np.where(old_distance > 0, shaping, 0.0)
        elif self.distance_shaping_mode == "fractional":
            shaping = np.divide(
                old_distance - new_distance,
                old_distance,
                out=np.zeros_like(old_distance),
                where=old_distance > 0,
            )
            rewards += self.distance_shaping_scale * shaping
        rewards += solved.astype(np.float32) * self.goal_reward
        self.steps += 1
        timeout = ~solved & (self.steps >= self.max_episode_steps)
        rewards += timeout.astype(np.float32) * self.timeout_penalty
        dones = solved | timeout
        self.total_rewards += rewards
        return BatchStep(
            states=self.observations(),
            rewards=rewards,
            dones=dones,
            action_masks=self.valid_action_masks(),
            invalid=invalid,
            solved=solved,
            timeout=timeout,
        )

    def episode_stats(self, index: int) -> EpisodeStats:
        """Return terminal statistics for one slot before replacing it."""

        solved = bool(np.array_equal(self.positions[index], self.goals[index]))
        timeout = (not solved) and self.steps[index] >= self.max_episode_steps[index]
        return EpisodeStats(
            total_reward=float(self.total_rewards[index]),
            steps=int(self.steps[index]),
            solved=solved,
            timeout=bool(timeout),
            invalid_moves=int(self.invalid_moves[index]),
            optimal_steps=int(self.optimal_steps[index]),
        )


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """Small convolutional residual block for spatial maze policies."""

    def __init__(self, channels: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x + self.layers(x))


class SpatialTrunk(nn.Module):
    """Fully convolutional trunk over wall, goal, and coordinate channels."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channels: int = 64,
        residual_blocks: int = 6,
    ):
        super().__init__()
        _input_channels, rows, cols = input_shape
        row_coords = torch.linspace(-1.0, 1.0, rows)
        col_coords = torch.linspace(-1.0, 1.0, cols)
        row_plane = row_coords.view(1, rows, 1).expand(1, rows, cols)
        col_plane = col_coords.view(1, 1, cols).expand(1, rows, cols)
        self.register_buffer(
            "coordinate_planes",
            torch.cat((row_plane, col_plane), dim=0),
            persistent=False,
        )
        layers: list[nn.Module] = [
            nn.Conv2d(4, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        layers.extend(ResidualBlock(channels) for _ in range(residual_blocks))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        walls_and_goal = torch.cat((x[:, 0:1], x[:, 2:3]), dim=1)
        coordinates = self.coordinate_planes.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        return self.net(torch.cat((walls_and_goal, coordinates), dim=1))


def _gather_agent_cell(cell_values: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    agent_mask = observations[:, 1:2]
    return (cell_values * agent_mask).flatten(2).sum(dim=2)


class SpatialQNetwork(nn.Module):
    """Dueling Q-map network that gathers Q-values at the agent cell."""

    def __init__(self, input_shape: tuple[int, int, int], output_size: int = 4):
        super().__init__()
        _channels, _rows, _cols = input_shape
        hidden_channels = 64
        self.trunk = SpatialTrunk(input_shape, hidden_channels)
        self.value_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.advantage_head = nn.Conv2d(hidden_channels, output_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value_map = self.value_head(features)
        advantage_map = self.advantage_head(features)
        q_map = value_map + advantage_map - advantage_map.mean(dim=1, keepdim=True)
        return _gather_agent_cell(q_map, x)


class FlatQNetwork(nn.Module):
    """Legacy flat dueling CNN kept for old checkpoints and comparisons."""

    def __init__(self, input_shape: tuple[int, int, int], output_size: int = 4):
        super().__init__()
        channels, rows, cols = input_shape
        self.conv = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, channels, rows, cols)
            flat_size = int(self.conv(dummy).flatten(1).shape[1])
        self.features = nn.Sequential(
            nn.Linear(flat_size, 256),
            nn.ReLU(),
        )
        self.value = nn.Linear(256, 1)
        self.advantage = nn.Linear(256, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x).flatten(1)
        hidden = self.features(features)
        value = self.value(hidden)
        advantage = self.advantage(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


QNetwork = SpatialQNetwork


def build_q_network(
    input_shape: tuple[int, int, int],
    network_type: str,
) -> nn.Module:
    if network_type == "spatial":
        return SpatialQNetwork(input_shape)
    if network_type == "flat":
        return FlatQNetwork(input_shape)
    raise ValueError(f"unsupported network_type: {network_type!r}")


def infer_q_network_type(state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("trunk.") for key in state_dict):
        return "spatial"
    return "flat"


class MaskedActorCriticNetwork(nn.Module):
    """Spatial actor-critic network that gathers logits/value at the agent cell."""

    def __init__(self, input_shape: tuple[int, int, int], output_size: int = 4):
        super().__init__()
        _channels, _rows, _cols = input_shape
        hidden_channels = 64
        self.trunk = SpatialTrunk(input_shape, hidden_channels)
        self.policy_head = nn.Conv2d(hidden_channels, output_size, kernel_size=1)
        self.value_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        logits = _gather_agent_cell(self.policy_head(features), x)
        value = _gather_agent_cell(self.value_head(features), x).squeeze(1)
        return logits, value


class MouseAgent:
    """Dueling Double DQN agent for MouseMaze."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        observation_shape_: tuple[int, int, int] | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or TrainConfig()
        self.device = device or select_device(self.config.device, self.config.require_cuda)
        self.algorithm = "dqn"
        self.network_type = self.config.network_type
        self.observation_mode = self.config.observation_mode
        self.view_size = self.config.view_size
        self.observation_shape = observation_shape_ or observation_shape(
            self.config.maze_size,
            self.config.observation_mode,
            self.config.view_size,
        )
        self._build_networks(self.network_type)
        self.buffer = ReplayBuffer(
            self.observation_shape,
            self.config.buffer_size,
            seed=self.config.seed,
        )
        self.update_count = 0
        self.total_env_steps = 0
        self.best_greedy_solve_rate = -1.0
        self.training_state: dict[str, object] = {}

    def _build_networks(self, network_type: str) -> None:
        self.network_type = network_type
        self.online_net = build_q_network(self.observation_shape, network_type).to(self.device)
        self.target_net = build_q_network(self.observation_shape, network_type).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.online_net.parameters(), lr=self.config.learning_rate)

    def get_actions(
        self,
        states: np.ndarray,
        epsilon: float = 0.0,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        if states.ndim == len(self.observation_shape):
            states = states[np.newaxis, ...]
        masks = self._normalize_action_masks(action_masks, states.shape[0])
        with torch.no_grad():
            q_values = self.online_net(
                torch.as_tensor(states, dtype=torch.float32, device=self.device)
            )
            if masks is not None:
                mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
                q_values = q_values.masked_fill(~mask_tensor, -torch.inf)
            actions = q_values.argmax(dim=1).cpu().numpy().astype(np.int64)
        if epsilon > 0:
            random_mask = np.random.random(size=actions.shape[0]) < epsilon
            if masks is None:
                actions[random_mask] = np.random.randint(0, 4, size=random_mask.sum())
            else:
                for idx in np.flatnonzero(random_mask):
                    valid_actions = np.flatnonzero(masks[idx])
                    actions[idx] = np.random.choice(valid_actions)
        return actions

    def get_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        return int(self.get_actions(state, epsilon=epsilon, action_masks=action_mask)[0])

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            q_values = self.online_net(
                torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            )
        return q_values.cpu().numpy()[0]

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray | None = None,
        count_env_step: bool = True,
    ) -> None:
        self.buffer.push(state, action, reward, next_state, done, next_action_mask)
        if count_env_step:
            self.total_env_steps += 1

    def train_step(self) -> float | None:
        if len(self.buffer) < self.config.min_replay_size:
            return None

        priority_progress = min(
            self.total_env_steps / self.config.priority_beta_steps,
            1.0,
        )
        priority_beta = self.config.priority_beta_start + (
            1.0 - self.config.priority_beta_start
        ) * priority_progress
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            next_action_masks,
            replay_indices,
            importance_weights,
        ) = self.buffer.sample(
            self.config.batch_size,
            prioritized=self.config.prioritized_replay,
            alpha=self.config.priority_alpha,
            beta=priority_beta,
        )
        s = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        r = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(dones, dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(next_action_masks, dtype=torch.bool, device=self.device)
        weights = torch.as_tensor(importance_weights, dtype=torch.float32, device=self.device)

        q_values = self.online_net(s)
        q_a = q_values.gather(1, a).squeeze(1)

        with torch.no_grad():
            online_next_q = self.online_net(ns).masked_fill(~next_masks, -torch.inf)
            next_actions = online_next_q.argmax(dim=1, keepdim=True)
            target_next_q = self.target_net(ns).masked_fill(~next_masks, -torch.inf)
            next_q = target_next_q.gather(1, next_actions).squeeze(1)
            bootstrap_discount = self.config.gamma ** self.config.n_step_returns
            target_q = r + bootstrap_discount * next_q * (1.0 - d)

        td_errors = target_q - q_a
        loss = (nn.functional.smooth_l1_loss(q_a, target_q, reduction="none") * weights).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()

        if self.config.prioritized_replay:
            self.buffer.update_priorities(
                replay_indices,
                td_errors.detach().abs().cpu().numpy() + 1e-6,
            )

        self.update_count += 1
        if self.config.target_tau > 0:
            with torch.no_grad():
                for target, online in zip(
                    self.target_net.parameters(),
                    self.online_net.parameters(),
                    strict=True,
                ):
                    target.lerp_(online, self.config.target_tau)
        elif self.update_count % self.config.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())
        return float(loss.item())

    def save(self, path: str) -> None:
        payload = {
            "algorithm": self.algorithm,
            "network_type": self.network_type,
            "state_dict": self.online_net.state_dict(),
            "target_state_dict": self.target_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "observation_shape": self.observation_shape,
            "observation_mode": self.observation_mode,
            "view_size": self.view_size,
            "maze_size": self.config.maze_size,
            "update_count": self.update_count,
            "total_env_steps": self.total_env_steps,
            "best_greedy_solve_rate": self.best_greedy_solve_rate,
            "training_state": self.training_state,
        }
        if self.config.checkpoint_replay:
            payload["replay_state"] = self.buffer.state_dict()
        _atomic_torch_save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        if isinstance(payload, dict):
            network_type = payload.get("network_type") or infer_q_network_type(state_dict)
        else:
            network_type = infer_q_network_type(state_dict)
        if network_type != self.network_type:
            self._build_networks(network_type)
        self.online_net.load_state_dict(state_dict)
        if isinstance(payload, dict) and "target_state_dict" in payload:
            self.target_net.load_state_dict(payload["target_state_dict"])
        else:
            self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()
        if isinstance(payload, dict) and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if isinstance(payload, dict):
            self.update_count = int(payload.get("update_count", self.update_count))
            self.total_env_steps = int(payload.get("total_env_steps", self.total_env_steps))
            self.best_greedy_solve_rate = float(
                payload.get("best_greedy_solve_rate", self.best_greedy_solve_rate)
            )
            self.training_state = dict(payload.get("training_state", {}))
            replay_state = payload.get("replay_state")
            if replay_state is not None:
                self.buffer.load_state_dict(replay_state)

    @staticmethod
    def _normalize_action_masks(
        action_masks: np.ndarray | None,
        batch_size: int,
    ) -> np.ndarray | None:
        if action_masks is None:
            return None
        masks = np.asarray(action_masks, dtype=np.bool_).copy()
        if masks.ndim == 1:
            masks = masks[np.newaxis, :]
        if masks.shape != (batch_size, 4):
            raise ValueError(
                f"action_masks must have shape ({batch_size}, 4), got {masks.shape}"
            )
        empty_rows = ~masks.any(axis=1)
        if empty_rows.any():
            masks[empty_rows] = True
        return masks


class MaskedPPOAgent:
    """Masked PPO actor-critic agent for MouseMaze."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        observation_shape_: tuple[int, int, int] | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or TrainConfig(algorithm="ppo")
        self.device = device or select_device(self.config.device, self.config.require_cuda)
        self.algorithm = "ppo"
        self.network_type = "spatial"
        self.observation_mode = self.config.observation_mode
        self.view_size = self.config.view_size
        self.observation_shape = observation_shape_ or observation_shape(
            self.config.maze_size,
            self.config.observation_mode,
            self.config.view_size,
        )
        self.policy_net = MaskedActorCriticNetwork(self.observation_shape).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.learning_rate)
        self.update_count = 0
        self.total_env_steps = 0
        self.best_greedy_solve_rate = -1.0
        self.training_state: dict[str, object] = {}

    def _logits_and_values(
        self,
        states: np.ndarray,
        action_masks: np.ndarray | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if states.ndim == len(self.observation_shape):
            states = states[np.newaxis, ...]
        masks = MouseAgent._normalize_action_masks(action_masks, states.shape[0])
        observations = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        logits, values = self.policy_net(observations)
        if masks is not None:
            mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=self.device)
            logits = logits.masked_fill(~mask_tensor, -torch.inf)
        return logits, values

    def get_actions(
        self,
        states: np.ndarray,
        epsilon: float = 0.0,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        if states.ndim == len(self.observation_shape):
            states = states[np.newaxis, ...]
        masks = MouseAgent._normalize_action_masks(action_masks, states.shape[0])
        with torch.no_grad():
            logits, _values = self._logits_and_values(states, masks)
            actions = logits.argmax(dim=1).cpu().numpy().astype(np.int64)
        if epsilon > 0:
            random_mask = np.random.random(size=actions.shape[0]) < epsilon
            if masks is None:
                actions[random_mask] = np.random.randint(0, 4, size=random_mask.sum())
            else:
                for idx in np.flatnonzero(random_mask):
                    valid_actions = np.flatnonzero(masks[idx])
                    actions[idx] = np.random.choice(valid_actions)
        return actions

    def get_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        return int(self.get_actions(state, epsilon=epsilon, action_masks=action_mask)[0])

    def q_values(self, state: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits, _value = self._logits_and_values(state, None)
        return logits.cpu().numpy()[0]

    def sample_actions(
        self,
        states: np.ndarray,
        action_masks: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        logits, values = self._logits_and_values(states, action_masks)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = distribution.sample()
        log_probs = distribution.log_prob(actions)
        return (
            actions.cpu().numpy().astype(np.int64),
            log_probs.detach().cpu().numpy().astype(np.float32),
            values.detach().cpu().numpy().astype(np.float32),
        )

    def evaluate_actions(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, values = self.policy_net(states)
        logits = logits.masked_fill(~action_masks, -torch.inf)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return log_probs, entropy, values

    def save(self, path: str) -> None:
        payload = {
            "algorithm": self.algorithm,
            "network_type": self.network_type,
            "state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "observation_shape": self.observation_shape,
            "observation_mode": self.observation_mode,
            "view_size": self.view_size,
            "maze_size": self.config.maze_size,
            "update_count": self.update_count,
            "total_env_steps": self.total_env_steps,
            "best_greedy_solve_rate": self.best_greedy_solve_rate,
            "training_state": self.training_state,
        }
        _atomic_torch_save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
        self.policy_net.load_state_dict(state_dict)
        if isinstance(payload, dict) and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if isinstance(payload, dict):
            self.update_count = int(payload.get("update_count", self.update_count))
            self.total_env_steps = int(payload.get("total_env_steps", self.total_env_steps))
            self.best_greedy_solve_rate = float(
                payload.get("best_greedy_solve_rate", self.best_greedy_solve_rate)
            )
            self.training_state = dict(payload.get("training_state", {}))


Agent = MouseAgent | MaskedPPOAgent


def collect_maze_expert_examples(
    config: TrainConfig,
    rng: random.Random,
    maze_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect full-map observations and shortest-path targets from tasks.

    The collector is deliberately isolated from DQN so another environment can
    replace it with demonstrations implementing the same three returned arrays.
    """

    if maze_count < 1:
        raise ValueError("maze_count must be >= 1")
    states: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(maze_count):
        environment = make_maze(config, rng=rng, episode=None)
        for row, col in np.argwhere(environment.bfs_distances >= 0):
            position = (int(row), int(col))
            environment.current_position = position
            states.append(environment.observation())
            masks.append(environment.valid_action_mask(position))
            targets.append(environment.expert_action_distribution(position))
        environment.reset()
    return (
        np.stack(states).astype(np.float32),
        np.stack(masks).astype(np.bool_),
        np.stack(targets).astype(np.float32),
    )


def pretrain_with_expert(
    agent: Agent,
    config: TrainConfig,
) -> float | None:
    """Optionally initialize an agent from environment-provided demonstrations."""

    if config.expert_pretrain_mazes == 0 or config.expert_pretrain_epochs == 0:
        return None
    if config.observation_mode != "full":
        raise ValueError("expert pretraining currently requires full observations")
    states, masks, targets = collect_maze_expert_examples(
        config,
        random.Random(config.seed + 17),
        config.expert_pretrain_mazes,
    )
    rng = np.random.default_rng(config.seed + 31)
    batch_size = min(config.batch_size, len(states))
    losses: list[float] = []
    for _ in range(config.expert_pretrain_epochs):
        for indices in np.array_split(rng.permutation(len(states)), math.ceil(len(states) / batch_size)):
            state_batch = torch.as_tensor(states[indices], dtype=torch.float32, device=agent.device)
            mask_batch = torch.as_tensor(masks[indices], dtype=torch.bool, device=agent.device)
            target_batch = torch.as_tensor(targets[indices], dtype=torch.float32, device=agent.device)
            if isinstance(agent, MouseAgent):
                logits = agent.online_net(state_batch)
            else:
                logits, _values = agent.policy_net(state_batch)
            logits = logits.masked_fill(~mask_batch, -torch.inf)
            log_probabilities = torch.log_softmax(logits, dim=1)
            positive_targets = target_batch > 0
            loss = -(
                target_batch[positive_targets] * log_probabilities[positive_targets]
            ).sum() / len(indices)
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(
                agent.online_net.parameters() if isinstance(agent, MouseAgent) else agent.policy_net.parameters(),
                config.ppo_max_grad_norm if isinstance(agent, MaskedPPOAgent) else 10.0,
            )
            agent.optimizer.step()
            agent.update_count += 1
            losses.append(float(loss.item()))
    if isinstance(agent, MouseAgent):
        agent.target_net.load_state_dict(agent.online_net.state_dict())
    return sum(losses) / len(losses)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
class MetricsTracker:
    """Rolling training and evaluation metrics."""

    def __init__(self, window: int = 100):
        self.rewards = deque(maxlen=window)
        self.losses = deque(maxlen=500)
        self.steps = deque(maxlen=window)
        self.solved = deque(maxlen=window)
        self.timeouts = deque(maxlen=window)
        self.invalid_rates = deque(maxlen=window)
        self.success_steps = deque(maxlen=window)
        self.train_solve_history: list[ChartPoint] = []
        self.reward_history: list[ChartPoint] = []
        self.loss_history: list[ChartPoint] = []
        self.greedy_solve_history: list[ChartPoint] = []
        self.greedy_steps_history: list[ChartPoint] = []
        self.greedy_optimality_history: list[ChartPoint] = []
        self.latest_eval = EvalMetrics()

    def record_episode(self, stats: EpisodeStats, episode: int | None = None) -> None:
        self.rewards.append(stats.total_reward)
        self.steps.append(stats.steps)
        self.solved.append(1 if stats.solved else 0)
        self.timeouts.append(1 if stats.timeout else 0)
        self.invalid_rates.append(stats.invalid_moves / max(stats.steps, 1))
        if stats.solved:
            self.success_steps.append(stats.steps)
        x_value = episode if episode is not None else len(self.reward_history) + 1
        self.train_solve_history.append((x_value, self.train_solve_rate))
        self.reward_history.append((x_value, self.avg_reward))

    def record_loss(self, loss: float | None, episode: int | None = None) -> None:
        if loss is None:
            return
        self.losses.append(loss)
        x_value = episode if episode is not None else len(self.loss_history) + 1
        self.loss_history.append((x_value, self.loss_ema))

    def record_eval(self, metrics: EvalMetrics, episode: int | None = None) -> None:
        self.latest_eval = metrics
        x_value = episode if episode is not None else len(self.greedy_solve_history) + 1
        self.greedy_solve_history.append((x_value, metrics.solve_rate))
        self.greedy_steps_history.append((x_value, metrics.avg_steps))
        self.greedy_optimality_history.append((x_value, metrics.optimality_ratio))

    @property
    def train_solve_rate(self) -> float:
        return sum(self.solved) / len(self.solved) if self.solved else 0.0

    @property
    def train_timeout_rate(self) -> float:
        return sum(self.timeouts) / len(self.timeouts) if self.timeouts else 0.0

    @property
    def train_invalid_rate(self) -> float:
        return sum(self.invalid_rates) / len(self.invalid_rates) if self.invalid_rates else 0.0

    @property
    def avg_reward(self) -> float:
        return sum(self.rewards) / len(self.rewards) if self.rewards else 0.0

    @property
    def avg_steps(self) -> float:
        return sum(self.steps) / len(self.steps) if self.steps else 0.0

    @property
    def avg_success_steps(self) -> float:
        return sum(self.success_steps) / len(self.success_steps) if self.success_steps else 0.0

    @property
    def loss_ema(self) -> float:
        if not self.losses:
            return 0.0
        alpha = 0.08
        value = self.losses[0]
        for loss in self.losses:
            value = alpha * loss + (1.0 - alpha) * value
        return value


def linear_epsilon(total_env_steps: int, config: TrainConfig) -> float:
    """Return the transition-indexed two-stage exploration rate."""

    if total_env_steps <= config.epsilon_decay_steps:
        progress = total_env_steps / config.epsilon_decay_steps
        return config.epsilon_start + (config.epsilon_mid - config.epsilon_start) * progress
    if total_env_steps <= config.epsilon_final_steps:
        progress = (total_env_steps - config.epsilon_decay_steps) / (
            config.epsilon_final_steps - config.epsilon_decay_steps
        )
        return config.epsilon_mid + (config.epsilon_end - config.epsilon_mid) * progress
    return config.epsilon_end


def build_n_step_transition(
    transitions: deque[RawTransition],
    gamma: float,
    n_steps: int,
) -> ReplayTransition:
    """Build an n-step replay transition from the front of a transition queue."""

    reward = 0.0
    next_state = transitions[0][3]
    done = transitions[0][4]
    next_action_mask = transitions[0][5]
    for idx, transition in enumerate(list(transitions)[:n_steps]):
        reward += (gamma**idx) * transition[2]
        next_state = transition[3]
        done = transition[4]
        next_action_mask = transition[5]
        if done:
            break

    state, action, _reward, _next_state, _done, _mask = transitions[0]
    return state, action, reward, next_state, done, next_action_mask


def store_ready_n_step_transition(
    agent: MouseAgent,
    transitions: deque[RawTransition],
    config: TrainConfig,
) -> None:
    if len(transitions) < config.n_step_returns:
        return
    agent.store_transition(
        *build_n_step_transition(transitions, config.gamma, config.n_step_returns),
        count_env_step=False,
    )
    transitions.popleft()


def flush_n_step_transitions(
    agent: MouseAgent,
    transitions: deque[RawTransition],
    config: TrainConfig,
) -> None:
    while transitions:
        agent.store_transition(
            *build_n_step_transition(transitions, config.gamma, config.n_step_returns),
            count_env_step=False,
        )
        transitions.popleft()


def episode_stats_from_env(env: Maze) -> EpisodeStats:
    solved = env.current_position == env.goal
    timeout = (not solved) and env.steps >= env.max_episode_steps
    return EpisodeStats(
        total_reward=env.total_reward,
        steps=env.steps,
        solved=solved,
        timeout=timeout,
        invalid_moves=env.invalid_moves,
        optimal_steps=env.optimal_start_steps,
    )


def difficulty_bucket(optimal_steps: int) -> str:
    """Map a task difficulty value into a stable benchmark bucket label."""

    if optimal_steps <= 8:
        return "1-8"
    if optimal_steps <= 16:
        return "9-16"
    if optimal_steps <= 24:
        return "17-24"
    return "25+"


def _wilson_lower_bound(successes: int, total: int, z_score: float = 1.96) -> float:
    """Return a conservative two-sided 95% binomial lower confidence bound."""

    if total < 1:
        return 0.0
    rate = successes / total
    denominator = 1.0 + z_score**2 / total
    centre = rate + z_score**2 / (2.0 * total)
    spread = z_score * math.sqrt(
        rate * (1.0 - rate) / total + z_score**2 / (4.0 * total**2)
    )
    return max(0.0, (centre - spread) / denominator)


def _agent_greedy_actions(
    agent: Agent,
    states: np.ndarray,
    action_masks: np.ndarray,
) -> np.ndarray:
    if hasattr(agent, "get_actions"):
        return agent.get_actions(states, epsilon=0.0, action_masks=action_masks)
    return np.array(
        [agent.get_action(state, epsilon=0.0) for state in states],
        dtype=np.int64,
    )


def _eval_greedy(
    agent: Agent,
    config: TrainConfig,
    maze_factory: Callable[[], Maze] | None = None,
) -> EvalMetrics:
    """Run epsilon=0 evaluation on a fixed batch of mazes."""

    solved_steps: list[int] = []
    optimality: list[float] = []
    timeouts = 0
    invalid_moves = 0
    total_steps = 0
    loop_episodes = 0
    repeated_state_action_episodes = 0
    failed_final_distances: list[float] = []
    bucket_totals: dict[str, int] = {}
    bucket_solves: dict[str, int] = {}
    eval_rng = random.Random(resolved_eval_seed(config))
    make_env = maze_factory or (lambda: make_maze(config, rng=eval_rng))

    envs = [make_env() for _ in range(config.eval_episodes)]
    buckets = [difficulty_bucket(env.optimal_start_steps) for env in envs]
    for bucket in buckets:
        bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1
    states = [env.reset() for env in envs]
    active = np.ones(len(envs), dtype=np.bool_)
    position_traces = [[env.current_position] for env in envs]
    seen_state_actions: list[set[tuple[tuple[int, int], int]]] = [set() for _ in envs]
    has_loop = np.zeros(len(envs), dtype=np.bool_)
    has_repeated_state_action = np.zeros(len(envs), dtype=np.bool_)

    while active.any():
        active_indices = np.flatnonzero(active)
        state_batch = np.stack([states[idx] for idx in active_indices])
        action_masks = np.stack(
            [envs[idx].valid_action_mask() for idx in active_indices]
        )
        actions = _agent_greedy_actions(agent, state_batch, action_masks)

        for batch_idx, env_idx in enumerate(active_indices):
            env = envs[env_idx]
            action = int(actions[batch_idx])
            state_action = (env.current_position, action)
            if state_action in seen_state_actions[env_idx]:
                has_repeated_state_action[env_idx] = True
            seen_state_actions[env_idx].add(state_action)

            state, _reward, done, info = env.step(action)
            states[env_idx] = state
            invalid_moves += int(info["invalid"])
            total_steps += 1
            position_traces[env_idx].append(env.current_position)
            positions = position_traces[env_idx]
            if (
                len(positions) >= 5
                and positions[-1] == positions[-3]
                and positions[-2] == positions[-4]
            ):
                has_loop[env_idx] = True

            if done:
                active[env_idx] = False
                if env.current_position == env.goal:
                    solved_steps.append(env.steps)
                    optimality.append(env.optimal_start_steps / max(env.steps, 1))
                    bucket = buckets[env_idx]
                    bucket_solves[bucket] = bucket_solves.get(bucket, 0) + 1
                else:
                    timeouts += 1
                    loop_episodes += int(has_loop[env_idx])
                    repeated_state_action_episodes += int(
                        has_repeated_state_action[env_idx]
                    )
                    failed_final_distances.append(
                        float(env.bfs_distances[env.current_position])
                    )

    total = max(config.eval_episodes, 1)
    return EvalMetrics(
        solve_rate=len(solved_steps) / total,
        solve_rate_lower_bound=_wilson_lower_bound(len(solved_steps), total),
        avg_steps=sum(solved_steps) / len(solved_steps) if solved_steps else 0.0,
        optimality_ratio=sum(optimality) / len(optimality) if optimality else 0.0,
        timeout_rate=timeouts / total,
        invalid_move_rate=invalid_moves / max(total_steps, 1),
        loop_rate=loop_episodes / total,
        repeated_state_action_rate=repeated_state_action_episodes / total,
        failed_final_distance=(
            sum(failed_final_distances) / len(failed_final_distances)
            if failed_final_distances
            else 0.0
        ),
        difficulty_solve_rates={
            bucket: bucket_solves.get(bucket, 0) / count
            for bucket, count in sorted(bucket_totals.items())
        },
    )


def run_benchmark(
    agent: Agent,
    config: TrainConfig,
    episodes: int = 2_000,
) -> BenchmarkResult:
    """Evaluate one policy on deterministic, disjoint benchmark suites."""

    if episodes < 1:
        raise ValueError("benchmark episodes must be >= 1")
    root_seed = resolved_eval_seed(config)
    suites = {
        "validation": root_seed,
        "final_test": root_seed + 1,
        "stress_test": root_seed + 2,
    }
    metrics = {
        name: _eval_greedy(
            agent,
            replace(config, eval_seed=seed, eval_episodes=episodes),
        )
        for name, seed in suites.items()
    }
    return BenchmarkResult(
        validation=metrics["validation"],
        final_test=metrics["final_test"],
        stress_test=metrics["stress_test"],
    )


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
Rect = tuple[int, int, int, int]

DASHBOARD_TOOLTIPS = {
    "header": (
        "Training status summary. Greedy solve is the main policy quality signal; "
        "higher is better."
    ),
    "metric_0": (
        "Greedy Solve: percent of evaluation mazes solved with epsilon=0. "
        "Higher is better."
    ),
    "metric_1": (
        "Greedy Steps: average steps used on solved evaluation mazes. "
        "Lower is better after solve rate is strong."
    ),
    "metric_2": (
        "Optimality: shortest-path steps divided by actual solved steps. "
        "Higher is better, with 100% meaning shortest-path solves."
    ),
    "metric_3": (
        "Train Solve: rolling percent of training episodes that reached the goal. "
        "Higher is better, but greedy solve is the cleaner policy metric."
    ),
    "metric_4": (
        "Timeouts: rolling percent of training episodes that hit the step limit. "
        "Lower is better."
    ),
    "metric_5": (
        "Invalid Moves: rolling share of moves blocked by walls or bounds. "
        "Lower is better."
    ),
    "metric_6": (
        "Reward: rolling average shaped training reward. Higher is generally better, "
        "but solve metrics matter more."
    ),
    "metric_7": (
        "Loss EMA: smoothed TD error for DQN updates. Lower is generally better, "
        "but it can bounce while the replay data changes."
    ),
    "chart_reward": (
        "Reward history: full-run rolling average shaped reward over training. "
        "Higher is generally better."
    ),
    "chart_loss": (
        "Loss EMA history: full-run smoothed TD error trend. Lower is generally better, "
        "but policy improvement can happen while loss moves around."
    ),
    "chart_greedy": (
        "Greedy Solve history: full-run evaluation solve rate with no exploration. "
        "Higher is better."
    ),
}


def dashboard_layout(width: int, height: int) -> dict[str, Rect]:
    """Return fixed dashboard rectangles that do not overlap."""

    margin = 16
    gap = 12
    min_width = 900
    width = max(width, min_width)
    header_h = 74
    metric_h = 66
    chart_h = max(180, height - (margin * 2 + header_h + gap + metric_h * 2 + gap * 3))
    content_w = width - margin * 2
    layouts: dict[str, Rect] = {
        "header": (margin, margin, content_w, header_h),
    }

    metric_y = margin + header_h + gap
    metric_w = (content_w - gap * 3) // 4
    metric_names = [
        "metric_0",
        "metric_1",
        "metric_2",
        "metric_3",
        "metric_4",
        "metric_5",
        "metric_6",
        "metric_7",
    ]
    for idx, name in enumerate(metric_names):
        row, col = divmod(idx, 4)
        x = margin + col * (metric_w + gap)
        y = metric_y + row * (metric_h + gap)
        layouts[name] = (x, y, metric_w, metric_h)

    chart_y = metric_y + metric_h * 2 + gap * 2
    chart_w = (content_w - gap * 2) // 3
    for idx, name in enumerate(("chart_reward", "chart_loss", "chart_greedy")):
        x = margin + idx * (chart_w + gap)
        layouts[name] = (x, chart_y, chart_w, chart_h)
    return layouts


class Dashboard:
    """Pygame training dashboard with bounded charts and metric panels."""

    def __init__(self, width: int = 1100, height: int = 720):
        self.running = True
        self.disabled = False
        self.screen = None
        self.pygame = None
        self._last_state: DashboardState | None = None
        self._last_tracker: MetricsTracker | None = None
        display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if not display:
            print("[dashboard] No DISPLAY/WAYLAND_DISPLAY; GUI disabled.")
            self.disabled = True
            return
        import pygame

        self.pygame = pygame
        pygame.init()
        try:
            self.screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption("MouseMaze Training")
        except pygame.error:
            print("[dashboard] pygame display failed; GUI disabled.")
            self.disabled = True
            self.screen = None

    def draw(self, state: DashboardState, tracker: MetricsTracker) -> None:
        if self.disabled or not self.running or self.screen is None or self.pygame is None:
            return

        self._last_state = state
        self._last_tracker = tracker
        self._process_events()
        if not self.running:
            return
        self._render(state, tracker)

    def poll(self) -> None:
        """Process events and redraw cached content when hover state changes."""

        if self.disabled or not self.running or self.screen is None or self.pygame is None:
            return

        mouse_moved = self._process_events()
        if not self.running:
            return
        if mouse_moved and self._last_state is not None and self._last_tracker is not None:
            self._render(self._last_state, self._last_tracker)

    def _process_events(self) -> bool:
        pg = self.pygame
        assert pg is not None
        mouse_moved = False
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.running = False
            if event.type == pg.MOUSEMOTION:
                mouse_moved = True
        return mouse_moved

    def _render(self, state: DashboardState, tracker: MetricsTracker) -> None:
        pg = self.pygame
        assert pg is not None and self.screen is not None
        width, height = self.screen.get_size()
        rects = dashboard_layout(width, height)
        self.screen.fill((24, 28, 34))

        self._draw_header(rects["header"], state)
        metric_values = [
            ("Greedy Solve", _format_pct(state.greedy.solve_rate)),
            ("Greedy Steps", _format_number(state.greedy.avg_steps)),
            ("Optimality", _format_pct(state.greedy.optimality_ratio)),
            ("Train Solve", _format_pct(state.train_solve_rate)),
            ("Timeouts", _format_pct(state.train_timeout_rate)),
            ("Invalid Moves", _format_pct(state.train_invalid_rate)),
            ("Reward", f"{state.reward_avg:+.2f}"),
            ("Loss EMA", f"{state.loss_ema:.4f}"),
        ]
        for idx, (label, value) in enumerate(metric_values):
            self._draw_metric(rects[f"metric_{idx}"], label, value)

        self._draw_chart(
            rects["chart_reward"],
            "Reward",
            tracker.reward_history,
            (76, 201, 240),
            current_episode=state.episode,
        )
        self._draw_chart(
            rects["chart_loss"],
            "Loss EMA",
            tracker.loss_history,
            (245, 124, 87),
            current_episode=state.episode,
        )
        self._draw_chart(
            rects["chart_greedy"],
            "Greedy Solve",
            tracker.greedy_solve_history,
            (105, 214, 124),
            current_episode=state.episode,
            y_bounds=(0.0, 1.0),
            percent=True,
        )
        self._draw_hover_tooltip(rects, pg.mouse.get_pos())
        pg.display.flip()

    def _draw_panel(self, rect: Rect, fill: tuple[int, int, int] = (34, 40, 48)) -> None:
        pg = self.pygame
        assert pg is not None and self.screen is not None
        pg.draw.rect(self.screen, fill, rect, border_radius=6)
        pg.draw.rect(self.screen, (60, 68, 78), rect, width=1, border_radius=6)

    def _draw_header(self, rect: Rect, state: DashboardState) -> None:
        pg = self.pygame
        assert pg is not None and self.screen is not None
        self._draw_panel(rect, (31, 37, 45))
        x, y, _w, _h = rect
        title_font = pg.font.SysFont("arial", 28, bold=True)
        info_font = pg.font.SysFont("arial", 15)
        title = title_font.render(
            f"Greedy solve {_format_pct(state.greedy.solve_rate)}",
            True,
            (236, 240, 244),
        )
        info = info_font.render(
            (
                f"ep {state.episode} | eps {state.epsilon:.3f} | "
                f"replay {state.replay_size:,} | {state.steps_per_second:.0f} steps/s | "
                f"{state.episodes_per_second:.2f} eps/s | {state.device} | "
                f"{state.maze_size[0]}x{state.maze_size[1]} | {state.observation_mode}"
            ),
            True,
            (180, 188, 198),
        )
        self.screen.blit(title, (x + 14, y + 10))
        self.screen.blit(info, (x + 16, y + 48))

    def _draw_metric(self, rect: Rect, label: str, value: str) -> None:
        pg = self.pygame
        assert pg is not None and self.screen is not None
        self._draw_panel(rect)
        x, y, w, _h = rect
        label_font = pg.font.SysFont("arial", 14)
        value_font = pg.font.SysFont("arial", 24, bold=True)
        label_surf = label_font.render(label, True, (158, 168, 180))
        value_surf = value_font.render(value, True, (236, 240, 244))
        if value_surf.get_width() > w - 20:
            value_font = pg.font.SysFont("arial", 20, bold=True)
            value_surf = value_font.render(value, True, (236, 240, 244))
        self.screen.blit(label_surf, (x + 12, y + 9))
        self.screen.blit(value_surf, (x + 12, y + 31))

    def _draw_chart(
        self,
        rect: Rect,
        label: str,
        data: list[ChartPoint],
        color: tuple[int, int, int],
        current_episode: int,
        y_bounds: tuple[float, float] | None = None,
        percent: bool = False,
    ) -> None:
        pg = self.pygame
        assert pg is not None and self.screen is not None
        self._draw_panel(rect)
        x, y, w, h = rect
        font = pg.font.SysFont("arial", 14, bold=True)
        self.screen.blit(font.render(label, True, color), (x + 12, y + 9))
        plot = (x + 46, y + 34, w - 62, h - 74)
        pg.draw.rect(self.screen, (63, 72, 84), plot, width=1)

        values = [value for _episode, value in data]
        if y_bounds is None:
            y_min, y_max = (min(values), max(values)) if values else (0.0, 1.0)
            if y_min == y_max:
                y_min -= 1.0
                y_max += 1.0
        else:
            y_min, y_max = y_bounds

        px, py, pw, ph = plot
        x_max = _chart_x_max(current_episode, data)

        tick_font = pg.font.SysFont("arial", 11)
        for ratio, raw_value in ((0.0, y_min), (0.5, (y_min + y_max) / 2), (1.0, y_max)):
            ty = py + ph - 1 - int(ratio * (ph - 2))
            label_text = _format_pct(raw_value) if percent else _format_chart_tick(raw_value)
            tick = tick_font.render(label_text, True, (150, 160, 172))
            self.screen.blit(tick, (x + 7, ty - 7))
            pg.draw.line(self.screen, (52, 60, 70), (px, ty), (px + pw, ty))
        for episode in _episode_tick_values(x_max, pw):
            tx = px + int(episode / max(x_max, 1) * (pw - 1))
            pg.draw.line(self.screen, (52, 60, 70), (tx, py), (tx, py + ph))
            tick_label = str(episode)
            tick = tick_font.render(tick_label, True, (150, 160, 172))
            self.screen.blit(tick, (tx - tick.get_width() // 2, py + ph + 4))
        axis_label = tick_font.render("Episode", True, (150, 160, 172))
        self.screen.blit(axis_label, (px + (pw - axis_label.get_width()) // 2, y + h - 19))

        if not data:
            return
        points = []
        for episode, value in data:
            clipped = max(y_min, min(y_max, value))
            nx = px + int(episode / max(x_max, 1) * (pw - 1))
            ny = py + ph - 1 - int((clipped - y_min) / max(y_max - y_min, 1e-9) * (ph - 2))
            points.append((nx, ny))
        if len(points) == 1:
            pg.draw.circle(self.screen, color, points[0], 2)
        else:
            pg.draw.lines(self.screen, color, False, points, 2)

    def _draw_hover_tooltip(
        self,
        rects: dict[str, Rect],
        mouse_pos: tuple[int, int],
    ) -> None:
        pg = self.pygame
        assert pg is not None and self.screen is not None
        tooltip = _dashboard_tooltip_at(rects, mouse_pos)
        if tooltip is None:
            return

        font = pg.font.SysFont("arial", 13)
        lines = _wrap_tooltip_text(tooltip, font, 280)
        if not lines:
            return

        padding = 10
        line_height = font.get_linesize()
        width = max(font.size(line)[0] for line in lines) + padding * 2
        height = line_height * len(lines) + padding * 2
        screen_w, screen_h = self.screen.get_size()
        x = min(mouse_pos[0] + 16, screen_w - width - 8)
        y = min(mouse_pos[1] + 16, screen_h - height - 8)
        x = max(8, x)
        y = max(8, y)

        pg.draw.rect(self.screen, (238, 241, 245), (x, y, width, height), border_radius=5)
        pg.draw.rect(self.screen, (76, 84, 96), (x, y, width, height), width=1, border_radius=5)
        for idx, line in enumerate(lines):
            surface = font.render(line, True, (26, 31, 38))
            self.screen.blit(surface, (x + padding, y + padding + idx * line_height))

    def close(self) -> None:
        self.running = False
        if self.pygame is not None:
            self.pygame.quit()
        self.screen = None


def _format_pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def _format_number(value: float) -> str:
    return "-" if value <= 0 else f"{value:.1f}"


def _format_chart_tick(value: float) -> str:
    abs_value = abs(value)
    if abs_value == 0.0:
        return "0"
    if abs_value < 0.01:
        return f"{value:.4f}"
    if abs_value < 0.1:
        return f"{value:.3f}"
    return f"{value:.1f}"


def _chart_x_max(current_episode: int, data: list[ChartPoint]) -> int:
    latest_point = max((episode for episode, _value in data), default=0)
    return max(1, current_episode, latest_point)


def _episode_tick_values(max_episode: int, plot_width: int) -> list[int]:
    max_episode = max(1, int(max_episode))
    max_ticks = max(2, plot_width // 64)
    raw_step = max_episode / max(max_ticks - 1, 1)
    step = _nice_episode_step(raw_step)
    ticks = list(range(0, max_episode + 1, step))
    if ticks[-1] != max_episode:
        ticks.append(max_episode)
    return ticks


def _nice_episode_step(raw_step: float) -> int:
    if raw_step <= 1:
        return 1
    magnitude = 10 ** int(math.floor(math.log10(raw_step)))
    normalized = raw_step / magnitude
    if normalized <= 1:
        nice = 1
    elif normalized <= 2:
        nice = 2
    elif normalized <= 5:
        nice = 5
    else:
        nice = 10
    return max(1, int(nice * magnitude))


def _dashboard_tooltip_at(
    rects: dict[str, Rect],
    mouse_pos: tuple[int, int],
) -> str | None:
    for name, rect in rects.items():
        if name in DASHBOARD_TOOLTIPS and _point_in_rect(mouse_pos, rect):
            return DASHBOARD_TOOLTIPS[name]
    return None


def _point_in_rect(point: tuple[int, int], rect: Rect) -> bool:
    x, y = point
    rx, ry, rw, rh = rect
    return rx <= x < rx + rw and ry <= y < ry + rh


def _wrap_tooltip_text(text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        candidate = word if current == "" else f"{current} {word}"
        if current and font.size(candidate)[0] > max_width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _rects_overlap(a: Rect, b: Rect) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


# ---------------------------------------------------------------------------
# Training log
# ---------------------------------------------------------------------------
class _TrainingLogger:
    """Append-only JSONL logger for later training analysis."""

    def __init__(self, path: str | None):
        self.path = path
        self.run_id = uuid.uuid4().hex

    @property
    def enabled(self) -> bool:
        return self.path is not None

    def log(self, event: str, **payload) -> None:
        if self.path is None:
            return
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        record = {
            "schema_version": 1,
            "run_id": self.run_id,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time_unix": time.time(),
        }
        record.update(payload)
        with open(self.path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.device):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _train_config_payload(config: TrainConfig) -> dict[str, object]:
    return {field_.name: getattr(config, field_.name) for field_ in fields(config)}


def _training_environment_payload(device: torch.device) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    cuda_devices = torch.cuda.device_count() if cuda_available else 0
    gpu_name = torch.cuda.get_device_name(0) if cuda_devices > 0 else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_devices": cuda_devices,
        "selected_device": str(device),
        "gpu_name": gpu_name,
    }


def _eval_metrics_payload(metrics: EvalMetrics) -> dict[str, object]:
    return {
        "solve_rate": metrics.solve_rate,
        "solve_rate_lower_bound": metrics.solve_rate_lower_bound,
        "avg_steps": metrics.avg_steps,
        "optimality_ratio": metrics.optimality_ratio,
        "timeout_rate": metrics.timeout_rate,
        "invalid_move_rate": metrics.invalid_move_rate,
        "loop_rate": metrics.loop_rate,
        "repeated_state_action_rate": metrics.repeated_state_action_rate,
        "failed_final_distance": metrics.failed_final_distance,
        "difficulty_solve_rates": metrics.difficulty_solve_rates or {},
    }


def _training_metrics_payload(tracker: MetricsTracker) -> dict[str, float]:
    return {
        "train_solve_rate": tracker.train_solve_rate,
        "train_avg_steps": tracker.avg_steps,
        "train_avg_success_steps": tracker.avg_success_steps,
        "train_timeout_rate": tracker.train_timeout_rate,
        "train_invalid_rate": tracker.train_invalid_rate,
        "reward_avg": tracker.avg_reward,
        "loss_ema": tracker.loss_ema,
    }


def _training_speed_payload(
    completed: int,
    total_steps: int,
    start_time: float,
) -> dict[str, float]:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    return {
        "elapsed_seconds": elapsed,
        "steps_per_second": total_steps / elapsed,
        "episodes_per_second": completed / elapsed,
    }


def _runtime_utilization_payload(
    device: torch.device,
    start_time: float,
    process_start_cpu: float,
) -> dict[str, object]:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    process_cpu_seconds = max(time.process_time() - process_start_cpu, 0.0)
    payload: dict[str, object] = {
        "process_cpu_seconds": process_cpu_seconds,
        "process_cpu_percent": 100.0 * process_cpu_seconds / elapsed,
    }
    if hasattr(os, "getloadavg"):
        load_1m, load_5m, load_15m = os.getloadavg()
        payload["load_average"] = {
            "1m": load_1m,
            "5m": load_5m,
            "15m": load_15m,
        }
    if device.type == "cuda" and torch.cuda.is_available():
        payload["cuda_memory"] = {
            "allocated_bytes": torch.cuda.memory_allocated(device),
            "reserved_bytes": torch.cuda.memory_reserved(device),
            "max_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "max_reserved_bytes": torch.cuda.max_memory_reserved(device),
        }
        nvidia_smi = _nvidia_smi_utilization_payload()
        if nvidia_smi is not None:
            payload["nvidia_smi"] = nvidia_smi
    return payload


def _nvidia_smi_utilization_payload() -> list[dict[str, float | int]] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None

    gpus: list[dict[str, float | int]] = []
    for line in result.stdout.strip().splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        gpus.append(
            {
                "index": int(float(parts[0])),
                "gpu_utilization_percent": _optional_float(parts[1]),
                "memory_utilization_percent": _optional_float(parts[2]),
                "memory_used_mib": _optional_float(parts[3]),
                "memory_total_mib": _optional_float(parts[4]),
                "temperature_c": _optional_float(parts[5]),
                "power_draw_w": _optional_float(parts[6]),
            }
        )
    return gpus or None


def _optional_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return 0.0


def _training_snapshot_payload(
    completed: int,
    total_steps: int,
    epsilon: float,
    agent: Agent,
    tracker: MetricsTracker,
    greedy: EvalMetrics,
    best_eval_rate: float,
    start_time: float,
    process_start_cpu: float,
) -> dict[str, object]:
    return {
        "episode": completed,
        "total_steps": total_steps,
        "epsilon": epsilon,
        "replay_size": agent_replay_size(agent),
        "update_count": agent.update_count,
        "agent_total_env_steps": agent.total_env_steps,
        "best_greedy_solve_rate": max(best_eval_rate, 0.0),
        "metrics": _training_metrics_payload(tracker),
        "greedy": _eval_metrics_payload(greedy),
        "speed": _training_speed_payload(completed, total_steps, start_time),
        "utilization": _runtime_utilization_payload(
            agent.device,
            start_time,
            process_start_cpu,
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _merge_train_args(
    config: TrainConfig | None,
    maze_size: tuple[int, int] | None,
    episodes: int | None,
    save_path: str | None,
    dashboard_flag: bool | None,
    eval_every: int | None,
    training_log_path: str | None,
) -> TrainConfig:
    if config is None:
        config = TrainConfig()
    values = {field_.name: getattr(config, field_.name) for field_ in fields(config)}
    if maze_size is not None:
        values["maze_size"] = maze_size
    if episodes is not None:
        values["episodes"] = episodes
    if save_path is not None:
        values["save_path"] = save_path
    if training_log_path is not None:
        values["training_log_path"] = training_log_path
    if dashboard_flag is not None:
        values["dashboard_flag"] = dashboard_flag
    if eval_every is not None:
        values["eval_every"] = eval_every
        values["dashboard_every"] = eval_every
    return TrainConfig(**values)


def agent_replay_size(agent: Agent) -> int:
    return len(agent.buffer) if isinstance(agent, MouseAgent) else 0


def clone_agent_weights(agent: Agent) -> dict[str, torch.Tensor]:
    model = agent.online_net if isinstance(agent, MouseAgent) else agent.policy_net
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def restore_agent_weights(agent: Agent, weights: dict[str, torch.Tensor]) -> None:
    if isinstance(agent, MouseAgent):
        agent.online_net.load_state_dict(weights)
        agent.target_net.load_state_dict(agent.online_net.state_dict())
        agent.target_net.eval()
    else:
        agent.policy_net.load_state_dict(weights)


def create_agent(
    config: TrainConfig,
    observation_shape_: tuple[int, int, int],
    device: torch.device,
) -> Agent:
    if config.algorithm == "dqn":
        return MouseAgent(config=config, observation_shape_=observation_shape_, device=device)
    return MaskedPPOAgent(config=config, observation_shape_=observation_shape_, device=device)


def checkpoint_algorithm(path: str) -> str:
    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if isinstance(payload, dict):
        return str(payload.get("algorithm", "dqn"))
    return "dqn"


def ensure_agent_matches_config(agent: Agent, config: TrainConfig) -> None:
    if agent.algorithm != config.algorithm:
        raise ValueError(
            "agent algorithm does not match TrainConfig. "
            f"agent={agent.algorithm}, config={config.algorithm}"
        )


def _restore_rng_state(training_state: dict[str, object]) -> None:
    """Restore process RNGs from an optional training checkpoint snapshot."""

    if not training_state:
        return
    python_state = training_state.get("python_rng_state")
    numpy_state = training_state.get("numpy_rng_state")
    torch_state = training_state.get("torch_rng_state")
    cuda_states = training_state.get("cuda_rng_states")
    if python_state is not None:
        random.setstate(python_state)
    if isinstance(numpy_state, dict):
        np.random.set_state(
            (
                str(numpy_state["bit_generator"]),
                np.asarray(numpy_state["keys"], dtype=np.uint32),
                int(numpy_state["position"]),
                int(numpy_state["has_gauss"]),
                float(numpy_state["cached_gaussian"]),
            )
        )
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_states)


def _serialized_numpy_rng_state() -> dict[str, object]:
    """Return NumPy RNG state using only safe checkpoint value types."""

    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return {
        "bit_generator": bit_generator,
        "keys": keys.tolist(),
        "position": int(position),
        "has_gauss": int(has_gauss),
        "cached_gaussian": float(cached_gaussian),
    }


def _update_training_state(
    agent: Agent,
    completed: int,
    total_steps: int,
    last_eval_step: int,
    train_rng: random.Random,
    update_budget: float = 0.0,
    curriculum: CurriculumController | None = None,
) -> None:
    """Attach restart metadata immediately before a checkpoint can be saved."""

    state: dict[str, object] = {
        "completed_episodes": completed,
        "total_steps": total_steps,
        "last_eval_step": last_eval_step,
        "update_budget": update_budget,
        "python_rng_state": random.getstate(),
        "numpy_rng_state": _serialized_numpy_rng_state(),
        "torch_rng_state": torch.get_rng_state(),
        "train_rng_state": train_rng.getstate(),
    }
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    if curriculum is not None:
        state["curriculum"] = {
            "level": curriculum.level,
            "success_streak": curriculum.success_streak,
        }
    agent.training_state = state


def _maybe_run_eval(
    agent: Agent,
    config: TrainConfig,
    logger: _TrainingLogger,
    tracker: MetricsTracker,
    completed: int,
    total_steps: int,
    epsilon: float,
    best_eval_rate: float,
    best_weights: dict[str, torch.Tensor] | None,
    last_eval_step: int,
    start_time: float,
    process_start_cpu: float,
    on_evaluation: Callable[[EvalMetrics], None] | None = None,
) -> tuple[EvalMetrics | None, int, float, dict[str, torch.Tensor] | None]:
    should_eval = completed > 0 and (
        last_eval_step == 0
        or completed >= config.episodes
        or total_steps - last_eval_step >= config.eval_every_steps
    )
    if not should_eval:
        return None, last_eval_step, best_eval_rate, best_weights

    greedy = _eval_greedy(agent, config)
    tracker.record_eval(greedy, completed)
    last_eval_step = total_steps
    if on_evaluation is not None:
        on_evaluation(greedy)
    is_new_best = False
    if greedy.solve_rate > best_eval_rate:
        is_new_best = True
        best_eval_rate = greedy.solve_rate
        agent.best_greedy_solve_rate = best_eval_rate
        best_weights = clone_agent_weights(agent)
        if config.save_path:
            agent.save(config.save_path)
            print(
                "[train] new best weights saved to "
                f"{config.save_path} ({best_eval_rate:.1%})."
            )
            if logger.enabled:
                logger.log(
                    "checkpoint",
                    checkpoint_path=config.save_path,
                    reason="new_best",
                    **_training_snapshot_payload(
                        completed,
                        total_steps,
                        epsilon,
                        agent,
                        tracker,
                        greedy,
                        best_eval_rate,
                        start_time,
                        process_start_cpu,
                    ),
                )

    _print_progress(
        completed,
        config,
        tracker,
        greedy,
        best_eval_rate,
        epsilon,
        agent_replay_size(agent),
        total_steps,
        start_time,
    )
    if logger.enabled:
        logger.log(
            "eval",
            is_new_best=is_new_best,
            **_training_snapshot_payload(
                completed,
                total_steps,
                epsilon,
                agent,
                tracker,
                greedy,
                best_eval_rate,
                start_time,
                process_start_cpu,
            ),
        )
    return greedy, last_eval_step, best_eval_rate, best_weights


def _maybe_draw_dashboard(
    dashboard: Dashboard | None,
    completed: int,
    total_steps: int,
    epsilon: float,
    agent: Agent,
    config: TrainConfig,
    tracker: MetricsTracker,
    greedy: EvalMetrics,
    best_eval_rate: float,
    start_time: float,
    last_dashboard_episode: int,
) -> int:
    if dashboard is None:
        return last_dashboard_episode
    if completed - last_dashboard_episode >= config.dashboard_every:
        dashboard.draw(
            _dashboard_state(
                completed,
                total_steps,
                epsilon,
                agent,
                config,
                tracker,
                greedy,
                best_eval_rate,
                start_time,
            ),
            tracker,
        )
        return completed
    dashboard.poll()
    return last_dashboard_episode


def _finish_training(
    agent: Agent,
    config: TrainConfig,
    logger: _TrainingLogger,
    tracker: MetricsTracker,
    dashboard: Dashboard | None,
    completed: int,
    total_steps: int,
    last_eval: EvalMetrics,
    best_eval_rate: float,
    best_weights: dict[str, torch.Tensor] | None,
    start_time: float,
    process_start_cpu: float,
) -> MetricsTracker:
    if best_weights is not None:
        restore_agent_weights(agent, best_weights)
        agent.best_greedy_solve_rate = best_eval_rate
        last_eval = _eval_greedy(agent, config)
        tracker.latest_eval = last_eval
        print(f"[train] restored best greedy weights ({best_eval_rate:.1%}).")

    if config.save_path:
        agent.save(config.save_path)
        print(f"[train] weights saved to {config.save_path}")
        if logger.enabled:
            logger.log(
                "checkpoint",
                checkpoint_path=config.save_path,
                reason="final",
                **_training_snapshot_payload(
                    completed,
                    total_steps,
                    linear_epsilon(total_steps, config),
                    agent,
                    tracker,
                    last_eval,
                    best_eval_rate,
                    start_time,
                    process_start_cpu,
                ),
            )

    if dashboard is not None:
        dashboard.close()

    if logger.enabled:
        logger.log(
            "train_end",
            final_checkpoint_path=config.save_path,
            restored_best=best_weights is not None,
            **_training_snapshot_payload(
                completed,
                total_steps,
                linear_epsilon(total_steps, config),
                agent,
                tracker,
                last_eval,
                best_eval_rate,
                start_time,
                process_start_cpu,
            ),
        )

    print(
        f"[train] done | greedy_solve={tracker.latest_eval.solve_rate:.1%} | "
        f"train_solve={tracker.train_solve_rate:.1%} | "
        f"avg_reward={tracker.avg_reward:+.2f} | "
        f"loss_ema={tracker.loss_ema:.4f}"
    )
    return tracker


def train(
    agent: Agent | None = None,
    maze_size: tuple[int, int] | None = None,
    episodes: int | None = None,
    save_path: str | None = None,
    dashboard_flag: bool | None = None,
    eval_every: int | None = None,
    training_log_path: str | None = None,
    config: TrainConfig | None = None,
) -> MetricsTracker:
    """Train a MouseMaze agent and return the collected metrics."""

    config = _merge_train_args(
        config,
        maze_size,
        episodes,
        save_path,
        dashboard_flag,
        eval_every,
        training_log_path,
    )
    set_global_seed(config.seed)
    device = select_device(config.device, config.require_cuda)
    logger = _TrainingLogger(config.training_log_path)
    start_time = time.perf_counter()
    process_start_cpu = time.process_time()
    print(f"[train] {device_diagnostics(device)}")
    if config.observation_mode == "local":
        print(
            "[train] observation_mode=local is experimental for this pass; "
            "full-map observations are the reliable baseline."
        )

    expected_shape = observation_shape(config.maze_size, config.observation_mode, config.view_size)
    if agent is None:
        agent = create_agent(config, expected_shape, device)
    elif agent.observation_shape != expected_shape:
        raise ValueError(
            "agent observation shape does not match TrainConfig. "
            f"agent={agent.observation_shape}, config={expected_shape}"
        )
    else:
        ensure_agent_matches_config(agent, config)

    resumed = False
    if config.save_path and os.path.exists(config.save_path) and config.resume:
        agent.load(config.save_path)
        resumed = True
        _restore_rng_state(agent.training_state)
        print(f"[train] resumed weights from {config.save_path}")
        if logger.enabled:
            logger.log(
                "resume",
                checkpoint_path=config.save_path,
                update_count=agent.update_count,
                agent_total_env_steps=agent.total_env_steps,
                best_greedy_solve_rate=agent.best_greedy_solve_rate,
            )
    elif config.save_path and os.path.exists(config.save_path):
        print(f"[train] starting fresh; pass --resume to load {config.save_path}")

    if logger.enabled:
        logger.log(
            "train_start",
            config=_train_config_payload(config),
            environment=_training_environment_payload(device),
            checkpoint_path=config.save_path,
            resumed=resumed,
            update_count=agent.update_count,
            agent_total_env_steps=agent.total_env_steps,
            best_greedy_solve_rate=agent.best_greedy_solve_rate,
        )

    if not resumed:
        expert_loss = pretrain_with_expert(agent, config)
        if expert_loss is not None:
            print(f"[train] expert pretraining complete | loss={expert_loss:.4f}")
            if logger.enabled:
                logger.log(
                    "expert_pretrain",
                    maze_count=config.expert_pretrain_mazes,
                    epochs=config.expert_pretrain_epochs,
                    loss=expert_loss,
                )

    if config.algorithm == "dqn":
        assert isinstance(agent, MouseAgent)
        return _train_dqn(
            agent,
            config,
            logger,
            start_time,
            process_start_cpu,
        )
    assert isinstance(agent, MaskedPPOAgent)
    return _train_ppo(
        agent,
        config,
        logger,
        start_time,
        process_start_cpu,
    )


def _train_dqn(
    agent: MouseAgent,
    config: TrainConfig,
    logger: _TrainingLogger,
    start_time: float,
    process_start_cpu: float,
) -> MetricsTracker:
    """Train DQN with a vectorized collector when the task supports it."""

    if config.vectorized_envs and config.observation_mode == "full":
        return _train_dqn_vectorized(
            agent,
            config,
            logger,
            start_time,
            process_start_cpu,
        )
    return _train_dqn_sequential(
        agent,
        config,
        logger,
        start_time,
        process_start_cpu,
    )


def _train_dqn_sequential(
    agent: MouseAgent,
    config: TrainConfig,
    logger: _TrainingLogger,
    start_time: float,
    process_start_cpu: float,
    initial_envs: list[object] | None = None,
) -> MetricsTracker:
    tracker = MetricsTracker()
    dashboard = Dashboard() if config.dashboard_flag else None
    best_eval_rate = agent.best_greedy_solve_rate
    best_weights = clone_agent_weights(agent) if best_eval_rate >= 0.0 else None

    saved_state = agent.training_state
    completed = min(int(saved_state.get("completed_episodes", 0)), config.episodes)
    total_steps = max(int(saved_state.get("total_steps", 0)), agent.total_env_steps)
    last_eval_step = int(saved_state.get("last_eval_step", 0))
    update_budget = float(saved_state.get("update_budget", 0.0))
    env_count = min(config.num_envs, max(config.episodes - completed, 1))
    train_rng = random.Random(config.seed)
    if "train_rng_state" in saved_state:
        train_rng.setstate(saved_state["train_rng_state"])
    sampler = MazeTaskSampler(config, train_rng)
    curriculum_state = saved_state.get("curriculum")
    if isinstance(curriculum_state, dict):
        sampler.curriculum.level = int(curriculum_state.get("level", 0))
        sampler.curriculum.success_streak = int(curriculum_state.get("success_streak", 0))
    envs = initial_envs or [
        make_training_maze(config, train_rng, completed + 1, sampler=sampler)
        for _ in range(env_count)
    ]
    states = [env.reset() for env in envs]
    n_step_queues: list[deque[RawTransition]] = [deque() for _ in envs]

    last_eval = EvalMetrics()
    last_dashboard_episode = -config.dashboard_every

    while completed < config.episodes:
        epsilon = linear_epsilon(total_steps, config)
        state_batch = np.stack(states)
        action_masks = np.stack([env.valid_action_mask() for env in envs])
        actions = agent.get_actions(
            state_batch,
            epsilon=epsilon,
            action_masks=action_masks,
        )

        collected_steps = 0
        for idx, env in enumerate(envs):
            if completed >= config.episodes:
                break
            state = states[idx]
            next_state, reward, done, _info = env.step(int(actions[idx]))
            next_action_mask = env.valid_action_mask()
            n_step_queues[idx].append(
                (
                    state,
                    int(actions[idx]),
                    reward,
                    next_state,
                    done,
                    next_action_mask,
                )
            )
            store_ready_n_step_transition(agent, n_step_queues[idx], config)
            total_steps += 1
            agent.total_env_steps += 1
            collected_steps += 1
            states[idx] = next_state

            if done:
                flush_n_step_transitions(agent, n_step_queues[idx], config)
                completed += 1
                tracker.record_episode(episode_stats_from_env(env), completed)
                if completed < config.episodes:
                    envs[idx] = make_training_maze(
                        config,
                        train_rng,
                        completed + 1,
                        sampler=sampler,
                    )
                    states[idx] = envs[idx].reset()

        if total_steps >= config.warmup_steps:
            update_budget += collected_steps * config.updates_per_transition
            while update_budget >= 1.0:
                tracker.record_loss(agent.train_step(), completed)
                update_budget -= 1.0

        def record_evaluation(metrics: EvalMetrics) -> None:
            sampler.curriculum.record_validation(metrics)
            _update_training_state(
                agent,
                completed,
                total_steps,
                total_steps,
                train_rng,
                update_budget,
                sampler.curriculum,
            )

        eval_result, last_eval_step, best_eval_rate, best_weights = _maybe_run_eval(
            agent,
            config,
            logger,
            tracker,
            completed,
            total_steps,
            epsilon,
            best_eval_rate,
            best_weights,
            last_eval_step,
            start_time,
            process_start_cpu,
            on_evaluation=record_evaluation,
        )
        if eval_result is not None:
            last_eval = eval_result

        last_dashboard_episode = _maybe_draw_dashboard(
            dashboard,
            completed,
            total_steps,
            epsilon,
            agent,
            config,
            tracker,
            last_eval,
            best_eval_rate,
            start_time,
            last_dashboard_episode,
        )

    _update_training_state(
        agent,
        completed,
        total_steps,
        last_eval_step,
        train_rng,
        update_budget,
        sampler.curriculum,
    )
    return _finish_training(
        agent,
        config,
        logger,
        tracker,
        dashboard,
        completed,
        total_steps,
        last_eval,
        best_eval_rate,
        best_weights,
        start_time,
        process_start_cpu,
    )


def _train_dqn_vectorized(
    agent: MouseAgent,
    config: TrainConfig,
    logger: _TrainingLogger,
    start_time: float,
    process_start_cpu: float,
) -> MetricsTracker:
    """Train DQN with batched full-map Maze transitions and transition clocks."""

    tracker = MetricsTracker()
    dashboard = Dashboard() if config.dashboard_flag else None
    best_eval_rate = agent.best_greedy_solve_rate
    best_weights = clone_agent_weights(agent) if best_eval_rate >= 0.0 else None
    saved_state = agent.training_state
    completed = min(int(saved_state.get("completed_episodes", 0)), config.episodes)
    total_steps = max(int(saved_state.get("total_steps", 0)), agent.total_env_steps)
    last_eval_step = int(saved_state.get("last_eval_step", 0))
    update_budget = float(saved_state.get("update_budget", 0.0))
    train_rng = random.Random(config.seed)
    if "train_rng_state" in saved_state:
        train_rng.setstate(saved_state["train_rng_state"])
    sampler = MazeTaskSampler(config, train_rng)
    curriculum_state = saved_state.get("curriculum")
    if isinstance(curriculum_state, dict):
        sampler.curriculum.level = int(curriculum_state.get("level", 0))
        sampler.curriculum.success_streak = int(curriculum_state.get("success_streak", 0))

    remaining_episodes = max(config.episodes - completed, 1)
    env_count = min(config.num_envs, remaining_episodes)
    environments = [
        make_training_maze(config, train_rng, completed + 1, sampler=sampler)
        for _ in range(env_count)
    ]
    if not all(isinstance(environment, Maze) for environment in environments):
        return _train_dqn_sequential(
            agent,
            config,
            logger,
            start_time,
            process_start_cpu,
            initial_envs=environments,
        )
    environment_batch = MazeBatch(environments)
    states = environment_batch.observations()
    n_step_queues: list[deque[RawTransition]] = [deque() for _ in range(env_count)]
    last_eval = EvalMetrics()
    last_dashboard_episode = -config.dashboard_every

    while completed < config.episodes:
        epsilon = linear_epsilon(total_steps, config)
        action_masks = environment_batch.valid_action_masks()
        actions = agent.get_actions(states, epsilon=epsilon, action_masks=action_masks)
        step_result = environment_batch.step(actions)
        next_states = step_result.states

        for index in range(env_count):
            n_step_queues[index].append(
                (
                    states[index],
                    int(actions[index]),
                    float(step_result.rewards[index]),
                    next_states[index],
                    bool(step_result.dones[index]),
                    step_result.action_masks[index],
                )
            )
            store_ready_n_step_transition(agent, n_step_queues[index], config)
            if not step_result.dones[index]:
                continue
            flush_n_step_transitions(agent, n_step_queues[index], config)
            if completed >= config.episodes:
                continue
            completed += 1
            tracker.record_episode(environment_batch.episode_stats(index), completed)
            if completed < config.episodes:
                replacement = make_training_maze(
                    config,
                    train_rng,
                    completed + 1,
                    sampler=sampler,
                )
                environment_batch.replace(index, replacement)

        collected_steps = env_count
        total_steps += collected_steps
        agent.total_env_steps += collected_steps
        if total_steps >= config.warmup_steps:
            update_budget += collected_steps * config.updates_per_transition
            while update_budget >= 1.0:
                tracker.record_loss(agent.train_step(), completed)
                update_budget -= 1.0

        states = environment_batch.observations()

        def record_evaluation(metrics: EvalMetrics) -> None:
            sampler.curriculum.record_validation(metrics)
            _update_training_state(
                agent,
                completed,
                total_steps,
                total_steps,
                train_rng,
                update_budget,
                sampler.curriculum,
            )

        eval_result, last_eval_step, best_eval_rate, best_weights = _maybe_run_eval(
            agent,
            config,
            logger,
            tracker,
            completed,
            total_steps,
            epsilon,
            best_eval_rate,
            best_weights,
            last_eval_step,
            start_time,
            process_start_cpu,
            on_evaluation=record_evaluation,
        )
        if eval_result is not None:
            last_eval = eval_result

        last_dashboard_episode = _maybe_draw_dashboard(
            dashboard,
            completed,
            total_steps,
            epsilon,
            agent,
            config,
            tracker,
            last_eval,
            best_eval_rate,
            start_time,
            last_dashboard_episode,
        )

    _update_training_state(
        agent,
        completed,
        total_steps,
        last_eval_step,
        train_rng,
        update_budget,
        sampler.curriculum,
    )
    return _finish_training(
        agent,
        config,
        logger,
        tracker,
        dashboard,
        completed,
        total_steps,
        last_eval,
        best_eval_rate,
        best_weights,
        start_time,
        process_start_cpu,
    )


def _compute_gae(
    rewards: np.ndarray,
    dones: np.ndarray,
    values: np.ndarray,
    next_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_advantage = np.zeros(rewards.shape[1], dtype=np.float32)
    for step in range(rewards.shape[0] - 1, -1, -1):
        if step == rewards.shape[0] - 1:
            next_value = next_values
        else:
            next_value = values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[step] = last_advantage
    returns = advantages + values
    return advantages, returns


def _ppo_update(
    agent: MaskedPPOAgent,
    config: TrainConfig,
    states: np.ndarray,
    action_masks: np.ndarray,
    actions: np.ndarray,
    old_log_probs: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
) -> float:
    flat_count = states.shape[0]
    advantages = advantages.astype(np.float32)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    states_t = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
    masks_t = torch.as_tensor(action_masks, dtype=torch.bool, device=agent.device)
    actions_t = torch.as_tensor(actions, dtype=torch.long, device=agent.device)
    old_log_probs_t = torch.as_tensor(
        old_log_probs,
        dtype=torch.float32,
        device=agent.device,
    )
    advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=agent.device)
    returns_t = torch.as_tensor(returns, dtype=torch.float32, device=agent.device)
    batch_size = min(config.batch_size, flat_count)
    losses: list[float] = []

    for _epoch in range(config.ppo_epochs):
        indices = torch.randperm(flat_count, device=agent.device)
        for start in range(0, flat_count, batch_size):
            batch_idx = indices[start : start + batch_size]
            log_probs, entropy, values = agent.evaluate_actions(
                states_t[batch_idx],
                masks_t[batch_idx],
                actions_t[batch_idx],
            )
            ratio = torch.exp(log_probs - old_log_probs_t[batch_idx])
            unclipped = ratio * advantages_t[batch_idx]
            clipped = torch.clamp(
                ratio,
                1.0 - config.ppo_clip_range,
                1.0 + config.ppo_clip_range,
            ) * advantages_t[batch_idx]
            policy_loss = -torch.min(unclipped, clipped).mean()
            value_loss = nn.functional.mse_loss(values, returns_t[batch_idx])
            entropy_bonus = entropy.mean()
            loss = (
                policy_loss
                + config.ppo_value_coef * value_loss
                - config.ppo_entropy_coef * entropy_bonus
            )
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy_net.parameters(), config.ppo_max_grad_norm)
            agent.optimizer.step()
            agent.update_count += 1
            losses.append(float(loss.item()))
    return sum(losses) / len(losses) if losses else 0.0


def _train_ppo(
    agent: MaskedPPOAgent,
    config: TrainConfig,
    logger: _TrainingLogger,
    start_time: float,
    process_start_cpu: float,
) -> MetricsTracker:
    tracker = MetricsTracker()
    dashboard = Dashboard() if config.dashboard_flag else None
    best_eval_rate = agent.best_greedy_solve_rate
    best_weights = clone_agent_weights(agent) if best_eval_rate >= 0.0 else None

    saved_state = agent.training_state
    completed = min(int(saved_state.get("completed_episodes", 0)), config.episodes)
    total_steps = max(int(saved_state.get("total_steps", 0)), agent.total_env_steps)
    last_eval_step = int(saved_state.get("last_eval_step", 0))
    env_count = min(config.num_envs, max(config.episodes - completed, 1))
    train_rng = random.Random(config.seed)
    if "train_rng_state" in saved_state:
        train_rng.setstate(saved_state["train_rng_state"])
    sampler = MazeTaskSampler(config, train_rng)
    curriculum_state = saved_state.get("curriculum")
    if isinstance(curriculum_state, dict):
        sampler.curriculum.level = int(curriculum_state.get("level", 0))
        sampler.curriculum.success_streak = int(curriculum_state.get("success_streak", 0))
    envs = [
        make_training_maze(config, train_rng, completed + 1, sampler=sampler)
        for _ in range(env_count)
    ]
    states = [env.reset() for env in envs]

    last_eval = EvalMetrics()
    last_dashboard_episode = -config.dashboard_every

    while completed < config.episodes:
        rollout_states: list[np.ndarray] = []
        rollout_masks: list[np.ndarray] = []
        rollout_actions: list[np.ndarray] = []
        rollout_log_probs: list[np.ndarray] = []
        rollout_values: list[np.ndarray] = []
        rollout_rewards: list[np.ndarray] = []
        rollout_dones: list[np.ndarray] = []

        for _step in range(config.ppo_rollout_steps):
            if completed >= config.episodes:
                break
            state_batch = np.stack(states)
            action_masks = np.stack([env.valid_action_mask() for env in envs])
            actions, log_probs, values = agent.sample_actions(state_batch, action_masks)
            rewards = np.zeros(env_count, dtype=np.float32)
            dones = np.zeros(env_count, dtype=np.float32)

            rollout_states.append(state_batch)
            rollout_masks.append(action_masks)
            rollout_actions.append(actions)
            rollout_log_probs.append(log_probs)
            rollout_values.append(values)

            for idx, env in enumerate(envs):
                next_state, reward, done, _info = env.step(int(actions[idx]))
                rewards[idx] = reward
                dones[idx] = float(done)
                total_steps += 1
                agent.total_env_steps += 1
                states[idx] = next_state

                if done and completed < config.episodes:
                    completed += 1
                    tracker.record_episode(episode_stats_from_env(env), completed)
                    if completed < config.episodes:
                        envs[idx] = make_training_maze(
                            config,
                            train_rng,
                            completed + 1,
                            sampler=sampler,
                        )
                        states[idx] = envs[idx].reset()

            rollout_rewards.append(rewards)
            rollout_dones.append(dones)

        if not rollout_states:
            break

        next_state_batch = np.stack(states)
        next_masks = np.stack([env.valid_action_mask() for env in envs])
        with torch.no_grad():
            _logits, next_values_t = agent._logits_and_values(next_state_batch, next_masks)
        next_values = next_values_t.cpu().numpy().astype(np.float32)

        rewards_arr = np.stack(rollout_rewards)
        dones_arr = np.stack(rollout_dones)
        values_arr = np.stack(rollout_values)
        advantages, returns = _compute_gae(
            rewards_arr,
            dones_arr,
            values_arr,
            next_values,
            config.gamma,
            config.ppo_gae_lambda,
        )

        loss = _ppo_update(
            agent,
            config,
            np.concatenate(rollout_states, axis=0),
            np.concatenate(rollout_masks, axis=0),
            np.concatenate(rollout_actions, axis=0),
            np.concatenate(rollout_log_probs, axis=0),
            advantages.reshape(-1),
            returns.reshape(-1),
        )
        tracker.record_loss(loss, completed)

        epsilon = 0.0
        def record_evaluation(metrics: EvalMetrics) -> None:
            sampler.curriculum.record_validation(metrics)
            _update_training_state(
                agent,
                completed,
                total_steps,
                total_steps,
                train_rng,
                curriculum=sampler.curriculum,
            )

        eval_result, last_eval_step, best_eval_rate, best_weights = _maybe_run_eval(
            agent,
            config,
            logger,
            tracker,
            completed,
            total_steps,
            epsilon,
            best_eval_rate,
            best_weights,
            last_eval_step,
            start_time,
            process_start_cpu,
            on_evaluation=record_evaluation,
        )
        if eval_result is not None:
            last_eval = eval_result

        last_dashboard_episode = _maybe_draw_dashboard(
            dashboard,
            completed,
            total_steps,
            epsilon,
            agent,
            config,
            tracker,
            last_eval,
            best_eval_rate,
            start_time,
            last_dashboard_episode,
        )

    _update_training_state(
        agent,
        completed,
        total_steps,
        last_eval_step,
        train_rng,
        curriculum=sampler.curriculum,
    )
    return _finish_training(
        agent,
        config,
        logger,
        tracker,
        dashboard,
        completed,
        total_steps,
        last_eval,
        best_eval_rate,
        best_weights,
        start_time,
        process_start_cpu,
    )


def _dashboard_state(
    completed: int,
    total_steps: int,
    epsilon: float,
    agent: Agent,
    config: TrainConfig,
    tracker: MetricsTracker,
    greedy: EvalMetrics,
    best_eval_rate: float,
    start_time: float,
) -> DashboardState:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    return DashboardState(
        episode=completed,
        total_steps=total_steps,
        epsilon=epsilon,
        replay_size=agent_replay_size(agent),
        device=str(agent.device),
        maze_size=config.maze_size,
        observation_mode=config.observation_mode,
        train_solve_rate=tracker.train_solve_rate,
        train_avg_steps=tracker.avg_steps,
        train_timeout_rate=tracker.train_timeout_rate,
        train_invalid_rate=tracker.train_invalid_rate,
        reward_avg=tracker.avg_reward,
        loss_ema=tracker.loss_ema,
        greedy=greedy,
        best_greedy_solve_rate=max(best_eval_rate, 0.0),
        steps_per_second=total_steps / elapsed,
        episodes_per_second=completed / elapsed,
    )


def _print_progress(
    completed: int,
    config: TrainConfig,
    tracker: MetricsTracker,
    greedy: EvalMetrics,
    best_eval_rate: float,
    epsilon: float,
    replay_size: int,
    total_steps: int,
    start_time: float,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    print(
        f"Ep {completed:6d}/{config.episodes:<6d} "
        f"| greedy {greedy.solve_rate:5.1%} "
        f"| train {tracker.train_solve_rate:5.1%} "
        f"| avg_steps {greedy.avg_steps:5.1f} "
        f"| optimal {greedy.optimality_ratio:5.1%} "
        f"| loops {greedy.loop_rate:5.1%} "
        f"| invalid {tracker.train_invalid_rate:5.1%} "
        f"| reward {tracker.avg_reward:+7.2f} "
        f"| loss {tracker.loss_ema:7.4f} "
        f"| eps {epsilon:.3f} "
        f"| best {max(best_eval_rate, 0.0):5.1%} "
        f"| replay {replay_size:>7d} "
        f"| {total_steps / elapsed:6.0f} steps/s"
    )


# ---------------------------------------------------------------------------
# Inference visualization
# ---------------------------------------------------------------------------
def _draw_mouse_icon(pg, screen, center_x: int, center_y: int, cell_size: int) -> None:
    cx, cy = center_x + cell_size // 2, center_y + cell_size // 2
    body_r = cell_size // 4
    head_r = cell_size // 5
    pg.draw.circle(screen, (160, 160, 160), (cx, cy + 2), body_r)
    pg.draw.circle(screen, (160, 160, 160), (cx, cy - body_r + 2), head_r)
    ear_r = max(2, cell_size // 8)
    pg.draw.circle(screen, (220, 180, 180), (cx - head_r // 2, cy - body_r), ear_r)
    pg.draw.circle(screen, (220, 180, 180), (cx + head_r // 2, cy - body_r), ear_r)
    eye_r = max(1, cell_size // 16)
    pg.draw.circle(screen, (0, 0, 0), (cx - 3, cy - body_r), eye_r)
    pg.draw.circle(screen, (0, 0, 0), (cx + 3, cy - body_r), eye_r)
    tail_points = [
        (cx + body_r, cy + 2),
        (cx + body_r + cell_size // 4, cy - cell_size // 6),
        (cx + body_r + cell_size // 3, cy - cell_size // 3),
    ]
    pg.draw.lines(screen, (220, 180, 180), False, tail_points, 2)


def _draw_cheese_icon(pg, screen, center_x: int, center_y: int, cell_size: int) -> None:
    cx, cy = center_x + cell_size // 2, center_y + cell_size // 2
    half = cell_size // 3
    points = [(cx, cy - half), (cx - half, cy + half // 2), (cx + half, cy + half // 2)]
    pg.draw.polygon(screen, (255, 200, 0), points)
    hole_r = max(1, cell_size // 10)
    pg.draw.circle(screen, (200, 150, 0), (cx - 4, cy - 2), hole_r)
    pg.draw.circle(screen, (200, 150, 0), (cx + 4, cy + 2), hole_r)
    pg.draw.circle(screen, (200, 150, 0), (cx, cy + 5), max(1, hole_r // 2))


def visualize_inference(
    agent: Agent | Planner,
    maze_grid: np.ndarray,
    fps: int = 5,
    observation_mode: str | None = None,
    config: TrainConfig | None = None,
) -> None:
    """Render learned-policy or planner inference on a single maze."""

    import pygame

    agent_config = config or getattr(agent, "config", TrainConfig())
    mode = observation_mode or agent_config.observation_mode
    env = Maze(
        maze_grid.copy(),
        observation_mode=mode,
        view_size=getattr(agent, "view_size", agent_config.view_size),
        max_episode_steps=agent_config.max_episode_steps,
        timeout_step_factor=agent_config.timeout_step_factor,
        min_episode_steps=agent_config.min_episode_steps,
    )

    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.init()
    info = pygame.display.Info()
    rows, cols = env.grid.shape
    hud_h = 42
    raw_cell_size = (info.current_h // 2 - hud_h) // max(rows, cols)
    cell_size = max(4, min(raw_cell_size, 80))
    while rows * cell_size + hud_h > 860 and cell_size > 4:
        cell_size -= 1
    window_w = cols * cell_size
    window_h = rows * cell_size + hud_h
    screen = pygame.display.set_mode((window_w, window_h))
    pygame.display.set_caption("MouseMaze Inference")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("arial", 16)

    trail = [env.start]
    state = env.reset()
    action_mask = env.valid_action_mask()
    action, q_vals = _inference_action(agent, state, action_mask)
    done = False
    last_blocked = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        screen.fill((247, 248, 245))
        for r in range(rows):
            for c in range(cols):
                if env.grid[r, c] == 1:
                    pygame.draw.rect(
                        screen,
                        (25, 28, 32),
                        (c * cell_size, r * cell_size, cell_size, cell_size),
                    )

        pygame.draw.rect(
            screen,
            (142, 214, 128),
            (env.start[1] * cell_size, env.start[0] * cell_size, cell_size, cell_size),
        )
        _draw_cheese_icon(
            pygame,
            screen,
            env.goal[1] * cell_size,
            env.goal[0] * cell_size,
            cell_size,
        )

        inset = max(2, cell_size // 5)
        for r, c in trail[:-1]:
            pygame.draw.rect(
                screen,
                (76, 128, 224),
                (
                    c * cell_size + inset,
                    r * cell_size + inset,
                    cell_size - 2 * inset,
                    cell_size - 2 * inset,
                ),
                1,
            )

        _draw_mouse_icon(
            pygame,
            screen,
            env.current_position[1] * cell_size,
            env.current_position[0] * cell_size,
            cell_size,
        )

        hud_y = rows * cell_size
        pygame.draw.rect(screen, (226, 230, 235), (0, hud_y, window_w, hud_h))
        action_name, arrow = Maze.ACTION_NAMES[action]
        blocked = " blocked" if last_blocked else ""
        if q_vals is None:
            hud = f"Steps {env.steps:>3} | Planner {action_name} {arrow}{blocked} | {mode}"
        else:
            masked_q_vals = q_vals.copy()
            masked_q_vals[~action_mask] = -np.inf
            best_action = int(np.argmax(masked_q_vals))
            best_name, best_arrow = Maze.ACTION_NAMES[best_action]
            hud = (
                f"Steps {env.steps:>3} | Last {action_name} {arrow}{blocked} | "
                f"Q-max {best_name} {best_arrow} ({q_vals[best_action]:.2f}) | {mode}"
            )
        screen.blit(font.render(hud, True, (30, 35, 42)), (8, hud_y + 11))
        pygame.display.flip()
        clock.tick(fps)

        if done:
            break

        trail.append(env.current_position)
        next_state, _reward, done, step_info = env.step(action)
        last_blocked = not bool(step_info["moved"])
        state = next_state
        action_mask = env.valid_action_mask()
        action, q_vals = _inference_action(agent, state, action_mask)

    label = "SOLVED" if env.current_position == env.goal else "TIMEOUT"
    print(f"Steps: {env.steps} -- {label}")
    pygame.quit()


def _inference_action(
    policy: Agent | Planner,
    state: np.ndarray,
    action_mask: np.ndarray,
) -> tuple[int, np.ndarray | None]:
    """Choose a legal action and optionally return learned action values."""

    q_values = getattr(policy, "q_values", None)
    if callable(q_values):
        values = q_values(state)
        masked_values = values.copy()
        masked_values[~action_mask] = -np.inf
        return int(np.argmax(masked_values)), values
    actions = policy.get_actions(state, action_masks=action_mask)
    return int(actions[0]), None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and run MouseMaze DQN.")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument(
        "--maze-size",
        type=int,
        nargs=2,
        metavar=("ROWS", "COLS"),
        default=None,
    )
    parser.add_argument("--save-path", default=None)
    parser.add_argument("--training-log-path", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default=None)
    parser.add_argument("--network-type", choices=NETWORK_TYPES, default=None)
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--observation-mode", choices=OBSERVATION_MODES, default=None)
    parser.add_argument("--view-size", type=int, default=None)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--timeout-step-factor", type=float, default=None)
    parser.add_argument("--min-episode-steps", type=int, default=None)
    parser.add_argument("--step-penalty", type=float, default=None)
    parser.add_argument("--invalid-move-penalty", type=float, default=None)
    parser.add_argument("--goal-reward", type=float, default=None)
    parser.add_argument("--timeout-penalty", type=float, default=None)
    parser.add_argument("--distance-shaping-scale", type=float, default=None)
    parser.add_argument(
        "--distance-shaping-mode",
        choices=DISTANCE_SHAPING_MODES,
        default=None,
    )
    parser.add_argument(
        "--curriculum",
        dest="curriculum_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--curriculum-easy-episodes", type=int, default=None)
    parser.add_argument("--curriculum-medium-episodes", type=int, default=None)
    parser.add_argument(
        "--curriculum-easy-range",
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument(
        "--curriculum-medium-range",
        type=int,
        nargs=2,
        metavar=("LOW", "HIGH"),
        default=None,
    )
    parser.add_argument("--curriculum-max-retries", type=int, default=None)
    parser.add_argument("--buffer-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--min-replay-size", type=int, default=None)
    parser.add_argument("--target-update-freq", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--n-step-returns", type=int, default=None)
    parser.add_argument("--epsilon-start", type=float, default=None)
    parser.add_argument("--epsilon-mid", type=float, default=None)
    parser.add_argument("--epsilon-end", type=float, default=None)
    parser.add_argument("--epsilon-decay-episodes", type=int, default=None)
    parser.add_argument("--epsilon-decay-steps", type=int, default=None)
    parser.add_argument("--epsilon-final-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--train-updates-per-step", type=int, default=None)
    parser.add_argument("--updates-per-transition", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--target-tau", type=float, default=None)
    parser.add_argument(
        "--prioritized-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--priority-alpha", type=float, default=None)
    parser.add_argument("--priority-beta-start", type=float, default=None)
    parser.add_argument("--priority-beta-steps", type=int, default=None)
    parser.add_argument("--ppo-rollout-steps", type=int, default=None)
    parser.add_argument("--ppo-epochs", type=int, default=None)
    parser.add_argument("--ppo-clip-range", type=float, default=None)
    parser.add_argument("--ppo-gae-lambda", type=float, default=None)
    parser.add_argument("--ppo-value-coef", type=float, default=None)
    parser.add_argument("--ppo-entropy-coef", type=float, default=None)
    parser.add_argument("--ppo-max-grad-norm", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--dashboard-every", type=int, default=None)
    parser.add_argument("--curriculum-promotion-rate", type=float, default=None)
    parser.add_argument("--curriculum-promotion-evals", type=int, default=None)
    parser.add_argument("--curriculum-previous-fraction", type=float, default=None)
    parser.add_argument("--curriculum-uniform-fraction", type=float, default=None)
    parser.add_argument(
        "--vectorized-envs",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--checkpoint-replay",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--expert-pretrain-mazes", type=int, default=None)
    parser.add_argument("--expert-pretrain-epochs", type=int, default=None)
    parser.add_argument(
        "--planner-fallback",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use the exact full-map planner for inference instead of the learned policy.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Evaluate the saved policy on disjoint deterministic suites and exit.",
    )
    parser.add_argument("--benchmark-episodes", type=int, default=2_000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default=None)
    parser.add_argument(
        "--require-cuda",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--dashboard",
        dest="dashboard_flag",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--train",
        dest="train_flag",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--infer",
        dest="infer_flag",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    return parser.parse_args(argv)


def _train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    config = TrainConfig()
    values = {field_.name: getattr(config, field_.name) for field_ in fields(config)}
    if args.maze_size is not None:
        values["maze_size"] = tuple(args.maze_size)
    if args.curriculum_easy_range is not None:
        values["curriculum_easy_range"] = tuple(args.curriculum_easy_range)
    if args.curriculum_medium_range is not None:
        values["curriculum_medium_range"] = tuple(args.curriculum_medium_range)
    for name in (
        "episodes",
        "seed",
        "eval_seed",
        "algorithm",
        "network_type",
        "resume",
        "observation_mode",
        "view_size",
        "max_episode_steps",
        "timeout_step_factor",
        "min_episode_steps",
        "step_penalty",
        "invalid_move_penalty",
        "goal_reward",
        "timeout_penalty",
        "distance_shaping_scale",
        "distance_shaping_mode",
        "curriculum_enabled",
        "curriculum_easy_episodes",
        "curriculum_medium_episodes",
        "curriculum_max_retries",
        "buffer_size",
        "batch_size",
        "min_replay_size",
        "target_update_freq",
        "learning_rate",
        "gamma",
        "n_step_returns",
        "epsilon_start",
        "epsilon_mid",
        "epsilon_end",
        "epsilon_decay_episodes",
        "epsilon_decay_steps",
        "epsilon_final_steps",
        "num_envs",
        "train_updates_per_step",
        "updates_per_transition",
        "warmup_steps",
        "target_tau",
        "prioritized_replay",
        "priority_alpha",
        "priority_beta_start",
        "priority_beta_steps",
        "ppo_rollout_steps",
        "ppo_epochs",
        "ppo_clip_range",
        "ppo_gae_lambda",
        "ppo_value_coef",
        "ppo_entropy_coef",
        "ppo_max_grad_norm",
        "eval_episodes",
        "eval_every",
        "eval_every_steps",
        "dashboard_every",
        "curriculum_promotion_rate",
        "curriculum_promotion_evals",
        "curriculum_previous_fraction",
        "curriculum_uniform_fraction",
        "vectorized_envs",
        "checkpoint_replay",
        "expert_pretrain_mazes",
        "expert_pretrain_epochs",
        "planner_fallback",
        "dashboard_flag",
        "save_path",
        "training_log_path",
        "device",
        "require_cuda",
    ):
        value = getattr(args, name)
        if value is not None:
            values[name] = value
    if args.eval_every is not None and args.dashboard_every is None:
        values["dashboard_every"] = args.eval_every
    return TrainConfig(**values)


def _optional_bool(value: bool | None, default: bool) -> bool:
    return default if value is None else value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _train_config_from_args(args)
    device = select_device(config.device, config.require_cuda)
    expected_shape = observation_shape(config.maze_size, config.observation_mode, config.view_size)
    agent = create_agent(config, expected_shape, device)

    if args.benchmark:
        if not config.save_path or not os.path.exists(config.save_path):
            raise FileNotFoundError("--benchmark requires an existing --save-path checkpoint")
        checkpoint_algo = checkpoint_algorithm(config.save_path)
        if checkpoint_algo != agent.algorithm:
            values = {field_.name: getattr(config, field_.name) for field_ in fields(config)}
            values["algorithm"] = checkpoint_algo
            config = TrainConfig(**values)
            agent = create_agent(config, expected_shape, device)
        agent.load(config.save_path)
        result = run_benchmark(agent, config, episodes=args.benchmark_episodes)
        print(
            json.dumps(
                {
                    "validation": _eval_metrics_payload(result.validation),
                    "final_test": _eval_metrics_payload(result.final_test),
                    "stress_test": _eval_metrics_payload(result.stress_test),
                },
                sort_keys=True,
            )
        )
        return

    if _optional_bool(args.train_flag, DEFAULT_TRAIN_FLAG):
        train(agent=agent, config=config)
    elif config.save_path and os.path.exists(config.save_path):
        checkpoint_algo = checkpoint_algorithm(config.save_path)
        if checkpoint_algo != agent.algorithm:
            values = {field_.name: getattr(config, field_.name) for field_ in fields(config)}
            values["algorithm"] = checkpoint_algo
            config = TrainConfig(**values)
            agent = create_agent(config, expected_shape, device)
        agent.load(config.save_path)

    if _optional_bool(args.infer_flag, DEFAULT_INFER_FLAG):
        maze_grid = generate_random_maze(config.maze_size[0], config.maze_size[1])
        print("Inference on a fresh maze:")
        inference_policy: Agent | Planner = BfsPlanner() if config.planner_fallback else agent
        visualize_inference(
            inference_policy,
            maze_grid.copy(),
            observation_mode=config.observation_mode,
            config=config,
        )


if __name__ == "__main__":
    main()
