from __future__ import annotations

import argparse
import copy
import contextlib
import importlib.metadata
import json
import math
import multiprocessing
import os
import platform
import random
import socket
import subprocess
import sys
import time
import uuid
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field, fields, replace
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
# Size of the centered local observation window; it must remain odd.
VIEW_SIZE = 7
# Rows and columns used when no maze size is supplied on the command line.
DEFAULT_MAZE_SIZE = (11, 11)
# Observation choices: the full map or a centered local window.
OBSERVATION_MODES = ("full", "local")
# Algorithms supported by the trainer and checkpoint loader.
ALGORITHMS = ("dqn", "ppo", "recurrent_ppo")
# Q-network layouts: spatial convolutions or a flattened MLP input.
NETWORK_TYPES = ("spatial", "flat")
# Reward-shaping choices for distance-to-goal progress.
DISTANCE_SHAPING_MODES = ("potential", "fractional", "none")
# Hardware/reproducibility tuning profiles exposed by the CLI.
PERFORMANCE_PROFILES = ("auto", "rtx3090-fast", "strict", "portable")
# Checkpoint format version written by the current agent implementation.
CHECKPOINT_SCHEMA_VERSION = 2
# Number of training episodes when training is enabled without an override.
DEFAULT_EPISODES = 1_000_000
# Seed used for training maze generation and other random operations.
DEFAULT_SEED = 0
# Observation representation used by default for training and inference.
DEFAULT_OBSERVATION_MODE = "local"
# Agent algorithm selected when no checkpoint metadata or CLI override exists.
DEFAULT_ALGORITHM = "recurrent_ppo"
# Network layout used by default for algorithms that support both layouts.
DEFAULT_NETWORK_TYPE = "spatial"
# Whether training resumes from the checkpoint instead of starting fresh.
DEFAULT_RESUME = True
# Whether recurrent PPO ignores episode and transition caps until it reaches its target.
DEFAULT_TARGET_ONLY_STOP: bool | None = None

# Maximum number of transitions retained by the DQN replay buffer.
BUFFER_SIZE = 200_000
# Number of replay transitions sampled for each DQN update.
BATCH_SIZE = 512
# Replay size required before DQN updates begin.
MIN_REPLAY_SIZE = 2_000
# Number of environment steps between DQN target-network updates.
TARGET_UPDATE_FREQ = 2_000
# Optimizer learning rate shared by the trainable agents.
LEARNING_RATE = 3e-4
# Discount factor applied to future rewards.
GAMMA = 0.99
# Number of steps included in each DQN bootstrapped return.
N_STEP_RETURNS = 3
# Initial DQN epsilon for exploratory action selection.
EPSILON_START = 1.0
# Intermediate DQN epsilon after the first exploration-decay phase.
EPSILON_MID = 0.10
# Final minimum DQN epsilon after exploration decay completes.
EPSILON_END = 0.02
# Transition at which epsilon reaches EPSILON_MID.
EPSILON_DECAY_STEPS = 500_000
# Transition at which epsilon reaches EPSILON_END.
EPSILON_FINAL_STEPS = 1_500_000
# Fallback number of parallel environments used by vectorized collection.
DEFAULT_NUM_ENVS = 256
# Recurrent environments used by the RTX 3090 fast profile when auto-sized.
RTX3090_RECURRENT_NUM_ENVS = 512
# Desired learner updates per collected transition for DQN.
DEFAULT_UPDATES_PER_TRANSITION = 0.125
# Transitions collected before the learner begins updating.
DEFAULT_WARMUP_STEPS = 10_000
# Polyak averaging factor used for soft target-network updates.
DEFAULT_TARGET_TAU = 0.005
# Whether DQN samples replay transitions by priority.
DEFAULT_PRIORITIZED_REPLAY = True
# Exponent controlling how strongly replay priorities affect sampling.
DEFAULT_PRIORITY_ALPHA = 0.6
# Initial importance-sampling correction applied to prioritized replay.
DEFAULT_PRIORITY_BETA_START = 0.4
# Number of transitions over which replay correction reaches full strength.
DEFAULT_PRIORITY_BETA_STEPS = 1_500_000
# Hard cap on steps in one maze episode.
MAX_EPISODE_STEPS = 300
# Multiplier used to provide a maze-size-aware recovery budget when enabled.
DEFAULT_TIMEOUT_STEP_FACTOR = 4.0
# Minimum episode length allowing local agents to recover from short wrong turns.
DEFAULT_MIN_EPISODE_STEPS = 20
# Number of episodes in each standard evaluation pass during training.
NUM_EVAL_EPISODES = 500
# Episode-based dashboard/evaluation cadence compatibility default.
EVAL_PERIOD = 100
# Transition interval between evaluation passes during training.
EVAL_PERIOD_STEPS = 50_000
# Offset keeping evaluation maze seeds separate from training seeds.
EVAL_SEED_OFFSET = 1_000_003
# Whether the live training dashboard is enabled by default.
DEFAULT_DASHBOARD_FLAG = True
# Number of episodes between dashboard refreshes.
DEFAULT_DASHBOARD_EVERY = EVAL_PERIOD
# Maximum representative samples retained by each dashboard chart.
DASHBOARD_MAX_HISTORY_POINTS = 2_048
# Checkpoint path before the CLI assigns an invocation-specific default.
DEFAULT_SAVE_PATH = None
# Training-log path before the CLI assigns an invocation-specific default.
DEFAULT_TRAINING_LOG_PATH = None
# Device selection mode; auto chooses CUDA when it is available.
DEFAULT_DEVICE = "auto"
# Whether a missing CUDA device should be treated as an error in auto mode.
DEFAULT_REQUIRE_CUDA = True
# Training is opt-in so launching the script does not alter a checkpoint.
DEFAULT_TRAIN_FLAG = True
# Inference is enabled by default after optional training/checkpoint loading.
DEFAULT_INFER_FLAG = False
# Number of fresh mazes rendered for inference; zero means run indefinitely.
DEFAULT_INFERENCE_MAZES = 0

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, "models")
LOG_RESULTS_DIR = os.path.join(RESULTS_DIR, "logs")
ARTIFACT_BASENAME = "mousemaze"

# Small per-step reward encouraging shorter solutions.
STEP_PENALTY = -0.05
# Additional penalty for attempting to move into a wall or outside the maze.
INVALID_MOVE_PENALTY = -0.20
# Reward granted when the mouse reaches the cheese.
GOAL_REWARD = 10.0
# Penalty applied when an episode times out.
TIMEOUT_PENALTY = -2.0
# Multiplier applied to the selected distance-shaping reward.
DISTANCE_SHAPING_SCALE = 1.0
# Distance-shaping strategy used unless local observations disable it.
DEFAULT_DISTANCE_SHAPING_MODE = "potential"

# Whether curriculum sampling changes maze difficulty during training.
DEFAULT_CURRICULUM_ENABLED = True
# Goal-distance range used for the easiest curriculum stage.
DEFAULT_CURRICULUM_EASY_RANGE = (6, 10)
# Goal-distance range used for the medium curriculum stage.
DEFAULT_CURRICULUM_MEDIUM_RANGE = (8, 16)
# Episode cutoffs retained for callers of the legacy episode-indexed helper.
LEGACY_CURRICULUM_EASY_EPISODES = 5_000
LEGACY_CURRICULUM_MEDIUM_EPISODES = 15_000
# Maximum attempts to generate a maze matching a requested difficulty.
DEFAULT_CURRICULUM_MAX_RETRIES = 200
# Validation solve rate required before promoting curriculum difficulty.
DEFAULT_CURRICULUM_PROMOTION_RATE = 0.90
# Number of validation evaluations required for a curriculum promotion.
DEFAULT_CURRICULUM_PROMOTION_EVALS = 3
# Fraction of curriculum tasks sampled from previously successful stages.
DEFAULT_CURRICULUM_PREVIOUS_FRACTION = 0.20
# Fraction of curriculum tasks sampled uniformly across available stages.
DEFAULT_CURRICULUM_UNIFORM_FRACTION = 0.10
# Fraction of training tasks drawn from transformed held-out failures.
DEFAULT_HARD_MAZE_FRACTION = 0.0
# Maximum number of transformed hard-maze variants retained in memory.
DEFAULT_HARD_MAZE_POOL_SIZE = 256
# Validation range over which hard replay ramps to its configured maximum.
HARD_MAZE_RAMP_START = 0.90
HARD_MAZE_RAMP_END = 0.99
# Whether training collects transitions from multiple environments together.
DEFAULT_VECTORIZED_ENVS = True
# Whether replay contents are included in checkpoints.
DEFAULT_CHECKPOINT_REPLAY = False
# Number of mazes used for optional expert-policy pretraining.
DEFAULT_EXPERT_PRETRAIN_MAZES = 0
# Number of passes through the expert-pretraining maze set.
DEFAULT_EXPERT_PRETRAIN_EPOCHS = 3
# Whether inference may use the exact full-map BFS planner as a fallback.
DEFAULT_PLANNER_FALLBACK = False
# Hardware profile selected when none is specified.
DEFAULT_PERFORMANCE_PROFILE = "auto"
# Default transition budget for full-map training.
DEFAULT_FULL_MAX_ENV_STEPS = 3_000_000
# Default transition budget for local-observation training.
DEFAULT_LOCAL_MAX_ENV_STEPS = 5_000_000
# Target validation solve rate for full-map training.
DEFAULT_FULL_TARGET_SOLVE_RATE = 1.00
# Target validation solve rate for local-observation training.
DEFAULT_LOCAL_TARGET_SOLVE_RATE = 1.00
# Number of deterministic suites used to confirm one frozen target candidate.
DEFAULT_TARGET_SOLVE_EVALS = 3
# Post-budget evaluations without improvement before a guarded recovery pass.
DEFAULT_PRECISION_PLATEAU_EVALS = 20
# Transitions in one guarded precision-recovery pass.
DEFAULT_PRECISION_RECOVERY_STEPS = 1_000_000
# Fraction of the initial learning rate used during precision recovery.
DEFAULT_PRECISION_RECOVERY_LR_FRACTION = 0.05
# Whether rollback-based post-budget precision recovery is active.
DEFAULT_PRECISION_RECOVERY_ENABLED = False
# Transition cadence for the resumable latest-training sidecar.
DEFAULT_LATEST_CHECKPOINT_EVERY_STEPS = 1_000_000
# Number of mazes used to evaluate a curriculum stage.
DEFAULT_CURRICULUM_EVAL_EPISODES = 200
# Worker processes used for deterministic background maze generation.
DEFAULT_MAZE_WORKERS = 8

# Number of transitions collected in one PPO rollout.
PPO_ROLLOUT_STEPS = 128
# Number of optimization passes over each PPO rollout.
PPO_EPOCHS = 4
# PPO policy clipping threshold.
PPO_CLIP_RANGE = 0.2
# GAE weighting of bias versus variance in PPO advantages.
PPO_GAE_LAMBDA = 0.95
# Weight of the PPO value-function loss.
PPO_VALUE_COEF = 0.5
# Weight of PPO policy entropy regularization.
PPO_ENTROPY_COEF = 0.01
# Maximum gradient norm used for PPO clipping.
PPO_MAX_GRAD_NORM = 0.5
# Approximate-KL threshold that can stop a PPO update early.
PPO_TARGET_KL = 0.02
# Clipping threshold for PPO value predictions.
PPO_VALUE_CLIP_RANGE = 0.2
# Size of the recurrent GRU hidden state.
RECURRENT_HIDDEN_SIZE = 128
# Number of consecutive steps in a recurrent PPO training sequence.
RECURRENT_SEQUENCE_LENGTH = 64
# Number of recurrent sequences processed in one PPO minibatch.
RECURRENT_SEQUENCE_MINIBATCH_SIZE = 32
# Coefficient for random-network-distillation intrinsic reward.
RND_REWARD_COEF = 0.10
# Maximum absolute intrinsic reward contribution from RND.
RND_REWARD_CLIP = 5.0

# Module-level device hint retained for callers that import this module.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass(slots=True)
class TrainConfig:
    """Training and evaluation settings for MouseMaze."""

    maze_size: tuple[int, int] = DEFAULT_MAZE_SIZE
    episodes: int = DEFAULT_EPISODES
    max_env_steps: int | None = None
    seed: int = DEFAULT_SEED
    eval_seed: int | None = None
    algorithm: str = DEFAULT_ALGORITHM
    network_type: str = DEFAULT_NETWORK_TYPE
    resume: bool = DEFAULT_RESUME
    target_only_stop: bool | None = DEFAULT_TARGET_ONLY_STOP
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
    epsilon_decay_steps: int = EPSILON_DECAY_STEPS
    epsilon_final_steps: int = EPSILON_FINAL_STEPS
    num_envs: int | None = None
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
    ppo_target_kl: float = PPO_TARGET_KL
    ppo_value_clip_range: float = PPO_VALUE_CLIP_RANGE
    recurrent_hidden_size: int = RECURRENT_HIDDEN_SIZE
    recurrent_sequence_length: int = RECURRENT_SEQUENCE_LENGTH
    recurrent_sequence_minibatch_size: int = RECURRENT_SEQUENCE_MINIBATCH_SIZE
    rnd_reward_coef: float = RND_REWARD_COEF
    rnd_reward_clip: float = RND_REWARD_CLIP
    eval_every_steps: int = EVAL_PERIOD_STEPS
    eval_episodes: int = NUM_EVAL_EPISODES
    curriculum_promotion_rate: float = DEFAULT_CURRICULUM_PROMOTION_RATE
    curriculum_promotion_evals: int = DEFAULT_CURRICULUM_PROMOTION_EVALS
    curriculum_previous_fraction: float = DEFAULT_CURRICULUM_PREVIOUS_FRACTION
    curriculum_uniform_fraction: float = DEFAULT_CURRICULUM_UNIFORM_FRACTION
    hard_maze_fraction: float = DEFAULT_HARD_MAZE_FRACTION
    hard_maze_pool_size: int = DEFAULT_HARD_MAZE_POOL_SIZE
    curriculum_eval_episodes: int = DEFAULT_CURRICULUM_EVAL_EPISODES
    vectorized_envs: bool = DEFAULT_VECTORIZED_ENVS
    checkpoint_replay: bool = DEFAULT_CHECKPOINT_REPLAY
    expert_pretrain_mazes: int = DEFAULT_EXPERT_PRETRAIN_MAZES
    expert_pretrain_epochs: int = DEFAULT_EXPERT_PRETRAIN_EPOCHS
    planner_fallback: bool = DEFAULT_PLANNER_FALLBACK
    target_solve_rate: float | None = None
    target_solve_evals: int = DEFAULT_TARGET_SOLVE_EVALS
    precision_plateau_evals: int = DEFAULT_PRECISION_PLATEAU_EVALS
    precision_recovery_steps: int = DEFAULT_PRECISION_RECOVERY_STEPS
    precision_recovery_lr_fraction: float = DEFAULT_PRECISION_RECOVERY_LR_FRACTION
    precision_recovery_enabled: bool = DEFAULT_PRECISION_RECOVERY_ENABLED
    latest_checkpoint_every_steps: int = DEFAULT_LATEST_CHECKPOINT_EVERY_STEPS
    performance_profile: str = DEFAULT_PERFORMANCE_PROFILE
    maze_workers: int = DEFAULT_MAZE_WORKERS
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
        if self.target_only_stop is None:
            self.target_only_stop = self.algorithm == "recurrent_ppo"
        elif self.target_only_stop and self.algorithm != "recurrent_ppo":
            raise ValueError("target_only_stop is supported only by recurrent_ppo")
        if self.network_type not in NETWORK_TYPES:
            raise ValueError(
                f"network_type must be one of {NETWORK_TYPES}; got {self.network_type!r}"
            )
        if self.observation_mode not in OBSERVATION_MODES:
            raise ValueError(
                f"observation_mode must be one of {OBSERVATION_MODES}; "
                f"got {self.observation_mode!r}"
            )
        if self.max_env_steps is None:
            self.max_env_steps = (
                DEFAULT_LOCAL_MAX_ENV_STEPS
                if self.observation_mode == "local"
                else DEFAULT_FULL_MAX_ENV_STEPS
            )
        if self.target_solve_rate is None:
            self.target_solve_rate = (
                DEFAULT_LOCAL_TARGET_SOLVE_RATE
                if self.observation_mode == "local"
                else DEFAULT_FULL_TARGET_SOLVE_RATE
            )
        if self.observation_mode == "local":
            self.distance_shaping_mode = "none"
        if self.performance_profile not in PERFORMANCE_PROFILES:
            raise ValueError(
                f"performance_profile must be one of {PERFORMANCE_PROFILES}; "
                f"got {self.performance_profile!r}"
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
        if self.max_env_steps < 1:
            raise ValueError("max_env_steps must be >= 1")
        if self.num_envs is not None and self.num_envs < 1:
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
        if not 0 <= self.hard_maze_fraction <= 1:
            raise ValueError("hard_maze_fraction must be in [0, 1]")
        if self.hard_maze_pool_size < 1:
            raise ValueError("hard_maze_pool_size must be >= 1")
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
        if self.ppo_target_kl <= 0 or self.ppo_value_clip_range <= 0:
            raise ValueError("PPO target KL and value clip range must be positive")
        if self.recurrent_hidden_size < 1 or self.recurrent_sequence_length < 1:
            raise ValueError("recurrent dimensions must be positive")
        if (
            self.algorithm == "recurrent_ppo"
            and self.ppo_rollout_steps % self.recurrent_sequence_length != 0
        ):
            raise ValueError("ppo_rollout_steps must be divisible by recurrent_sequence_length")
        if self.recurrent_sequence_minibatch_size < 1:
            raise ValueError("recurrent_sequence_minibatch_size must be >= 1")
        if self.rnd_reward_coef < 0 or self.rnd_reward_clip <= 0:
            raise ValueError("RND coefficient must be non-negative and clip must be positive")
        if self.eval_every_steps < 1:
            raise ValueError("eval_every_steps must be >= 1")
        if self.eval_episodes < 1:
            raise ValueError("eval_episodes must be >= 1")
        if self.curriculum_eval_episodes < 1:
            raise ValueError("curriculum_eval_episodes must be >= 1")
        if not 0 < self.target_solve_rate <= 1 or self.target_solve_evals < 1:
            raise ValueError("target solve settings are invalid")
        if self.precision_plateau_evals < 1 or self.precision_recovery_steps < 1:
            raise ValueError("precision recovery intervals must be >= 1")
        if not 0 < self.precision_recovery_lr_fraction <= 1:
            raise ValueError("precision_recovery_lr_fraction must be in (0, 1]")
        if self.latest_checkpoint_every_steps < 1:
            raise ValueError("latest_checkpoint_every_steps must be >= 1")
        if self.maze_workers < 0:
            raise ValueError("maze_workers must be >= 0")
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
    difficulty_counts: dict[str, int] | None = None
    failed_grids: list[np.ndarray] = field(default_factory=list, repr=False)


@dataclass(slots=True)
class PPOUpdateMetrics:
    """Aggregated recurrent PPO learner diagnostics."""

    loss: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    entropy: float = 0.0
    approx_kl: float = 0.0
    epochs: int = 0


@dataclass(slots=True)
class TargetConfirmationResult:
    """Results from deterministic validation of one frozen policy candidate."""

    confirmed: bool
    suites: list[tuple[int, EvalMetrics]]


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


def _append_chart_point(history: list[ChartPoint], point: ChartPoint) -> None:
    """Append one point while retaining a bounded whole-run representation."""

    history.append(point)
    if len(history) <= DASHBOARD_MAX_HISTORY_POINTS:
        return
    latest = history[-1]
    history[:] = history[::2]
    if history[-1] != latest:
        history.append(latest)


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


def resolved_performance_profile(profile: str, device: torch.device) -> str:
    """Resolve automatic hardware tuning without changing algorithm semantics."""

    if profile != "auto":
        return profile
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        if properties.major == 8 and properties.minor == 6 and properties.total_memory >= 20 << 30:
            return "rtx3090-fast"
    return "portable"


def configure_performance(profile: str, device: torch.device) -> str:
    """Configure PyTorch for a fast seeded or strict reproducibility profile."""

    resolved = resolved_performance_profile(profile, device)
    try:
        torch.set_num_threads(
            4 if resolved == "rtx3090-fast" else max(1, min(8, os.cpu_count() or 1))
        )
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if resolved == "strict":
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    elif resolved == "rtx3090-fast" and device.type == "cuda":
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    return resolved


def resolve_num_envs(config: TrainConfig) -> int:
    """Resolve profile-selected environment parallelism when left automatic."""

    if config.num_envs is not None:
        return config.num_envs
    if (
        config.algorithm == "recurrent_ppo"
        and config.performance_profile == "rtx3090-fast"
    ):
        return RTX3090_RECURRENT_NUM_ENVS
    return DEFAULT_NUM_ENVS


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
        return (4, maze_size[0], maze_size[1])
    if observation_mode == "local":
        return (4, view_size, view_size)
    raise ValueError(f"unsupported observation_mode: {observation_mode!r}")


def resolved_eval_seed(config: TrainConfig) -> int:
    """Return the deterministic evaluation seed for a training config."""

    if config.eval_seed is not None:
        return config.eval_seed
    return config.seed + EVAL_SEED_OFFSET


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------
class SumTree:
    """Vectorized proportional-priority tree with O(log N) sampling and updates."""

    def __init__(self, capacity: int):
        leaf_count = 1
        while leaf_count < capacity:
            leaf_count *= 2
        self.capacity = capacity
        self.leaf_count = leaf_count
        self.values = np.zeros(2 * leaf_count, dtype=np.float64)

    @property
    def total(self) -> float:
        return float(self.values[1])

    def rebuild(self, priorities: np.ndarray, alpha: float) -> None:
        self.values.fill(0.0)
        count = len(priorities)
        self.values[self.leaf_count : self.leaf_count + count] = np.maximum(
            priorities,
            1e-6,
        ) ** alpha
        for level_start in range(self.leaf_count - 1, 0, -1):
            self.values[level_start] = (
                self.values[2 * level_start] + self.values[2 * level_start + 1]
            )

    def update(self, indices: np.ndarray, values: np.ndarray) -> None:
        nodes = np.asarray(indices, dtype=np.int64) + self.leaf_count
        self.values[nodes] = np.asarray(values, dtype=np.float64)
        nodes = np.unique(nodes // 2)
        while nodes.size and nodes[0] > 0:
            self.values[nodes] = self.values[2 * nodes] + self.values[2 * nodes + 1]
            nodes = np.unique(nodes // 2)

    def sample(self, masses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        nodes = np.ones(len(masses), dtype=np.int64)
        remaining = np.asarray(masses, dtype=np.float64).copy()
        while nodes[0] < self.leaf_count:
            left = nodes * 2
            left_values = self.values[left]
            go_right = remaining >= left_values
            remaining -= go_right * left_values
            nodes = left + go_right.astype(np.int64)
        indices = nodes - self.leaf_count
        return indices, self.values[nodes]


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
        "_sum_tree",
        "_tree_alpha",
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
        self._sum_tree = SumTree(capacity)
        self._tree_alpha: float | None = None

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
        if self._tree_alpha is not None:
            self._sum_tree.update(
                np.array([idx]),
                np.array([self.priorities[idx] ** self._tree_alpha]),
            )
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
            if self._tree_alpha != alpha:
                self._sum_tree.rebuild(self.priorities[: self._size], alpha)
                self._tree_alpha = alpha
            total = self._sum_tree.total
            segment = total / n
            masses = (np.arange(n) + self._rng.random(n)) * segment
            idx, sampled_values = self._sum_tree.sample(masses)
            probabilities = sampled_values / total
            weights = (self._size * probabilities) ** (-beta)
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
        if self._tree_alpha is not None:
            self._sum_tree.update(
                index_array,
                self.priorities[index_array] ** self._tree_alpha,
            )

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
        self._tree_alpha = None

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
    Observations contain walls, agent, goal, and remaining-time channels.
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
        obs = np.zeros((4, *self.grid.shape), dtype=np.float32)
        obs[0] = self.grid == 1
        obs[1, self.current_position[0], self.current_position[1]] = 1.0
        obs[2, self.goal[0], self.goal[1]] = 1.0
        obs[3].fill(self.remaining_time_fraction)
        return obs

    def _local_observation(self, position: tuple[int, int]) -> np.ndarray:
        half = self.view_size // 2
        obs = np.zeros((4, self.view_size, self.view_size), dtype=np.float32)
        obs[1, half, half] = 1.0
        obs[3].fill(self.remaining_time_fraction)
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

    @property
    def remaining_time_fraction(self) -> float:
        """Return the observable fraction of the episode budget remaining."""

        return max(self.max_episode_steps - self.steps, 0) / max(self.max_episode_steps, 1)

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

    if not config.curriculum_enabled or episode is None:
        return None
    scale = max(config.maze_size) / max(DEFAULT_MAZE_SIZE)
    if episode <= LEGACY_CURRICULUM_EASY_EPISODES:
        return _scale_distance_range(config.curriculum_easy_range, scale)
    if episode <= LEGACY_CURRICULUM_MEDIUM_EPISODES:
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
        self.hard_grids: list[np.ndarray] = []
        self._hard_grid_keys: set[bytes] = set()
        self._seen_hard_grid_keys: set[bytes] = set()
        self.hard_candidates_seen = 0
        self.validation_solve_rate: float | None = None

    def sample(self) -> Maze:
        """Sample from current, previous, and uniform task distributions."""

        hard_grid = self.sample_hard_grid()
        if hard_grid is not None:
            return _maze_from_grid(self.config, hard_grid)
        target_range = self.sample_target_range()
        try:
            return make_maze(self.config, rng=self.rng, target_range=target_range)
        except TypeError as exc:
            if "target_range" not in str(exc):
                raise
            return make_maze(self.config, rng=self.rng)

    def add_failed_grids(self, failed_grids: list[np.ndarray]) -> int:
        """Reservoir-sample unique transformed failures without held-out grids."""

        added = 0
        for failed_grid in failed_grids:
            for variant in _hard_maze_variants(failed_grid):
                key = variant.tobytes()
                if key in self._seen_hard_grid_keys:
                    continue
                self._seen_hard_grid_keys.add(key)
                self.hard_candidates_seen += 1
                if len(self.hard_grids) < self.config.hard_maze_pool_size:
                    self.hard_grids.append(variant)
                    self._hard_grid_keys.add(key)
                    added += 1
                    continue
                replacement = self.rng.randrange(self.hard_candidates_seen)
                if replacement >= self.config.hard_maze_pool_size:
                    continue
                removed = self.hard_grids[replacement]
                self._hard_grid_keys.discard(removed.tobytes())
                self.hard_grids[replacement] = variant
                self._hard_grid_keys.add(key)
                added += 1
        return added

    def restore_hard_grids(
        self,
        grids: object,
        seen_keys: object = None,
        candidates_seen: object = None,
        validation_solve_rate: object = None,
    ) -> None:
        """Restore replay reservoir state from current or legacy checkpoints."""

        grid_values = grids if isinstance(grids, (list, tuple)) else ()
        for value in grid_values[-self.config.hard_maze_pool_size :]:
            grid = np.ascontiguousarray(value, dtype=np.uint8)
            if grid.shape != self.config.maze_size:
                continue
            key = grid.tobytes()
            if key in self._hard_grid_keys:
                continue
            self.hard_grids.append(grid)
            self._hard_grid_keys.add(key)
        if isinstance(seen_keys, (list, tuple, set)):
            self._seen_hard_grid_keys.update(
                bytes(key) for key in seen_keys if isinstance(key, (bytes, bytearray))
            )
        self._seen_hard_grid_keys.update(self._hard_grid_keys)
        try:
            restored_seen = int(candidates_seen)
        except (TypeError, ValueError):
            restored_seen = len(self._seen_hard_grid_keys)
        self.hard_candidates_seen = max(
            restored_seen,
            len(self._seen_hard_grid_keys),
            len(self.hard_grids),
        )
        if isinstance(validation_solve_rate, (int, float)):
            self.validation_solve_rate = min(
                max(float(validation_solve_rate), 0.0),
                1.0,
            )

    def record_validation_solve_rate(self, solve_rate: float) -> None:
        """Update the validation signal controlling the hard-replay ramp."""

        self.validation_solve_rate = min(max(float(solve_rate), 0.0), 1.0)

    def effective_hard_maze_fraction(self) -> float:
        """Return the validation-ramped fraction of hard replay tasks."""

        if not self.hard_grids or self.validation_solve_rate is None:
            return 0.0
        progress = (self.validation_solve_rate - HARD_MAZE_RAMP_START) / (
            HARD_MAZE_RAMP_END - HARD_MAZE_RAMP_START
        )
        return self.config.hard_maze_fraction * min(max(progress, 0.0), 1.0)

    def sample_hard_grid(self) -> np.ndarray | None:
        """Return a hard-maze variant according to the configured replay mix."""

        hard_fraction = self.effective_hard_maze_fraction()
        if hard_fraction == 0.0 or self.rng.random() >= hard_fraction:
            return None
        return self.rng.choice(self.hard_grids).copy()

    def sample_target_range(self) -> tuple[int, int] | None:
        """Choose a curriculum range without turning a missing previous stage into uniform."""

        target_range = self.curriculum.target_range()
        if target_range is None:
            return None
        draw = self.rng.random()
        if draw < self.config.curriculum_uniform_fraction:
            return None
        previous = self.curriculum.previous_range()
        if previous is not None and draw < (
            self.config.curriculum_uniform_fraction
            + self.config.curriculum_previous_fraction
        ):
            return previous
        return target_range

    def sampling_mix(self) -> dict[str, float]:
        """Return the effective current/previous/unrestricted sampling proportions."""

        hard = self.effective_hard_maze_fraction()
        ordinary = 1.0 - hard
        if self.curriculum.complete or not self.config.curriculum_enabled:
            return {
                "hard": hard,
                "current": 0.0,
                "previous": 0.0,
                "unrestricted": ordinary,
            }
        previous = (
            self.config.curriculum_previous_fraction
            if self.curriculum.previous_range() is not None
            else 0.0
        )
        unrestricted = self.config.curriculum_uniform_fraction
        return {
            "hard": hard,
            "current": ordinary * (1.0 - previous - unrestricted),
            "previous": ordinary * previous,
            "unrestricted": ordinary * unrestricted,
        }

    def describe(self, environment: Maze) -> dict[str, float | str]:
        """Describe a task using reusable difficulty metadata."""

        distance = environment.optimal_start_steps
        return {
            "difficulty": float(distance),
            "bucket": difficulty_bucket(distance),
        }


def _hard_maze_variants(grid: np.ndarray) -> list[np.ndarray]:
    """Return shape-preserving symmetries, excluding the held-out original."""

    original = np.asarray(grid, dtype=np.uint8)
    candidates = [
        np.flip(original, axis=0),
        np.flip(original, axis=1),
        np.rot90(original, 2),
    ]
    if original.shape[0] == original.shape[1]:
        candidates.extend((np.rot90(original, 1), np.rot90(original, 3)))
    original_key = original.tobytes()
    variants: list[np.ndarray] = []
    keys: set[bytes] = set()
    for candidate in candidates:
        variant = np.ascontiguousarray(candidate, dtype=np.uint8)
        key = variant.tobytes()
        if key == original_key or key in keys:
            continue
        variants.append(variant)
        keys.add(key)
    return variants


def _generate_prefetched_grid(
    config: TrainConfig,
    seed: int,
    target_range: tuple[int, int] | None,
) -> np.ndarray:
    """Generate one deterministically seeded task in a worker process."""

    return make_maze(
        config,
        rng=random.Random(seed),
        target_range=target_range,
    ).grid


class DeterministicMazePrefetcher:
    """Ordered process-backed maze generation with reproducible task seeds."""

    def __init__(self, sampler: MazeTaskSampler, workers: int):
        self.sampler = sampler
        self.workers = workers
        self.executor: ProcessPoolExecutor | None = None
        self.futures: deque[Future[np.ndarray]] = deque()
        if workers > 0:
            self.executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            self._fill()

    def _fill(self) -> None:
        if self.executor is None:
            return
        while len(self.futures) < self.workers * 2:
            target_range = self.sampler.sample_target_range()
            seed = self.sampler.rng.randrange(0, 2**63)
            self.futures.append(
                self.executor.submit(
                    _generate_prefetched_grid,
                    self.sampler.config,
                    seed,
                    target_range,
                )
            )

    def next(self) -> Maze:
        if self.executor is None:
            return self.sampler.sample()
        hard_grid = self.sampler.sample_hard_grid()
        if hard_grid is not None:
            return _maze_from_grid(self.sampler.config, hard_grid)
        grid = self.futures.popleft().result()
        self._fill()
        return _maze_from_grid(self.sampler.config, grid)

    def reset(self) -> None:
        """Discard queued tasks after curriculum promotion."""

        if self.executor is None:
            return
        for future in self.futures:
            future.cancel()
        self.futures.clear()
        self._fill()

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.futures.clear()


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
    """Generate a maze that can be completed within its episode budget.

    The generator produces connected mazes, but a connected maze can still
    have a shortest path longer than the effective episode limit.  Such a
    candidate must be discarded before it reaches training or inference.
    """

    if target_range is None:
        target_range = curriculum_distance_range(config, episode)

    for _ in range(config.curriculum_max_retries):
        grid = generate_random_maze(config.maze_size[0], config.maze_size[1], rng=rng)
        env = _maze_from_grid(config, grid)
        if env.optimal_start_steps > env.max_episode_steps:
            continue
        if target_range is None:
            return env
        low, high = target_range
        if low <= env.optimal_start_steps <= high:
            return env

    if target_range is not None:
        # Preserve the previous behavior of relaxing an unavailable
        # curriculum distance range, while retaining the episode-budget
        # guarantee for the fallback maze.
        for _ in range(config.curriculum_max_retries):
            grid = generate_random_maze(config.maze_size[0], config.maze_size[1], rng=rng)
            env = _maze_from_grid(config, grid)
            if env.optimal_start_steps <= env.max_episode_steps:
                return env

    raise RuntimeError(
        "could not generate a maze solvable within the effective episode "
        f"limit after {config.curriculum_max_retries} attempts"
    )


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
    """Vectorized Maze adapter used by DQN and PPO collection.

    Individual ``Maze`` objects remain the source of task construction and
    rendering. This adapter only batches their fixed-shape transition logic.
    """

    def __init__(self, environments: list[Maze]):
        if not environments:
            raise ValueError("MazeBatch requires at least one environment")
        first = environments[0]
        grid_shape = first.grid.shape
        if any(
            env.observation_mode != first.observation_mode
            or env.view_size != first.view_size
            or env.grid.shape != grid_shape
            for env in environments
        ):
            raise ValueError("all batched mazes must use one observation and grid shape")
        self.size = len(environments)
        self.grid_shape = grid_shape
        self.observation_mode = first.observation_mode
        self.view_size = first.view_size
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

        if (
            environment.observation_mode != self.observation_mode
            or environment.view_size != self.view_size
            or environment.grid.shape != self.grid_shape
        ):
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
        """Build batched full or centered local observations."""

        if self.observation_mode == "local":
            return self._local_observations()
        observations = np.zeros((self.size, 4, *self.grid_shape), dtype=np.float32)
        observations[:, 0] = self.grids == 1
        indices = np.arange(self.size)
        observations[indices, 1, self.positions[:, 0], self.positions[:, 1]] = 1.0
        observations[indices, 2, self.goals[:, 0], self.goals[:, 1]] = 1.0
        remaining = np.maximum(self.max_episode_steps - self.steps, 0) / np.maximum(
            self.max_episode_steps,
            1,
        )
        observations[:, 3] = remaining[:, np.newaxis, np.newaxis]
        return observations

    def _local_observations(self) -> np.ndarray:
        half = self.view_size // 2
        offsets = np.arange(-half, half + 1, dtype=np.int64)
        rows = self.positions[:, 0, np.newaxis, np.newaxis] + offsets[np.newaxis, :, np.newaxis]
        cols = self.positions[:, 1, np.newaxis, np.newaxis] + offsets[np.newaxis, np.newaxis, :]
        rows = np.broadcast_to(rows, (self.size, self.view_size, self.view_size))
        cols = np.broadcast_to(cols, (self.size, self.view_size, self.view_size))
        in_bounds = (
            (rows >= 0)
            & (rows < self.grid_shape[0])
            & (cols >= 0)
            & (cols < self.grid_shape[1])
        )
        clipped_rows = np.clip(rows, 0, self.grid_shape[0] - 1)
        clipped_cols = np.clip(cols, 0, self.grid_shape[1] - 1)
        batch_indices = np.arange(self.size)[:, np.newaxis, np.newaxis]
        cells = self.grids[batch_indices, clipped_rows, clipped_cols]
        observations = np.zeros(
            (self.size, 4, self.view_size, self.view_size),
            dtype=np.float32,
        )
        observations[:, 0] = (~in_bounds) | (cells == 1)
        observations[:, 1, half, half] = 1.0
        observations[:, 2] = in_bounds & (
            (rows == self.goals[:, 0, np.newaxis, np.newaxis])
            & (cols == self.goals[:, 1, np.newaxis, np.newaxis])
        )
        remaining = np.maximum(self.max_episode_steps - self.steps, 0) / np.maximum(
            self.max_episode_steps,
            1,
        )
        observations[:, 3] = remaining[:, np.newaxis, np.newaxis]
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

    def step(
        self,
        actions: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> BatchStep:
        """Apply actions to all or a selected subset of environments."""

        action_array = np.asarray(actions, dtype=np.int64)
        all_selected = indices is None
        selected = (
            np.arange(self.size, dtype=np.int64)
            if all_selected
            else np.asarray(indices, dtype=np.int64)
        )
        if selected.ndim != 1 or np.any((selected < 0) | (selected >= self.size)):
            raise ValueError("indices must be a one-dimensional in-range array")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("indices must not contain duplicates")
        if action_array.shape != (len(selected),):
            raise ValueError(
                f"actions must have shape ({len(selected)},), got {action_array.shape}"
            )
        if np.any((action_array < 0) | (action_array >= len(Maze.ACTIONS))):
            raise ValueError("actions must be in [0, 3]")

        deltas = np.asarray(Maze.ACTIONS, dtype=np.int64)
        old_positions = self.positions[selected].copy()
        action_masks = self.valid_action_masks()[selected]
        local_indices = np.arange(len(selected))
        valid = action_masks[local_indices, action_array]
        candidates = old_positions + deltas[action_array]
        self.positions[selected[valid]] = candidates[valid]
        invalid = ~valid
        self.invalid_moves[selected] += invalid.astype(np.int64)

        old_distance = self.bfs_distances[
            selected,
            old_positions[:, 0],
            old_positions[:, 1],
        ]
        new_distance = self.bfs_distances[
            selected,
            self.positions[selected, 0],
            self.positions[selected, 1],
        ]
        solved = np.all(self.positions[selected] == self.goals[selected], axis=1)
        rewards = np.full(len(selected), self.step_penalty, dtype=np.float32)
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
        self.steps[selected] += 1
        timeout = ~solved & (
            self.steps[selected] >= self.max_episode_steps[selected]
        )
        rewards += timeout.astype(np.float32) * self.timeout_penalty
        dones = solved | timeout
        self.total_rewards[selected] += rewards
        next_states = self.observations()
        next_masks = self.valid_action_masks()
        return BatchStep(
            states=next_states if all_selected else next_states[selected],
            rewards=rewards,
            dones=dones,
            action_masks=next_masks if all_selected else next_masks[selected],
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
            nn.Conv2d(5, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        layers.extend(ResidualBlock(channels) for _ in range(residual_blocks))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        walls_goal_time = torch.cat((x[:, 0:1], x[:, 2:4]), dim=1)
        coordinates = self.coordinate_planes.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        return self.net(torch.cat((walls_goal_time, coordinates), dim=1))


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
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
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
        if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint {path!r} is not a MouseMaze schema-v{CHECKPOINT_SCHEMA_VERSION} checkpoint"
            )
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
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
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
        if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint {path!r} is not a MouseMaze schema-v{CHECKPOINT_SCHEMA_VERSION} checkpoint"
            )
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


class RecurrentActorCriticNetwork(nn.Module):
    """Spatial actor-critic with explicit episodic GRU memory."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        hidden_size: int = RECURRENT_HIDDEN_SIZE,
        output_size: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.trunk = SpatialTrunk(input_shape, channels=64)
        self.gru = nn.GRUCell(64 + output_size + 1, hidden_size)
        self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def spatial_features(self, observations: torch.Tensor) -> torch.Tensor:
        return _gather_agent_cell(self.trunk(observations), observations)

    def forward_step(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden * (~episode_starts).to(hidden.dtype).unsqueeze(1)
        features = self.spatial_features(observations)
        valid_actions = previous_actions >= 0
        safe_actions = previous_actions.clamp_min(0)
        action_features = nn.functional.one_hot(safe_actions, 4).to(features.dtype)
        action_features *= valid_actions.to(features.dtype).unsqueeze(1)
        inputs = torch.cat(
            (features, action_features, previous_rewards.to(features.dtype).unsqueeze(1)),
            dim=1,
        )
        next_hidden = self.gru(inputs, hidden.to(inputs.dtype))
        logits = self.policy_head(next_hidden).float()
        values = self.value_head(next_hidden).squeeze(1).float()
        return logits, values, next_hidden.float()

    def forward_sequence(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        initial_hidden: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        hidden = initial_hidden
        for step in range(observations.shape[0]):
            step_logits, step_values, hidden = self.forward_step(
                observations[step],
                previous_actions[step],
                previous_rewards[step],
                episode_starts[step],
                hidden,
            )
            logits.append(step_logits)
            values.append(step_values)
        return torch.stack(logits), torch.stack(values), hidden


class RNDModule(nn.Module):
    """Random-network-distillation bonus using only observable state."""

    def __init__(self, input_shape: tuple[int, int, int], device: torch.device):
        super().__init__()
        input_size = math.prod(input_shape)
        self.target = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("error_variance", torch.ones(()))
        self.to(device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=1e-4)

    def bonus_and_update(self, observations: torch.Tensor, clip: float) -> torch.Tensor:
        bonuses: list[torch.Tensor] = []
        for chunk in observations.flatten(0, 1).split(2048):
            with torch.no_grad():
                target = self.target(chunk.float())
            prediction = self.predictor(chunk.float())
            errors = (prediction - target).square().mean(dim=1)
            loss = errors.mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
            self.optimizer.step()
            bonuses.append(errors.detach())
        errors = torch.cat(bonuses)
        batch_variance = errors.var(unbiased=False).clamp_min(1e-6)
        self.error_variance.mul_(0.99).add_(batch_variance, alpha=0.01)
        normalized = errors / self.error_variance.sqrt().clamp_min(1e-3)
        return normalized.clamp(0.0, clip).view(observations.shape[:2])


class RecurrentPPOAgent:
    """Masked recurrent PPO agent for fully and partially observed tasks."""

    def __init__(
        self,
        config: TrainConfig | None = None,
        observation_shape_: tuple[int, int, int] | None = None,
        device: torch.device | None = None,
    ):
        self.config = config or TrainConfig(algorithm="recurrent_ppo")
        self.device = device or select_device(self.config.device, self.config.require_cuda)
        self.algorithm = "recurrent_ppo"
        self.network_type = "recurrent_spatial"
        self.observation_mode = self.config.observation_mode
        self.view_size = self.config.view_size
        self.observation_shape = observation_shape_ or observation_shape(
            self.config.maze_size,
            self.config.observation_mode,
            self.config.view_size,
        )
        self.performance_profile = resolved_performance_profile(
            self.config.performance_profile,
            self.device,
        )
        self.policy_net = RecurrentActorCriticNetwork(
            self.observation_shape,
            self.config.recurrent_hidden_size,
        ).to(self.device)
        fused = self.device.type == "cuda" and self.performance_profile == "rtx3090-fast"
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.config.learning_rate,
            fused=fused,
        )
        self.rnd = (
            RNDModule(self.observation_shape, self.device)
            if self.observation_mode == "local" and self.config.rnd_reward_coef > 0
            else None
        )
        self.update_count = 0
        self.total_env_steps = 0
        self.best_greedy_solve_rate = -1.0
        self.training_state: dict[str, object] = {}
        self._compiled_step = None
        self._compiled_sequence = None
        if self.performance_profile == "rtx3090-fast" and hasattr(torch, "compile"):
            self._compiled_step = torch.compile(
                self.policy_net.forward_step,
                mode="reduce-overhead",
            )
            self._compiled_sequence = torch.compile(
                self.policy_net.forward_sequence,
                mode="reduce-overhead",
            )

    @property
    def use_amp(self) -> bool:
        return self.device.type == "cuda" and self.performance_profile == "rtx3090-fast"

    def autocast(self):
        if self.use_amp:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    def initial_policy_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(
            batch_size,
            self.config.recurrent_hidden_size,
            dtype=torch.float32,
            device=self.device,
        )

    def forward_step(self, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        implementation = self._compiled_step or self.policy_net.forward_step
        try:
            result = implementation(*args)
            if self._compiled_step is not None:
                return tuple(value.clone() for value in result)
            return result
        except Exception as exc:
            if self._compiled_step is None:
                raise
            print(f"[train] torch.compile step fallback: {exc}")
            self._compiled_step = None
            return self.policy_net.forward_step(*args)

    def forward_sequence(self, *args) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        implementation = self._compiled_sequence or self.policy_net.forward_sequence
        try:
            result = implementation(*args)
            if self._compiled_sequence is not None:
                return tuple(value.clone() for value in result)
            return result
        except Exception as exc:
            if self._compiled_sequence is None:
                raise
            print(f"[train] torch.compile sequence fallback: {exc}")
            self._compiled_sequence = None
            return self.policy_net.forward_sequence(*args)

    def step(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.autocast():
            logits, values, next_hidden = self.forward_step(
                states,
                previous_actions,
                previous_rewards,
                episode_starts,
                hidden,
            )
        logits = logits.masked_fill(~action_masks, -torch.inf)
        distribution = torch.distributions.Categorical(logits=logits)
        actions = logits.argmax(dim=1) if deterministic else distribution.sample()
        return actions, distribution.log_prob(actions), values, next_hidden

    def get_actions_stateful(
        self,
        states: np.ndarray,
        action_masks: np.ndarray,
        previous_actions: np.ndarray,
        previous_rewards: np.ndarray,
        episode_starts: np.ndarray,
        hidden: torch.Tensor,
    ) -> tuple[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            actions, _log_probs, _values, next_hidden = self.step(
                torch.as_tensor(states, dtype=torch.float32, device=self.device),
                torch.as_tensor(action_masks, dtype=torch.bool, device=self.device),
                torch.as_tensor(previous_actions, dtype=torch.long, device=self.device),
                torch.as_tensor(previous_rewards, dtype=torch.float32, device=self.device),
                torch.as_tensor(episode_starts, dtype=torch.bool, device=self.device),
                hidden,
                deterministic=True,
            )
        return actions.cpu().numpy().astype(np.int64), next_hidden

    def get_actions(
        self,
        states: np.ndarray,
        epsilon: float = 0.0,
        action_masks: np.ndarray | None = None,
    ) -> np.ndarray:
        del epsilon
        if states.ndim == len(self.observation_shape):
            states = states[np.newaxis]
        masks = MouseAgent._normalize_action_masks(action_masks, len(states))
        if masks is None:
            masks = np.ones((len(states), 4), dtype=np.bool_)
        actions, _hidden = self.get_actions_stateful(
            states,
            masks,
            np.full(len(states), -1, dtype=np.int64),
            np.zeros(len(states), dtype=np.float32),
            np.ones(len(states), dtype=np.bool_),
            self.initial_policy_state(len(states)),
        )
        return actions

    def get_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        action_mask: np.ndarray | None = None,
    ) -> int:
        return int(self.get_actions(state, epsilon, action_mask)[0])

    def save(self, path: str) -> None:
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "algorithm": self.algorithm,
            "network_type": self.network_type,
            "state_dict": self.policy_net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rnd_state_dict": self.rnd.state_dict() if self.rnd is not None else None,
            "rnd_optimizer_state_dict": (
                self.rnd.optimizer.state_dict() if self.rnd is not None else None
            ),
            "observation_shape": self.observation_shape,
            "observation_mode": self.observation_mode,
            "view_size": self.view_size,
            "maze_size": self.config.maze_size,
            "recurrent_hidden_size": self.config.recurrent_hidden_size,
            "performance_profile": self.performance_profile,
            "update_count": self.update_count,
            "total_env_steps": self.total_env_steps,
            "best_greedy_solve_rate": self.best_greedy_solve_rate,
            "training_state": self.training_state,
        }
        _atomic_torch_save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"checkpoint {path!r} is not a MouseMaze schema-v{CHECKPOINT_SCHEMA_VERSION} checkpoint"
            )
        if payload.get("algorithm") != self.algorithm:
            raise ValueError(f"checkpoint algorithm is {payload.get('algorithm')!r}, expected {self.algorithm!r}")
        self.policy_net.load_state_dict(payload["state_dict"])
        self.optimizer.load_state_dict(payload["optimizer_state_dict"])
        if self.rnd is not None and payload.get("rnd_state_dict") is not None:
            self.rnd.load_state_dict(payload["rnd_state_dict"])
            self.rnd.optimizer.load_state_dict(payload["rnd_optimizer_state_dict"])
        self.update_count = int(payload.get("update_count", 0))
        self.total_env_steps = int(payload.get("total_env_steps", 0))
        self.best_greedy_solve_rate = float(payload.get("best_greedy_solve_rate", -1.0))
        self.training_state = dict(payload.get("training_state", {}))


Agent = MouseAgent | MaskedPPOAgent | RecurrentPPOAgent


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
        x_value = (
            episode
            if episode is not None
            else (self.reward_history[-1][0] + 1 if self.reward_history else 1)
        )
        _append_chart_point(
            self.train_solve_history,
            (x_value, self.train_solve_rate),
        )
        _append_chart_point(self.reward_history, (x_value, self.avg_reward))

    def record_loss(self, loss: float | None, episode: int | None = None) -> None:
        if loss is None:
            return
        self.losses.append(loss)
        x_value = (
            episode
            if episode is not None
            else (self.loss_history[-1][0] + 1 if self.loss_history else 1)
        )
        _append_chart_point(self.loss_history, (x_value, self.loss_ema))

    def record_eval(self, metrics: EvalMetrics, episode: int | None = None) -> None:
        self.latest_eval = metrics
        x_value = (
            episode
            if episode is not None
            else (
                self.greedy_solve_history[-1][0] + 1
                if self.greedy_solve_history
                else 1
            )
        )
        _append_chart_point(self.greedy_solve_history, (x_value, metrics.solve_rate))
        _append_chart_point(self.greedy_steps_history, (x_value, metrics.avg_steps))
        _append_chart_point(
            self.greedy_optimality_history,
            (x_value, metrics.optimality_ratio),
        )

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


def _recurrent_ppo_precision_schedule(
    total_steps: int,
    config: TrainConfig,
) -> tuple[float, float]:
    """Return the proven finite-budget schedule and its stable precision floor."""

    budget = max(config.max_env_steps, 1)
    progress = min(max(total_steps, 0) / budget, 1.0)
    learning_rate = config.learning_rate * (1.0 - 0.9 * progress)
    return learning_rate, config.ppo_entropy_coef


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
    failed_grids: list[np.ndarray] = []
    bucket_totals: dict[str, int] = {}
    bucket_solves: dict[str, int] = {}
    eval_rng = random.Random(resolved_eval_seed(config))
    make_env = maze_factory or (lambda: make_maze(config, rng=eval_rng))

    envs = [make_env() for _ in range(config.eval_episodes)]
    buckets = [difficulty_bucket(env.optimal_start_steps) for env in envs]
    for bucket in buckets:
        bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1
    for env in envs:
        env.reset()
    environment_batch = MazeBatch(envs)
    states = environment_batch.observations()
    active = np.ones(len(envs), dtype=np.bool_)
    recurrent_hidden = (
        agent.initial_policy_state(len(envs))
        if isinstance(agent, RecurrentPPOAgent)
        else None
    )
    previous_actions = np.full(len(envs), -1, dtype=np.int64)
    previous_rewards = np.zeros(len(envs), dtype=np.float32)
    episode_starts = np.ones(len(envs), dtype=np.bool_)
    position_traces = [[env.current_position] for env in envs]
    seen_state_actions: list[set[tuple[tuple[int, int], int]]] = [set() for _ in envs]
    has_loop = np.zeros(len(envs), dtype=np.bool_)
    has_repeated_state_action = np.zeros(len(envs), dtype=np.bool_)

    while active.any():
        active_indices = np.flatnonzero(active)
        state_batch = states[active_indices]
        action_masks = environment_batch.valid_action_masks()[active_indices]
        if isinstance(agent, RecurrentPPOAgent):
            assert recurrent_hidden is not None
            hidden_indices = torch.as_tensor(
                active_indices,
                dtype=torch.long,
                device=agent.device,
            )
            actions, next_hidden = agent.get_actions_stateful(
                state_batch,
                action_masks,
                previous_actions[active_indices],
                previous_rewards[active_indices],
                episode_starts[active_indices],
                recurrent_hidden.index_select(0, hidden_indices),
            )
            recurrent_hidden.index_copy_(0, hidden_indices, next_hidden)
        else:
            actions = _agent_greedy_actions(agent, state_batch, action_masks)

        for batch_idx, env_idx in enumerate(active_indices):
            action = int(actions[batch_idx])
            position = tuple(int(value) for value in environment_batch.positions[env_idx])
            state_action = (position, action)
            if state_action in seen_state_actions[env_idx]:
                has_repeated_state_action[env_idx] = True
            seen_state_actions[env_idx].add(state_action)

        step_result = environment_batch.step(actions, active_indices)
        states[active_indices] = step_result.states
        previous_actions[active_indices] = actions
        previous_rewards[active_indices] = step_result.rewards
        episode_starts[active_indices] = False
        invalid_moves += int(step_result.invalid.sum())
        total_steps += len(active_indices)

        for batch_idx, env_idx in enumerate(active_indices):
            current_position = tuple(
                int(value) for value in environment_batch.positions[env_idx]
            )
            position_traces[env_idx].append(current_position)
            positions = position_traces[env_idx]
            if (
                len(positions) >= 5
                and positions[-1] == positions[-3]
                and positions[-2] == positions[-4]
            ):
                has_loop[env_idx] = True

            if step_result.dones[batch_idx]:
                active[env_idx] = False
                if step_result.solved[batch_idx]:
                    steps = int(environment_batch.steps[env_idx])
                    optimal_steps = int(environment_batch.optimal_steps[env_idx])
                    solved_steps.append(steps)
                    optimality.append(optimal_steps / max(steps, 1))
                    bucket = buckets[env_idx]
                    bucket_solves[bucket] = bucket_solves.get(bucket, 0) + 1
                else:
                    timeouts += 1
                    failed_grids.append(environment_batch.grids[env_idx].copy())
                    loop_episodes += int(has_loop[env_idx])
                    repeated_state_action_episodes += int(
                        has_repeated_state_action[env_idx]
                    )
                    failed_final_distances.append(
                        float(
                            environment_batch.bfs_distances[
                                env_idx,
                                environment_batch.positions[env_idx, 0],
                                environment_batch.positions[env_idx, 1],
                            ]
                        )
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
        difficulty_counts=dict(sorted(bucket_totals.items())),
        failed_grids=failed_grids,
    )


def _confirm_target_candidate(
    agent: Agent,
    config: TrainConfig,
    regular_metrics: EvalMetrics,
) -> TargetConfirmationResult:
    """Evaluate one unchanged policy on deterministic, disjoint maze suites."""

    base_seed = resolved_eval_seed(config)
    suites = [(base_seed, regular_metrics)]
    for suite_index in range(1, config.target_solve_evals):
        suite_seed = base_seed + suite_index * EVAL_SEED_OFFSET
        suite_config = replace(config, eval_seed=suite_seed)
        suites.append((suite_seed, _eval_greedy(agent, suite_config)))
    return TargetConfirmationResult(
        confirmed=all(
            metrics.solve_rate >= config.target_solve_rate
            for _seed, metrics in suites
        ),
        suites=suites,
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
    validation_config = replace(config, eval_seed=root_seed, eval_episodes=episodes)
    final_config = replace(config, eval_seed=root_seed + 1, eval_episodes=episodes)
    stress_config = replace(config, eval_seed=root_seed + 2, eval_episodes=episodes)
    stress_rng = random.Random(root_seed + 2)
    stress_ranges = ((1, 8), (9, 16), (17, 24), (25, 10_000))
    stress_index = 0

    def make_stress_maze() -> Maze:
        nonlocal stress_index
        target_range = stress_ranges[stress_index % len(stress_ranges)]
        stress_index += 1
        return make_maze(stress_config, rng=stress_rng, target_range=target_range)

    metrics = {
        "validation": _eval_greedy(agent, validation_config),
        "final_test": _eval_greedy(agent, final_config),
        "stress_test": _eval_greedy(
            agent,
            stress_config,
            maze_factory=make_stress_maze,
        ),
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
        candidate_ticks = _episode_tick_values(x_max, pw)
        tick_labels = [str(episode) for episode in candidate_ticks]
        visible_ticks = _non_overlapping_episode_ticks(
            candidate_ticks,
            x_max,
            pw,
            [tick_font.size(label)[0] for label in tick_labels],
        )
        for episode in visible_ticks:
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
    return f"{value * 100:.2f}%"


def _format_number(value: float) -> str:
    return "-" if value <= 0 else f"{value:.2f}"


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


def _non_overlapping_episode_ticks(
    ticks: list[int],
    max_episode: int,
    plot_width: int,
    label_widths: list[int],
    minimum_gap: int = 6,
) -> list[int]:
    """Keep endpoint ticks and only interior labels that fit without overlap."""

    if len(ticks) <= 2:
        return ticks.copy()
    if len(label_widths) != len(ticks):
        raise ValueError("label_widths must match ticks")
    maximum = max(int(max_episode), 1)
    positions = [int(tick / maximum * max(plot_width - 1, 0)) for tick in ticks]
    boxes = [
        (position - width / 2, position + width / 2)
        for position, width in zip(positions, label_widths)
    ]
    selected = [0, len(ticks) - 1]
    for index in range(1, len(ticks) - 1):
        left, right = boxes[index]
        if all(
            right + minimum_gap <= boxes[other][0]
            or left >= boxes[other][1] + minimum_gap
            for other in selected
        ):
            selected.append(index)
    return [ticks[index] for index in sorted(selected)]


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
def invocation_timestamp(now: datetime | None = None) -> str:
    """Return a precise, filesystem-safe UTC timestamp for run artifacts."""

    instant = now or datetime.now(timezone.utc)
    return instant.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def default_artifact_paths(timestamp: str) -> tuple[str, str]:
    """Return paired default model and log paths for an invocation."""

    filename = f"{timestamp}_{ARTIFACT_BASENAME}"
    return (
        os.path.join(MODEL_RESULTS_DIR, f"{filename}.pth"),
        os.path.join(LOG_RESULTS_DIR, f"{filename}.jsonl"),
    )


def paired_log_path(model_path: str, log_directory: str | None = None) -> str:
    """Return the timestamp-matched JSONL path for a model artifact."""

    suffix = f"_{ARTIFACT_BASENAME}.pth"
    name = os.path.basename(model_path)
    if not name.endswith(suffix):
        raise ValueError(
            f"model path must end with {suffix!r} to derive its paired training log"
        )
    timestamp = name[: -len(suffix)]
    try:
        datetime.strptime(timestamp, "%Y%m%dT%H%M%S%fZ")
    except ValueError as error:
        raise ValueError(
            "model path does not use a timestamped MouseMaze artifact name: "
            f"{model_path!r}"
        ) from error
    return os.path.join(
        log_directory or LOG_RESULTS_DIR,
        f"{timestamp}_{ARTIFACT_BASENAME}.jsonl",
    )


def latest_checkpoint_path(save_path: str) -> str:
    """Return the resumable sidecar path paired with a frozen-best checkpoint."""

    stem, extension = os.path.splitext(save_path)
    return f"{stem}.latest{extension}"


def latest_model_path(model_directory: str = MODEL_RESULTS_DIR) -> str:
    """Return the latest timestamp-named MouseMaze model in a directory."""

    suffix = f"_{ARTIFACT_BASENAME}.pth"
    candidates: list[str] = []
    try:
        for name in os.listdir(model_directory):
            if not name.endswith(suffix):
                continue
            timestamp = name[: -len(suffix)]
            try:
                datetime.strptime(timestamp, "%Y%m%dT%H%M%S%fZ")
            except ValueError:
                continue
            candidates.append(os.path.join(model_directory, name))
    except FileNotFoundError:
        pass
    candidates.sort()
    if not candidates:
        raise FileNotFoundError(
            "no timestamped MouseMaze model found in "
            f"{model_directory!r}; train a model or pass --save-path explicitly"
        )
    return candidates[-1]


def resolve_cli_artifacts(
    config: TrainConfig,
    args: argparse.Namespace,
    timestamp: str,
) -> TrainConfig:
    """Resolve invocation-specific CLI artifact paths without changing overrides."""

    values = {field_.name: getattr(config, field_.name) for field_ in fields(config)}
    default_model, default_log = default_artifact_paths(timestamp)
    training = _optional_bool(args.train_flag, DEFAULT_TRAIN_FLAG)
    if training:
        resumed_model: str | None = None
        if config.resume and args.save_path is None:
            try:
                resumed_model = latest_model_path()
            except FileNotFoundError:
                pass
        if resumed_model is not None:
            values["save_path"] = resumed_model
            if args.training_log_path is None:
                resumed_log = paired_log_path(resumed_model)
                if not os.path.exists(resumed_log):
                    raise FileNotFoundError(
                        "latest MouseMaze model has no matching training log: "
                        f"{resumed_log!r}; pass --training-log-path explicitly"
                    )
                values["training_log_path"] = resumed_log
        else:
            if args.save_path is None:
                values["save_path"] = default_model
            if args.training_log_path is None:
                values["training_log_path"] = default_log
    inference = _optional_bool(args.infer_flag, DEFAULT_INFER_FLAG)
    if (
        not training
        and args.save_path is None
        and (args.benchmark or (inference and not config.planner_fallback))
    ):
        values["save_path"] = latest_model_path()
    return TrainConfig(**values)


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
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
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


def _safe_command_output(command: list[str], *, allow_empty: bool = False) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_DIR,
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    output = result.stdout.strip()
    return output if output or allow_empty else None


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for package in ("numpy", "torch", "pygame", "psutil"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _git_provenance() -> dict[str, object]:
    status = _safe_command_output(
        ["git", "status", "--porcelain"], allow_empty=True
    )
    return {
        "commit": _safe_command_output(["git", "rev-parse", "HEAD"]),
        "branch": _safe_command_output(["git", "branch", "--show-current"]),
        "dirty": None if status is None else bool(status),
    }


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
        "hostname": socket.gethostname(),
        "process_id": os.getpid(),
        "command_line": sys.argv,
        "working_directory": os.getcwd(),
        "packages": _package_versions(),
        "git": _git_provenance(),
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
        "difficulty_counts": metrics.difficulty_counts or {},
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
    start_completed: int = 0,
    start_total_steps: int = 0,
) -> dict[str, float | int]:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    process_steps = max(total_steps - start_total_steps, 0)
    process_episodes = max(completed - start_completed, 0)
    return {
        "elapsed_seconds": elapsed,
        "process_steps": process_steps,
        "process_episodes": process_episodes,
        "steps_per_second": process_steps / elapsed,
        "episodes_per_second": process_episodes / elapsed,
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
    process_start_steps = int(getattr(agent, "_process_start_total_steps", 0))
    process_start_episodes = int(getattr(agent, "_process_start_completed", 0))
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
        "speed": _training_speed_payload(
            completed,
            total_steps,
            start_time,
            process_start_episodes,
            process_start_steps,
        ),
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


def clone_optimizer_state(agent: Agent) -> dict[str, object]:
    """Clone optimizer state for an in-memory frozen-best rollback."""

    return copy.deepcopy(agent.optimizer.state_dict())


def restore_optimizer_state(agent: Agent, state: dict[str, object]) -> None:
    """Restore optimizer moments without changing training counters or RNGs."""

    agent.optimizer.load_state_dict(copy.deepcopy(state))


def checkpoint_training_states(
    path: str,
) -> tuple[dict[str, torch.Tensor], dict[str, object] | None]:
    """Load frozen model and optimizer states without mutating the live agent."""

    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise ValueError(
            f"checkpoint {path!r} is not a MouseMaze "
            f"schema-v{CHECKPOINT_SCHEMA_VERSION} checkpoint"
        )
    weights = {
        key: value.detach().cpu().clone()
        for key, value in payload["state_dict"].items()
    }
    optimizer_state = payload.get("optimizer_state_dict")
    return weights, copy.deepcopy(optimizer_state)


def create_agent(
    config: TrainConfig,
    observation_shape_: tuple[int, int, int],
    device: torch.device,
) -> Agent:
    if config.algorithm == "dqn":
        return MouseAgent(config=config, observation_shape_=observation_shape_, device=device)
    if config.algorithm == "ppo":
        return MaskedPPOAgent(config=config, observation_shape_=observation_shape_, device=device)
    return RecurrentPPOAgent(config=config, observation_shape_=observation_shape_, device=device)


def checkpoint_algorithm(path: str) -> str:
    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint {path!r} is not a MouseMaze schema-v{CHECKPOINT_SCHEMA_VERSION} checkpoint"
        )
    return str(payload.get("algorithm", "dqn"))


def ensure_agent_matches_config(agent: Agent, config: TrainConfig) -> None:
    if agent.algorithm != config.algorithm:
        raise ValueError(
            "agent algorithm does not match TrainConfig. "
            f"agent={agent.algorithm}, config={config.algorithm}"
        )


def _cpu_rng_state(state: object) -> torch.Tensor:
    """Convert a serialized RNG state to the CPU byte tensor PyTorch expects."""

    if isinstance(state, torch.Tensor):
        return state.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return torch.as_tensor(state, dtype=torch.uint8, device="cpu").contiguous()


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
        # ``Agent.load`` maps checkpoint tensors to the agent device.  The
        # PyTorch generators expect CPU byte tensors for their state, even
        # when the model and the CUDA generators are used on the GPU.
        torch.set_rng_state(_cpu_rng_state(torch_state))
    if cuda_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(_cpu_rng_state(state) for state in cuda_states)


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
    hard_grids: list[np.ndarray] | None = None,
    hard_sampler: MazeTaskSampler | None = None,
    precision_recovery: dict[str, object] | None = None,
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
    if "last_latest_checkpoint_step" in agent.training_state:
        state["last_latest_checkpoint_step"] = int(
            agent.training_state["last_latest_checkpoint_step"]
        )
    if torch.cuda.is_available():
        state["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    if curriculum is not None:
        state["curriculum"] = {
            "level": curriculum.level,
            "success_streak": curriculum.success_streak,
        }
    if hard_sampler is not None and hard_sampler.config.hard_maze_fraction > 0.0:
        state["hard_maze_grids"] = [
            grid.copy() for grid in hard_sampler.hard_grids
        ]
        state["hard_maze_seen_keys"] = sorted(hard_sampler._seen_hard_grid_keys)
        state["hard_maze_candidates_seen"] = hard_sampler.hard_candidates_seen
        state["hard_maze_validation_solve_rate"] = (
            hard_sampler.validation_solve_rate
        )
    elif hard_grids is not None:
        state["hard_maze_grids"] = [grid.copy() for grid in hard_grids]
    if precision_recovery is not None:
        state["precision_recovery"] = dict(precision_recovery)
    agent.training_state = state


def _save_latest_checkpoint(
    agent: Agent,
    config: TrainConfig,
    logger: _TrainingLogger,
    total_steps: int,
    reason: str,
    *,
    force: bool = False,
) -> bool:
    """Atomically save resumable training state when its cadence is due."""

    if config.save_path is None:
        return False
    last_save_step = int(agent.training_state.get("last_latest_checkpoint_step", 0))
    if (
        not force
        and total_steps - last_save_step < config.latest_checkpoint_every_steps
    ):
        return False
    agent.training_state["total_steps"] = max(
        int(agent.training_state.get("total_steps", 0)),
        total_steps,
        agent.total_env_steps,
    )
    agent.training_state["last_latest_checkpoint_step"] = total_steps
    sidecar_path = latest_checkpoint_path(config.save_path)
    agent.save(sidecar_path)
    if logger.enabled:
        logger.log(
            "checkpoint",
            checkpoint_path=sidecar_path,
            reason=reason,
            total_steps=total_steps,
            agent_total_env_steps=agent.total_env_steps,
            update_count=agent.update_count,
            best_greedy_solve_rate=agent.best_greedy_solve_rate,
        )
    return True


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
    on_new_best: Callable[[EvalMetrics], None] | None = None,
) -> tuple[EvalMetrics | None, int, float, dict[str, torch.Tensor] | None]:
    should_eval = _should_run_eval(config, completed, total_steps, last_eval_step)
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
        if on_new_best is not None:
            on_new_best(greedy)
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
    _save_latest_checkpoint(
        agent,
        config,
        logger,
        total_steps,
        "periodic_latest",
    )
    return greedy, last_eval_step, best_eval_rate, best_weights


def _should_run_eval(
    config: TrainConfig,
    completed: int,
    total_steps: int,
    last_eval_step: int,
) -> bool:
    """Return whether evaluation is due under bounded or target-only training."""

    return completed > 0 and (
        last_eval_step == 0
        or (not config.target_only_stop and completed >= config.episodes)
        or total_steps - last_eval_step >= config.eval_every_steps
    )


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
    _save_latest_checkpoint(
        agent,
        config,
        logger,
        total_steps,
        "final_latest",
        force=True,
    )
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
    training_log_path: str | None = None,
    config: TrainConfig | None = None,
    run_timestamp: str | None = None,
) -> MetricsTracker:
    """Train a MouseMaze agent and return the collected metrics."""

    config = _merge_train_args(
        config,
        maze_size,
        episodes,
        save_path,
        dashboard_flag,
        training_log_path,
    )
    set_global_seed(config.seed)
    device = select_device(config.device, config.require_cuda)
    resolved_profile = configure_performance(config.performance_profile, device)
    config.performance_profile = resolved_profile
    config.num_envs = resolve_num_envs(config)
    logger = _TrainingLogger(config.training_log_path)
    run_timestamp = run_timestamp or invocation_timestamp()
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
    resume_checkpoint_path = config.save_path
    frozen_best_weights: dict[str, torch.Tensor] | None = None
    frozen_best_optimizer_state: dict[str, object] | None = None
    if config.save_path and config.resume:
        sidecar_path = latest_checkpoint_path(config.save_path)
        if os.path.exists(sidecar_path):
            resume_checkpoint_path = sidecar_path
            if os.path.exists(config.save_path):
                (
                    frozen_best_weights,
                    frozen_best_optimizer_state,
                ) = checkpoint_training_states(config.save_path)
    if (
        resume_checkpoint_path
        and os.path.exists(resume_checkpoint_path)
        and config.resume
    ):
        agent.load(resume_checkpoint_path)
        resumed = True
        _restore_rng_state(agent.training_state)
        print(f"[train] resumed training state from {resume_checkpoint_path}")
        if logger.enabled:
            logger.log(
                "resume",
                checkpoint_path=resume_checkpoint_path,
                frozen_best_checkpoint_path=(
                    config.save_path if frozen_best_weights is not None else None
                ),
                update_count=agent.update_count,
                agent_total_env_steps=agent.total_env_steps,
                best_greedy_solve_rate=agent.best_greedy_solve_rate,
            )
    elif config.save_path and os.path.exists(config.save_path):
        print(
            "[train] starting a fresh experiment without loading "
            f"{config.save_path}"
        )

    setattr(
        agent,
        "_process_start_total_steps",
        max(int(agent.training_state.get("total_steps", 0)), agent.total_env_steps),
    )
    setattr(
        agent,
        "_process_start_completed",
        int(agent.training_state.get("completed_episodes", 0)),
    )

    if logger.enabled:
        logger.log(
            "train_start",
            artifact_timestamp=run_timestamp,
            artifact_paths={
                "model": config.save_path,
                "latest": (
                    latest_checkpoint_path(config.save_path)
                    if config.save_path is not None
                    else None
                ),
                "log": config.training_log_path,
            },
            config=_train_config_payload(config),
            environment=_training_environment_payload(device),
            checkpoint_path=resume_checkpoint_path if resumed else config.save_path,
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

    try:
        if config.algorithm == "dqn":
            assert isinstance(agent, MouseAgent)
            return _train_dqn(
                agent,
                config,
                logger,
                start_time,
                process_start_cpu,
            )
        if config.algorithm == "ppo":
            assert isinstance(agent, MaskedPPOAgent)
            return _train_ppo(
                agent,
                config,
                logger,
                start_time,
                process_start_cpu,
            )
        assert isinstance(agent, RecurrentPPOAgent)
        return _train_recurrent_ppo(
            agent,
            config,
            logger,
            start_time,
            process_start_cpu,
            initial_best_weights=frozen_best_weights,
            initial_best_optimizer_state=frozen_best_optimizer_state,
        )
    except KeyboardInterrupt:
        interrupted_steps = max(
            int(agent.training_state.get("total_steps", 0)),
            agent.total_env_steps,
        )
        saved_interruption = _save_latest_checkpoint(
            agent,
            config,
            logger,
            interrupted_steps,
            "interrupted_latest",
            force=True,
        )
        if saved_interruption:
            print("[train] interruption checkpoint saved.")
        raise


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


def _compute_gae_torch(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GAE without moving a GPU rollout back to NumPy."""

    advantages = torch.zeros_like(rewards)
    last_advantage = torch.zeros(rewards.shape[1], device=rewards.device)
    for step in range(rewards.shape[0] - 1, -1, -1):
        next_value = next_values if step == rewards.shape[0] - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        last_advantage = delta + gamma * gae_lambda * nonterminal * last_advantage
        advantages[step] = last_advantage
    return advantages, advantages + values


def _recurrent_ppo_update(
    agent: RecurrentPPOAgent,
    config: TrainConfig,
    states: torch.Tensor,
    action_masks: torch.Tensor,
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    previous_rewards: torch.Tensor,
    episode_starts: torch.Tensor,
    hidden_states: torch.Tensor,
    old_log_probs: torch.Tensor,
    old_values: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    entropy_coefficient: float | None = None,
) -> PPOUpdateMetrics:
    """Optimize recurrent PPO using intact truncated-BPTT sequences."""

    time_steps, env_count = actions.shape
    effective_entropy_coefficient = (
        config.ppo_entropy_coef
        if entropy_coefficient is None
        else float(entropy_coefficient)
    )
    sequence_length = config.recurrent_sequence_length
    chunks = [
        (env_index, start)
        for env_index in range(env_count)
        for start in range(0, time_steps, sequence_length)
    ]
    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    totals = PPOUpdateMetrics()
    update_count = 0
    stop_early = False

    for epoch in range(config.ppo_epochs):
        order = torch.randperm(len(chunks), device=agent.device).cpu().tolist()
        for offset in range(0, len(order), config.recurrent_sequence_minibatch_size):
            selected = [chunks[index] for index in order[offset : offset + config.recurrent_sequence_minibatch_size]]
            batch_count = len(selected)
            state_batch = torch.zeros(
                sequence_length,
                batch_count,
                *states.shape[2:],
                device=agent.device,
                dtype=states.dtype,
            )
            mask_batch = torch.ones(
                sequence_length,
                batch_count,
                4,
                device=agent.device,
                dtype=torch.bool,
            )
            action_batch = torch.zeros(sequence_length, batch_count, device=agent.device, dtype=torch.long)
            previous_action_batch = torch.full_like(action_batch, -1)
            previous_reward_batch = torch.zeros(sequence_length, batch_count, device=agent.device)
            start_batch = torch.ones(sequence_length, batch_count, device=agent.device, dtype=torch.bool)
            old_log_batch = torch.zeros(sequence_length, batch_count, device=agent.device)
            old_value_batch = torch.zeros(sequence_length, batch_count, device=agent.device)
            advantage_batch = torch.zeros(sequence_length, batch_count, device=agent.device)
            return_batch = torch.zeros(sequence_length, batch_count, device=agent.device)
            valid = torch.zeros(sequence_length, batch_count, device=agent.device, dtype=torch.bool)
            initial_hidden = torch.stack(
                [hidden_states[start, env_index] for env_index, start in selected]
            )
            for batch_index, (env_index, start) in enumerate(selected):
                end = min(start + sequence_length, time_steps)
                length = end - start
                source = slice(start, end)
                target = slice(0, length)
                state_batch[target, batch_index] = states[source, env_index]
                mask_batch[target, batch_index] = action_masks[source, env_index]
                action_batch[target, batch_index] = actions[source, env_index]
                previous_action_batch[target, batch_index] = previous_actions[source, env_index]
                previous_reward_batch[target, batch_index] = previous_rewards[source, env_index]
                start_batch[target, batch_index] = episode_starts[source, env_index]
                old_log_batch[target, batch_index] = old_log_probs[source, env_index]
                old_value_batch[target, batch_index] = old_values[source, env_index]
                advantage_batch[target, batch_index] = normalized_advantages[source, env_index]
                return_batch[target, batch_index] = returns[source, env_index]
                valid[target, batch_index] = True

            with agent.autocast():
                logits, new_values, _hidden = agent.forward_sequence(
                    state_batch,
                    previous_action_batch,
                    previous_reward_batch,
                    start_batch,
                    initial_hidden,
                )
            logits = logits.masked_fill(~mask_batch, -torch.inf)
            distribution = torch.distributions.Categorical(logits=logits)
            new_log_probs = distribution.log_prob(action_batch)
            ratio = torch.exp(new_log_probs - old_log_batch)
            unclipped = ratio * advantage_batch
            clipped = torch.clamp(
                ratio,
                1.0 - config.ppo_clip_range,
                1.0 + config.ppo_clip_range,
            ) * advantage_batch
            policy_loss = -torch.min(unclipped, clipped)[valid].mean()
            clipped_values = old_value_batch + torch.clamp(
                new_values - old_value_batch,
                -config.ppo_value_clip_range,
                config.ppo_value_clip_range,
            )
            value_loss = 0.5 * torch.maximum(
                (new_values - return_batch).square(),
                (clipped_values - return_batch).square(),
            )[valid].mean()
            entropy = distribution.entropy()[valid].mean()
            loss = (
                policy_loss
                + config.ppo_value_coef * value_loss
                - effective_entropy_coefficient * entropy
            )
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(agent.policy_net.parameters(), config.ppo_max_grad_norm)
            agent.optimizer.step()
            agent.update_count += 1
            approx_kl = (old_log_batch - new_log_probs)[valid].mean().detach()
            totals.loss += float(loss.item())
            totals.policy_loss += float(policy_loss.item())
            totals.value_loss += float(value_loss.item())
            totals.entropy += float(entropy.item())
            totals.approx_kl += float(approx_kl.item())
            update_count += 1
            if approx_kl > config.ppo_target_kl:
                stop_early = True
                break
        totals.epochs = epoch + 1
        if stop_early:
            break
    if update_count:
        totals.loss /= update_count
        totals.policy_loss /= update_count
        totals.value_loss /= update_count
        totals.entropy /= update_count
        totals.approx_kl /= update_count
    return totals


def _curriculum_stage_eval(
    agent: Agent,
    config: TrainConfig,
    curriculum: CurriculumController,
) -> EvalMetrics:
    target_range = curriculum.target_range()
    if target_range is None:
        return _eval_greedy(
            agent,
            replace(config, eval_episodes=config.curriculum_eval_episodes),
        )
    stage_rng = random.Random(resolved_eval_seed(config) + 100_000 * (curriculum.level + 1))
    stage_config = replace(config, eval_episodes=config.curriculum_eval_episodes)
    return _eval_greedy(
        agent,
        stage_config,
        maze_factory=lambda: make_maze(stage_config, rng=stage_rng, target_range=target_range),
    )


def _train_recurrent_ppo(
    agent: RecurrentPPOAgent,
    config: TrainConfig,
    logger: _TrainingLogger,
    start_time: float,
    process_start_cpu: float,
    initial_best_weights: dict[str, torch.Tensor] | None = None,
    initial_best_optimizer_state: dict[str, object] | None = None,
) -> MetricsTracker:
    """Train stateful PPO with GPU-resident rollouts and vectorized mazes."""

    tracker = MetricsTracker()
    dashboard = Dashboard() if config.dashboard_flag else None
    best_eval_rate = agent.best_greedy_solve_rate
    best_weights = initial_best_weights
    if best_weights is None and best_eval_rate >= 0:
        best_weights = clone_agent_weights(agent)
    saved_state = agent.training_state
    saved_completed = int(saved_state.get("completed_episodes", 0))
    completed = (
        saved_completed
        if config.target_only_stop
        else min(saved_completed, config.episodes)
    )
    total_steps = max(int(saved_state.get("total_steps", 0)), agent.total_env_steps)
    last_eval_step = int(saved_state.get("last_eval_step", 0))
    # Legacy checkpoints may contain target_streak. Confirmation now validates
    # one frozen candidate, so an evolving-policy streak is intentionally ignored.
    target_confirmed = bool(saved_state.get("target_confirmed", False))
    recovery_saved = saved_state.get("precision_recovery", {})
    if not isinstance(recovery_saved, dict):
        recovery_saved = {}
    plateau_evals = (
        max(int(recovery_saved.get("plateau_evals", 0)), 0)
        if config.precision_recovery_enabled
        else 0
    )
    recovery_active = (
        bool(recovery_saved.get("active", False))
        if config.precision_recovery_enabled
        else False
    )
    recovery_end_step = (
        max(int(recovery_saved.get("end_step", 0)), 0)
        if config.precision_recovery_enabled
        else 0
    )
    recovery_count = (
        max(int(recovery_saved.get("count", 0)), 0)
        if config.precision_recovery_enabled
        else 0
    )
    best_optimizer_state = initial_best_optimizer_state
    if best_optimizer_state is None and best_weights is not None:
        best_optimizer_state = clone_optimizer_state(agent)

    def precision_recovery_payload() -> dict[str, object]:
        return {
            "enabled": config.precision_recovery_enabled,
            "plateau_evals": plateau_evals,
            "active": recovery_active,
            "end_step": recovery_end_step,
            "count": recovery_count,
        }
    train_rng = random.Random(config.seed)
    if "train_rng_state" in saved_state:
        train_rng.setstate(saved_state["train_rng_state"])
    sampler = MazeTaskSampler(config, train_rng)
    if config.hard_maze_fraction > 0.0:
        sampler.restore_hard_grids(
            saved_state.get("hard_maze_grids"),
            saved_state.get("hard_maze_seen_keys"),
            saved_state.get("hard_maze_candidates_seen"),
            saved_state.get("hard_maze_validation_solve_rate"),
        )
    curriculum_state = saved_state.get("curriculum")
    if isinstance(curriculum_state, dict):
        sampler.curriculum.level = int(curriculum_state.get("level", 0))
        sampler.curriculum.success_streak = int(curriculum_state.get("success_streak", 0))
    maze_workers = (
        config.maze_workers if config.performance_profile == "rtx3090-fast" else 0
    )
    prefetcher = DeterministicMazePrefetcher(sampler, maze_workers)
    limits_enabled = not config.target_only_stop
    assert config.num_envs is not None
    env_count = (
        config.num_envs
        if not limits_enabled
        else min(config.num_envs, max(config.episodes - completed, 1))
    )
    environments = [prefetcher.next() for _ in range(env_count)]
    environment_batch = MazeBatch(environments)
    states = environment_batch.observations()
    hidden = agent.initial_policy_state(env_count)
    previous_actions_np = np.full(env_count, -1, dtype=np.int64)
    previous_rewards_np = np.zeros(env_count, dtype=np.float32)
    episode_starts_np = np.ones(env_count, dtype=np.bool_)

    def reset_training_batch() -> None:
        """Start fresh recurrent contexts after restoring frozen-best weights."""

        nonlocal states
        nonlocal hidden
        for index in range(env_count):
            environment_batch.replace(index, prefetcher.next())
        states = environment_batch.observations()
        hidden = agent.initial_policy_state(env_count)
        previous_actions_np.fill(-1)
        previous_rewards_np.fill(0.0)
        episode_starts_np.fill(True)

    pin_memory = agent.device.type == "cuda"
    state_staging = torch.empty(
        env_count,
        *agent.observation_shape,
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    mask_staging = torch.empty(env_count, 4, dtype=torch.bool, pin_memory=pin_memory)
    last_eval = EvalMetrics()
    last_dashboard_episode = -config.dashboard_every

    while (
        (not limits_enabled or total_steps < config.max_env_steps)
        and (not limits_enabled or completed < config.episodes)
        and not target_confirmed
    ):
        collector_started = time.perf_counter()
        rollout_steps = config.ppo_rollout_steps
        shape = (rollout_steps, env_count)
        rollout_states = torch.empty(
            *shape,
            *agent.observation_shape,
            dtype=torch.float32,
            device=agent.device,
        )
        rollout_masks = torch.empty(*shape, 4, dtype=torch.bool, device=agent.device)
        rollout_actions = torch.empty(*shape, dtype=torch.long, device=agent.device)
        rollout_previous_actions = torch.empty_like(rollout_actions)
        rollout_previous_rewards = torch.empty(*shape, device=agent.device)
        rollout_episode_starts = torch.empty(*shape, dtype=torch.bool, device=agent.device)
        rollout_hidden = torch.empty(
            *shape,
            config.recurrent_hidden_size,
            device=agent.device,
        )
        rollout_log_probs = torch.empty(*shape, device=agent.device)
        rollout_values = torch.empty(*shape, device=agent.device)
        rollout_rewards = torch.empty(*shape, device=agent.device)
        rollout_dones = torch.empty(*shape, device=agent.device)
        actual_steps = 0
        transfer_seconds = 0.0
        environment_seconds = 0.0
        maze_generation_seconds = 0.0

        for step in range(rollout_steps):
            if limits_enabled and (
                total_steps >= config.max_env_steps or completed >= config.episodes
            ):
                break
            transfer_started = time.perf_counter()
            state_staging.copy_(torch.from_numpy(states))
            state_tensor = state_staging.to(agent.device, non_blocking=pin_memory)
            mask_np = environment_batch.valid_action_masks()
            mask_staging.copy_(torch.from_numpy(mask_np))
            mask_tensor = mask_staging.to(agent.device, non_blocking=pin_memory)
            transfer_seconds += time.perf_counter() - transfer_started
            previous_action_tensor = torch.as_tensor(
                previous_actions_np,
                dtype=torch.long,
                device=agent.device,
            )
            previous_reward_tensor = torch.as_tensor(
                previous_rewards_np,
                dtype=torch.float32,
                device=agent.device,
            )
            episode_start_tensor = torch.as_tensor(
                episode_starts_np,
                dtype=torch.bool,
                device=agent.device,
            )
            rollout_states[step].copy_(state_tensor)
            rollout_masks[step].copy_(mask_tensor)
            rollout_previous_actions[step].copy_(previous_action_tensor)
            rollout_previous_rewards[step].copy_(previous_reward_tensor)
            rollout_episode_starts[step].copy_(episode_start_tensor)
            rollout_hidden[step].copy_(hidden)
            with torch.no_grad():
                actions_t, log_probs_t, values_t, hidden = agent.step(
                    state_tensor,
                    mask_tensor,
                    previous_action_tensor,
                    previous_reward_tensor,
                    episode_start_tensor,
                    hidden,
                    deterministic=False,
                )
            actions_np = actions_t.cpu().numpy().astype(np.int64)
            environment_started = time.perf_counter()
            result = environment_batch.step(actions_np)
            environment_seconds += time.perf_counter() - environment_started
            rollout_actions[step].copy_(actions_t)
            rollout_log_probs[step].copy_(log_probs_t)
            rollout_values[step].copy_(values_t)
            rollout_rewards[step].copy_(
                torch.as_tensor(result.rewards, device=agent.device)
            )
            rollout_dones[step].copy_(
                torch.as_tensor(result.dones, dtype=torch.float32, device=agent.device)
            )
            for index in np.flatnonzero(result.dones):
                if limits_enabled and completed >= config.episodes:
                    break
                completed += 1
                tracker.record_episode(environment_batch.episode_stats(int(index)), completed)
                if not limits_enabled or completed < config.episodes:
                    generation_started = time.perf_counter()
                    replacement = prefetcher.next()
                    maze_generation_seconds += time.perf_counter() - generation_started
                    environment_batch.replace(int(index), replacement)
            total_steps += env_count
            agent.total_env_steps += env_count
            actual_steps = step + 1
            states = environment_batch.observations()
            previous_actions_np = np.where(result.dones, -1, actions_np)
            previous_rewards_np = np.where(result.dones, 0.0, result.rewards).astype(np.float32)
            episode_starts_np = result.dones.copy()

        if actual_steps == 0:
            break
        collector_seconds = time.perf_counter() - collector_started
        rollout_states = rollout_states[:actual_steps]
        rollout_masks = rollout_masks[:actual_steps]
        rollout_actions = rollout_actions[:actual_steps]
        rollout_previous_actions = rollout_previous_actions[:actual_steps]
        rollout_previous_rewards = rollout_previous_rewards[:actual_steps]
        rollout_episode_starts = rollout_episode_starts[:actual_steps]
        rollout_hidden = rollout_hidden[:actual_steps]
        rollout_log_probs = rollout_log_probs[:actual_steps]
        rollout_values = rollout_values[:actual_steps]
        rollout_rewards = rollout_rewards[:actual_steps]
        rollout_dones = rollout_dones[:actual_steps]

        learner_started = time.perf_counter()
        intrinsic_rewards = torch.zeros_like(rollout_rewards)
        rnd_coefficient = 0.0
        if agent.rnd is not None and total_steps < config.max_env_steps:
            rnd_coefficient = config.rnd_reward_coef * max(
                0.0,
                1.0 - total_steps / config.max_env_steps,
            )
            intrinsic_rewards = agent.rnd.bonus_and_update(
                rollout_states,
                config.rnd_reward_clip,
            )
        combined_rewards = rollout_rewards + rnd_coefficient * intrinsic_rewards
        next_state_tensor = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
        with torch.no_grad():
            _actions, _log_probs, next_values, _next_hidden = agent.step(
                next_state_tensor,
                torch.as_tensor(
                    environment_batch.valid_action_masks(),
                    dtype=torch.bool,
                    device=agent.device,
                ),
                torch.as_tensor(previous_actions_np, dtype=torch.long, device=agent.device),
                torch.as_tensor(previous_rewards_np, dtype=torch.float32, device=agent.device),
                torch.as_tensor(episode_starts_np, dtype=torch.bool, device=agent.device),
                hidden,
                deterministic=True,
            )
        advantages, returns = _compute_gae_torch(
            combined_rewards,
            rollout_dones,
            rollout_values,
            next_values,
            config.gamma,
            config.ppo_gae_lambda,
        )
        learning_rate, entropy_coefficient = _recurrent_ppo_precision_schedule(
            total_steps,
            config,
        )
        if config.precision_recovery_enabled and recovery_active:
            learning_rate = (
                config.learning_rate * config.precision_recovery_lr_fraction
            )
        for group in agent.optimizer.param_groups:
            group["lr"] = learning_rate
        updates_before_rollout = agent.update_count
        update_metrics = _recurrent_ppo_update(
            agent,
            config,
            rollout_states,
            rollout_masks,
            rollout_actions,
            rollout_previous_actions,
            rollout_previous_rewards,
            rollout_episode_starts,
            rollout_hidden,
            rollout_log_probs,
            rollout_values,
            advantages,
            returns,
            entropy_coefficient=entropy_coefficient,
        )
        tracker.record_loss(update_metrics.loss, completed)
        learner_seconds = time.perf_counter() - learner_started
        if logger.enabled:
            elapsed = max(time.perf_counter() - start_time, 1e-9)
            process_start_steps = int(
                getattr(agent, "_process_start_total_steps", 0)
            )
            steps_per_second = max(total_steps - process_start_steps, 0) / elapsed
            estimated_seconds_remaining = (
                (config.max_env_steps - total_steps)
                / max(steps_per_second, 1e-9)
                if total_steps < config.max_env_steps
                else None
            )
            logger.log(
                "learner",
                episode=completed,
                total_steps=total_steps,
                update_count=agent.update_count,
                learning_rate=learning_rate,
                entropy_coefficient=entropy_coefficient,
                precision_recovery_active=recovery_active,
                rnd_coefficient=rnd_coefficient,
                extrinsic_reward_mean=float(rollout_rewards.mean().item()),
                intrinsic_reward_mean=float(intrinsic_rewards.mean().item()),
                ppo={
                    "loss": update_metrics.loss,
                    "policy_loss": update_metrics.policy_loss,
                    "value_loss": update_metrics.value_loss,
                    "entropy": update_metrics.entropy,
                    "approx_kl": update_metrics.approx_kl,
                    "epochs": update_metrics.epochs,
                    "effective_transition_minibatch_size": (
                        config.recurrent_sequence_length
                        * config.recurrent_sequence_minibatch_size
                    ),
                    "optimizer_updates_this_rollout": (
                        agent.update_count - updates_before_rollout
                    ),
                },
                timing={
                    "collector_seconds": collector_seconds,
                    "transfer_seconds": transfer_seconds,
                    "environment_seconds": environment_seconds,
                    "maze_generation_seconds": maze_generation_seconds,
                    "learner_seconds": learner_seconds,
                    "steps_per_second": steps_per_second,
                    "estimated_seconds_remaining": estimated_seconds_remaining,
                },
            )

        _update_training_state(
            agent,
            completed,
            total_steps,
            last_eval_step,
            train_rng,
            curriculum=sampler.curriculum,
            hard_sampler=sampler,
            precision_recovery=precision_recovery_payload(),
        )
        agent.training_state["target_confirmed"] = False
        _save_latest_checkpoint(
            agent,
            config,
            logger,
            total_steps,
            "periodic_latest",
        )

        def record_evaluation(metrics: EvalMetrics) -> None:
            promoted = False
            stage_metrics = metrics
            if config.curriculum_enabled and not sampler.curriculum.complete:
                stage_metrics = _curriculum_stage_eval(agent, config, sampler.curriculum)
                promoted = sampler.curriculum.record_validation(stage_metrics)
                if promoted:
                    prefetcher.reset()
            if not config.curriculum_enabled or sampler.curriculum.complete:
                sampler.record_validation_solve_rate(metrics.solve_rate)
            hard_variants_added = (
                sampler.add_failed_grids(metrics.failed_grids)
                if config.hard_maze_fraction > 0.0
                and (not config.curriculum_enabled or sampler.curriculum.complete)
                else 0
            )
            _update_training_state(
                agent,
                completed,
                total_steps,
                total_steps,
                train_rng,
                curriculum=sampler.curriculum,
                hard_sampler=sampler,
                precision_recovery=precision_recovery_payload(),
            )
            agent.training_state["target_confirmed"] = False
            if logger.enabled:
                logger.log(
                    "curriculum",
                    level=sampler.curriculum.level,
                    promoted=promoted,
                    stage_metrics=_eval_metrics_payload(stage_metrics),
                    sampling_mix=sampler.sampling_mix(),
                    hard_variants_added=hard_variants_added,
                    hard_maze_pool_size=len(sampler.hard_grids),
                    hard_maze_candidates_seen=sampler.hard_candidates_seen,
                    validation_solve_rate=sampler.validation_solve_rate,
                )

        evaluation_improved = False

        def record_new_best(_metrics: EvalMetrics) -> None:
            nonlocal best_optimizer_state
            nonlocal evaluation_improved
            nonlocal plateau_evals
            nonlocal recovery_active
            nonlocal recovery_end_step

            evaluation_improved = True
            was_recovering = recovery_active
            best_optimizer_state = clone_optimizer_state(agent)
            plateau_evals = 0
            recovery_active = False
            recovery_end_step = 0
            _update_training_state(
                agent,
                completed,
                total_steps,
                total_steps,
                train_rng,
                curriculum=sampler.curriculum,
                hard_sampler=sampler,
                precision_recovery=precision_recovery_payload(),
            )
            agent.training_state["target_confirmed"] = False
            if was_recovering and logger.enabled:
                logger.log(
                    "precision_recovery",
                    phase="end",
                    outcome="improved",
                    total_steps=total_steps,
                    recovery_count=recovery_count,
                    best_greedy_solve_rate=agent.best_greedy_solve_rate,
                )

        evaluation_started = time.perf_counter()
        eval_result, last_eval_step, best_eval_rate, best_weights = _maybe_run_eval(
            agent,
            config,
            logger,
            tracker,
            completed,
            total_steps,
            0.0,
            best_eval_rate,
            best_weights,
            last_eval_step,
            start_time,
            process_start_cpu,
            on_evaluation=record_evaluation,
            on_new_best=record_new_best,
        )
        evaluation_seconds = time.perf_counter() - evaluation_started
        if eval_result is not None:
            last_eval = eval_result
            if logger.enabled:
                logger.log(
                    "evaluation_timing",
                    episode=completed,
                    total_steps=total_steps,
                    evaluation_seconds=evaluation_seconds,
                )
            curriculum_complete = (
                not config.curriculum_enabled or sampler.curriculum.complete
            )
            if (
                curriculum_complete
                and eval_result.solve_rate >= config.target_solve_rate
            ):
                confirmation = _confirm_target_candidate(agent, config, eval_result)
                confirmation_variants_added = (
                    sum(
                        sampler.add_failed_grids(suite_metrics.failed_grids)
                        for _suite_seed, suite_metrics in confirmation.suites[1:]
                    )
                    if config.hard_maze_fraction > 0.0
                    else 0
                )
                for suite_index, (suite_seed, suite_metrics) in enumerate(
                    confirmation.suites,
                    start=1,
                ):
                    if logger.enabled:
                        logger.log(
                            "target_confirmation_suite",
                            suite_index=suite_index,
                            suite_count=len(confirmation.suites),
                            eval_seed=suite_seed,
                            passed=suite_metrics.solve_rate >= config.target_solve_rate,
                            metrics=_eval_metrics_payload(suite_metrics),
                        )
                target_confirmed = confirmation.confirmed
                agent.training_state["target_confirmed"] = target_confirmed
                if logger.enabled:
                    logger.log(
                        "target_confirmation",
                        confirmed=target_confirmed,
                        target_solve_rate=config.target_solve_rate,
                        suite_count=len(confirmation.suites),
                        hard_variants_added=confirmation_variants_added,
                        hard_maze_pool_size=len(sampler.hard_grids),
                        hard_maze_candidates_seen=sampler.hard_candidates_seen,
                    )
                if target_confirmed:
                    best_eval_rate = max(best_eval_rate, eval_result.solve_rate)
                    agent.best_greedy_solve_rate = best_eval_rate
                    best_weights = clone_agent_weights(agent)
                    plateau_evals = 0
                    recovery_active = False
                    recovery_end_step = 0
                    _update_training_state(
                        agent,
                        completed,
                        total_steps,
                        total_steps,
                        train_rng,
                        curriculum=sampler.curriculum,
                        hard_sampler=sampler,
                        precision_recovery=precision_recovery_payload(),
                    )
                    agent.training_state["target_confirmed"] = True
                    if config.save_path:
                        agent.save(config.save_path)
                        print(
                            "[train] confirmed target weights saved to "
                            f"{config.save_path} ({best_eval_rate:.1%})."
                        )
                        if logger.enabled:
                            logger.log(
                                "checkpoint",
                                checkpoint_path=config.save_path,
                                reason="target_confirmed",
                                **_training_snapshot_payload(
                                    completed,
                                    total_steps,
                                    0.0,
                                    agent,
                                    tracker,
                                    eval_result,
                                    best_eval_rate,
                                    start_time,
                                    process_start_cpu,
                                ),
                            )
                    break
            if (
                config.precision_recovery_enabled
                and curriculum_complete
                and total_steps >= config.max_env_steps
                and not evaluation_improved
            ):
                if recovery_active and total_steps >= recovery_end_step:
                    if best_weights is not None and best_optimizer_state is not None:
                        restore_agent_weights(agent, best_weights)
                        restore_optimizer_state(agent, best_optimizer_state)
                        reset_training_batch()
                    recovery_active = False
                    recovery_end_step = 0
                    plateau_evals = 0
                    if logger.enabled:
                        logger.log(
                            "precision_recovery",
                            phase="end",
                            outcome="rollback",
                            total_steps=total_steps,
                            recovery_count=recovery_count,
                            best_greedy_solve_rate=best_eval_rate,
                        )
                elif not recovery_active:
                    plateau_evals += 1
                    if (
                        plateau_evals >= config.precision_plateau_evals
                        and best_weights is not None
                        and best_optimizer_state is not None
                    ):
                        restore_agent_weights(agent, best_weights)
                        restore_optimizer_state(agent, best_optimizer_state)
                        reset_training_batch()
                        plateau_evals = 0
                        recovery_active = True
                        recovery_end_step = (
                            total_steps + config.precision_recovery_steps
                        )
                        recovery_count += 1
                        if logger.enabled:
                            logger.log(
                                "precision_recovery",
                                phase="start",
                                outcome="plateau",
                                total_steps=total_steps,
                                recovery_count=recovery_count,
                                recovery_end_step=recovery_end_step,
                                learning_rate=(
                                    config.learning_rate
                                    * config.precision_recovery_lr_fraction
                                ),
                                best_greedy_solve_rate=best_eval_rate,
                            )
                _update_training_state(
                    agent,
                    completed,
                    total_steps,
                    total_steps,
                    train_rng,
                    curriculum=sampler.curriculum,
                    hard_sampler=sampler,
                    precision_recovery=precision_recovery_payload(),
                )
                agent.training_state["target_confirmed"] = False
        last_dashboard_episode = _maybe_draw_dashboard(
            dashboard,
            completed,
            total_steps,
            0.0,
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
        hard_sampler=sampler,
        precision_recovery=precision_recovery_payload(),
    )
    agent.training_state["target_confirmed"] = target_confirmed
    prefetcher.close()
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
@dataclass(frozen=True, slots=True)
class InferenceLayout:
    """Pixel geometry for a responsive inference frame."""

    maze_rect: Rect
    hud_rect: Rect
    cell_size: int


def inference_layout(
    window_size: tuple[int, int],
    maze_shape: tuple[int, int],
) -> InferenceLayout:
    """Fit a centered, square-celled maze above a two-line status HUD."""

    width, height = (max(1, int(value)) for value in window_size)
    rows, cols = (int(value) for value in maze_shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("maze dimensions must be positive")

    margin = max(4, min(16, min(width, height) // 30))
    hud_height = max(44, min(68, height // 7))
    hud_y = max(margin, height - hud_height - margin)
    available_width = max(1, width - margin * 2)
    available_height = max(1, hud_y - margin * 2)
    cell_size = max(
        1,
        min(80, available_width // cols, available_height // rows),
    )
    maze_width = cols * cell_size
    maze_height = rows * cell_size
    maze_x = max(0, (width - maze_width) // 2)
    maze_y = margin + max(0, (available_height - maze_height) // 2)
    return InferenceLayout(
        maze_rect=(maze_x, maze_y, maze_width, maze_height),
        hud_rect=(margin, hud_y, max(1, width - margin * 2), hud_height),
        cell_size=cell_size,
    )


def local_observation_bounds(
    position: tuple[int, int],
    view_size: int,
    maze_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Return clipped, exclusive grid bounds for a centered local observation."""

    if view_size <= 0 or view_size % 2 == 0:
        raise ValueError("view_size must be a positive odd integer")
    rows, cols = maze_shape
    half = view_size // 2
    top = max(0, position[0] - half)
    left = max(0, position[1] - half)
    bottom = min(rows, position[0] + half + 1)
    right = min(cols, position[1] + half + 1)
    return top, left, bottom, right


def _draw_start_icon(pg, screen, cell_x: int, cell_y: int, cell_size: int) -> None:
    """Draw a scalable start marker that remains visible under the mouse."""

    cx = cell_x + cell_size // 2
    cy = cell_y + cell_size // 2
    radius = max(1, cell_size * 2 // 5)
    outline = max(1, cell_size // 18)
    pg.draw.circle(screen, (50, 112, 72), (cx, cy), radius)
    pg.draw.circle(screen, (126, 221, 148), (cx, cy), max(1, radius - outline))
    if cell_size < 10:
        return
    pole_x = cx - cell_size // 8
    pole_top = cy - cell_size // 4
    pole_bottom = cy + cell_size // 4
    pg.draw.line(
        screen,
        (244, 249, 242),
        (pole_x, pole_top),
        (pole_x, pole_bottom),
        max(1, cell_size // 14),
    )
    flag = [
        (pole_x, pole_top),
        (pole_x + cell_size // 3, pole_top + cell_size // 10),
        (pole_x, cy),
    ]
    pg.draw.polygon(screen, (250, 253, 248), flag)


def _draw_mouse_icon(pg, screen, center_x: int, center_y: int, cell_size: int) -> None:
    cx = center_x + cell_size // 2
    cy = center_y + cell_size // 2
    outline = max(1, cell_size // 18)
    fur = (174, 181, 190)
    dark_fur = (75, 82, 91)
    pink = (235, 157, 170)
    if cell_size < 12:
        pg.draw.circle(screen, dark_fur, (cx, cy), max(1, cell_size // 3))
        pg.draw.circle(screen, fur, (cx, cy), max(1, cell_size // 3 - 1))
        return

    tail_rect = (
        center_x + cell_size // 16,
        center_y + cell_size // 3,
        cell_size // 2,
        cell_size // 2,
    )
    pg.draw.arc(screen, pink, tail_rect, math.pi * 0.7, math.pi * 1.8, outline)
    pg.draw.line(
        screen,
        pink,
        (center_x + cell_size // 5, cy + cell_size // 5),
        (center_x + cell_size // 3, cy + cell_size // 4),
        outline,
    )
    body_rect = (
        center_x + cell_size // 5,
        center_y + cell_size * 2 // 5,
        cell_size * 3 // 5,
        cell_size * 2 // 5,
    )
    pg.draw.ellipse(screen, dark_fur, body_rect)
    inner_body = (
        body_rect[0] + outline,
        body_rect[1] + outline,
        max(1, body_rect[2] - outline * 2),
        max(1, body_rect[3] - outline * 2),
    )
    pg.draw.ellipse(screen, fur, inner_body)

    head_center = (cx + cell_size // 7, cy - cell_size // 8)
    head_radius = max(2, cell_size // 4)
    ear_radius = max(2, cell_size // 8)
    for ear_center in (
        (head_center[0] - cell_size // 8, head_center[1] - cell_size // 5),
        (head_center[0] + cell_size // 8, head_center[1] - cell_size // 5),
    ):
        pg.draw.circle(screen, dark_fur, ear_center, ear_radius)
        pg.draw.circle(screen, pink, ear_center, max(1, ear_radius - outline))
    pg.draw.circle(screen, dark_fur, head_center, head_radius)
    pg.draw.circle(screen, fur, head_center, max(1, head_radius - outline))

    eye = (head_center[0] + cell_size // 10, head_center[1] - cell_size // 16)
    nose = (head_center[0] + head_radius - outline, head_center[1] + cell_size // 16)
    pg.draw.circle(screen, (25, 28, 32), eye, max(1, cell_size // 24))
    pg.draw.circle(screen, (92, 44, 55), nose, max(1, cell_size // 22))
    whisker_length = cell_size // 5
    for offset in (-cell_size // 16, cell_size // 16):
        pg.draw.line(
            screen,
            (75, 82, 91),
            (nose[0] - outline, nose[1] + offset),
            (nose[0] + whisker_length, nose[1] + offset * 2),
            1,
        )


def _draw_cheese_icon(pg, screen, center_x: int, center_y: int, cell_size: int) -> None:
    cx = center_x + cell_size // 2
    cy = center_y + cell_size // 2
    half = max(1, cell_size * 3 // 8)
    outline = max(1, cell_size // 18)
    points = [
        (cx - half, cy + half // 2),
        (cx + half, cy + half // 2),
        (cx + half // 2, cy - half),
    ]
    pg.draw.polygon(screen, (127, 83, 12), points)
    if cell_size < 8:
        return
    inner = [
        (cx - half + outline, cy + half // 2 - outline),
        (cx + half - outline, cy + half // 2 - outline),
        (cx + half // 2, cy - half + outline * 2),
    ]
    pg.draw.polygon(screen, (255, 201, 45), inner)
    rind_y = cy + half // 3
    pg.draw.line(
        screen,
        (226, 145, 20),
        (cx - half + outline, rind_y),
        (cx + half - outline, rind_y),
        max(1, cell_size // 12),
    )
    hole_radius = max(1, cell_size // 12)
    for hole_x, hole_y, scale in (
        (cx, cy - cell_size // 8, 1),
        (cx + cell_size // 5, cy + cell_size // 10, 1),
        (cx - cell_size // 5, cy + cell_size // 8, 0),
    ):
        pg.draw.circle(
            screen,
            (202, 128, 12),
            (hole_x, hole_y),
            max(1, hole_radius - scale),
        )


def _initial_inference_window(pg, rows: int, cols: int) -> tuple[int, int]:
    """Choose a useful initial size while staying within the active display."""

    info = pg.display.Info()
    display_width = info.current_w if info.current_w > 0 else 1280
    display_height = info.current_h if info.current_h > 0 else 720
    target_cell = max(16, min(48, (display_height // 2 - 96) // max(rows, 1)))
    width = max(360, cols * target_cell + 32)
    height = max(260, rows * target_cell + 100)
    return min(width, max(320, display_width - 80)), min(
        height,
        max(240, display_height - 80),
    )


def _draw_local_observation_highlight(
    pg,
    screen,
    layout: InferenceLayout,
    position: tuple[int, int],
    view_size: int,
    maze_shape: tuple[int, int],
) -> None:
    """Dim cells outside the policy's current local observation footprint."""

    maze_x, maze_y, maze_width, maze_height = layout.maze_rect
    cell_size = layout.cell_size
    top, left, bottom, right = local_observation_bounds(
        position,
        view_size,
        maze_shape,
    )
    visible_rect = (
        left * cell_size,
        top * cell_size,
        (right - left) * cell_size,
        (bottom - top) * cell_size,
    )
    overlay = pg.Surface((maze_width, maze_height), pg.SRCALPHA)
    overlay.fill((12, 17, 23, 168))
    overlay.fill((0, 0, 0, 0), visible_rect)
    screen.blit(overlay, (maze_x, maze_y))
    border_rect = (
        maze_x + visible_rect[0],
        maze_y + visible_rect[1],
        visible_rect[2],
        visible_rect[3],
    )
    pg.draw.rect(
        screen,
        (75, 167, 232),
        border_rect,
        width=max(1, min(4, cell_size // 8)),
    )


def _fitted_font(pg, text: str, maximum_width: int, preferred_size: int):
    """Return a readable font whose rendered text fits the HUD width."""

    for size in range(preferred_size, 9, -1):
        font = pg.font.SysFont("arial", size)
        if font.size(text)[0] <= maximum_width:
            return font
    return pg.font.SysFont("arial", 10)


def visualize_inference(
    agent: Agent | Planner,
    maze_grid: np.ndarray,
    fps: int = 5,
    observation_mode: str | None = None,
    config: TrainConfig | None = None,
) -> bool:
    """Render one maze and return whether it completed without window closure."""

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
    rows, cols = env.grid.shape
    window_size = _initial_inference_window(pygame, rows, cols)
    screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
    pygame.display.set_caption("MouseMaze Inference")
    clock = pygame.time.Clock()

    trail = [env.start]
    state = env.reset()
    action_mask = env.valid_action_mask()
    recurrent_hidden = (
        agent.initial_policy_state(1) if isinstance(agent, RecurrentPPOAgent) else None
    )
    previous_action = -1
    previous_reward = 0.0
    episode_start = True
    action, q_vals, recurrent_hidden = _inference_action(
        agent,
        state,
        action_mask,
        recurrent_hidden,
        previous_action,
        previous_reward,
        episode_start,
    )
    done = False
    last_blocked = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.quit()
                return False
            if event.type == pygame.VIDEORESIZE:
                resized = (max(320, event.w), max(240, event.h))
                screen = pygame.display.set_mode(resized, pygame.RESIZABLE)

        window_w, window_h = screen.get_size()
        layout = inference_layout((window_w, window_h), (rows, cols))
        maze_x, maze_y, maze_width, maze_height = layout.maze_rect
        cell_size = layout.cell_size
        screen.fill((26, 31, 38))
        pygame.draw.rect(
            screen,
            (239, 242, 237),
            layout.maze_rect,
            border_radius=max(0, min(7, cell_size // 5)),
        )
        for r in range(rows):
            for c in range(cols):
                cell_rect = (
                    maze_x + c * cell_size,
                    maze_y + r * cell_size,
                    cell_size,
                    cell_size,
                )
                if env.grid[r, c] == 1:
                    pygame.draw.rect(
                        screen,
                        (25, 28, 32),
                        cell_rect,
                    )
                elif cell_size >= 14:
                    pygame.draw.rect(screen, (218, 224, 217), cell_rect, width=1)

        _draw_start_icon(
            pygame,
            screen,
            maze_x + env.start[1] * cell_size,
            maze_y + env.start[0] * cell_size,
            cell_size,
        )
        _draw_cheese_icon(
            pygame,
            screen,
            maze_x + env.goal[1] * cell_size,
            maze_y + env.goal[0] * cell_size,
            cell_size,
        )

        inset = max(1, cell_size // 4)
        for r, c in trail[:-1]:
            trail_rect = (
                maze_x + c * cell_size + inset,
                maze_y + r * cell_size + inset,
                max(1, cell_size - 2 * inset),
                max(1, cell_size - 2 * inset),
            )
            pygame.draw.ellipse(
                screen,
                (91, 142, 222),
                trail_rect,
                max(1, cell_size // 24),
            )

        _draw_mouse_icon(
            pygame,
            screen,
            maze_x + env.current_position[1] * cell_size,
            maze_y + env.current_position[0] * cell_size,
            cell_size,
        )
        if mode == "local":
            _draw_local_observation_highlight(
                pygame,
                screen,
                layout,
                env.current_position,
                env.view_size,
                env.grid.shape,
            )

        hud_x, hud_y, hud_width, hud_height = layout.hud_rect
        pygame.draw.rect(
            screen,
            (231, 235, 240),
            layout.hud_rect,
            border_radius=max(3, min(8, hud_height // 7)),
        )
        action_name, arrow = Maze.ACTION_NAMES[action]
        blocked = " blocked" if last_blocked else ""
        outcome = ""
        if done:
            outcome = " | SOLVED" if env.current_position == env.goal else " | TIMEOUT"
        view_status = f"{mode} observation"
        if mode == "local":
            view_status += f" ({env.view_size}x{env.view_size} highlighted)"
        status_line = f"Steps {env.steps:>3} | {view_status}{outcome}"
        if q_vals is None:
            action_line = f"Planner action: {action_name} {arrow}{blocked}"
        else:
            masked_q_vals = q_vals.copy()
            masked_q_vals[~action_mask] = -np.inf
            best_action = int(np.argmax(masked_q_vals))
            best_name, best_arrow = Maze.ACTION_NAMES[best_action]
            action_line = (
                f"Last action: {action_name} {arrow}{blocked} | "
                f"Q-max: {best_name} {best_arrow} ({q_vals[best_action]:.2f})"
            )
        text_width = max(1, hud_width - 20)
        preferred_font_size = max(12, min(17, hud_height // 3))
        status_font = _fitted_font(
            pygame,
            status_line,
            text_width,
            preferred_font_size,
        )
        action_font = _fitted_font(
            pygame,
            action_line,
            text_width,
            preferred_font_size,
        )
        line_gap = max(1, hud_height // 14)
        total_text_height = status_font.get_linesize() + action_font.get_linesize() + line_gap
        text_y = hud_y + max(3, (hud_height - total_text_height) // 2)
        screen.blit(
            status_font.render(status_line, True, (30, 35, 42)),
            (hud_x + 10, text_y),
        )
        screen.blit(
            action_font.render(action_line, True, (72, 82, 94)),
            (hud_x + 10, text_y + status_font.get_linesize() + line_gap),
        )
        pygame.display.flip()
        clock.tick(fps)

        if done:
            break

        trail.append(env.current_position)
        next_state, step_reward, done, step_info = env.step(action)
        last_blocked = not bool(step_info["moved"])
        state = next_state
        action_mask = env.valid_action_mask()
        previous_action = action
        previous_reward = step_reward
        episode_start = False
        action, q_vals, recurrent_hidden = _inference_action(
            agent,
            state,
            action_mask,
            recurrent_hidden,
            previous_action,
            previous_reward,
            episode_start,
        )

    label = "SOLVED" if env.current_position == env.goal else "TIMEOUT"
    print(f"Steps: {env.steps} -- {label}")
    pygame.quit()
    return True


def _inference_action(
    policy: Agent | Planner,
    state: np.ndarray,
    action_mask: np.ndarray,
    recurrent_hidden: torch.Tensor | None = None,
    previous_action: int = -1,
    previous_reward: float = 0.0,
    episode_start: bool = True,
) -> tuple[int, np.ndarray | None, torch.Tensor | None]:
    """Choose a legal action and optionally return learned action values."""

    if isinstance(policy, RecurrentPPOAgent):
        if recurrent_hidden is None:
            recurrent_hidden = policy.initial_policy_state(1)
        actions, next_hidden = policy.get_actions_stateful(
            state[np.newaxis],
            action_mask[np.newaxis],
            np.array([previous_action], dtype=np.int64),
            np.array([previous_reward], dtype=np.float32),
            np.array([episode_start], dtype=np.bool_),
            recurrent_hidden,
        )
        return int(actions[0]), None, next_hidden
    q_values = getattr(policy, "q_values", None)
    if callable(q_values):
        values = q_values(state)
        masked_values = values.copy()
        masked_values[~action_mask] = -np.inf
        return int(np.argmax(masked_values)), values, recurrent_hidden
    actions = policy.get_actions(state, action_masks=action_mask)
    return int(actions[0]), None, recurrent_hidden


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _inference_count(value: str) -> int:
    """Parse an inference count, accepting zero or ``infinite`` for no limit."""

    if value.lower() in {"infinite", "inf", "forever"}:
        return 0
    try:
        count = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "inference maze count must be a non-negative integer or 'infinite'"
        ) from error
    if count < 0:
        raise argparse.ArgumentTypeError("inference maze count must be non-negative")
    return count


def run_inference_loop(
    agent: Agent | Planner,
    config: TrainConfig,
    maze_count: int = DEFAULT_INFERENCE_MAZES,
) -> int:
    """Run inference on fresh mazes until a count or window close stops it.

    A ``maze_count`` of zero means that new mazes are generated indefinitely.
    The returned value is the number of mazes that were started.
    """

    if maze_count < 0:
        raise ValueError("maze_count must be non-negative")

    completed = 0
    while maze_count == 0 or completed < maze_count:
        completed += 1
        maze_grid = make_maze(config).grid
        limit = "infinite" if maze_count == 0 else str(maze_count)
        print(f"Inference on fresh maze {completed}/{limit}:")
        if not visualize_inference(
            agent,
            maze_grid.copy(),
            observation_mode=config.observation_mode,
            config=config,
        ):
            break
    return completed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train recurrent PPO or DQN agents on MouseMaze."
    )
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--max-env-steps", type=int, default=None)
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
        help=(
            "Resume the selected/latest checkpoint (default); --no-resume "
            "starts a fresh timestamped experiment."
        ),
    )
    parser.add_argument(
        "--target-only-stop",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "For recurrent PPO, ignore episode and transition caps until the "
            "target is confirmed on independent suites after curriculum completion."
        ),
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
    parser.add_argument(
        "--num-envs",
        type=int,
        default=None,
        help=(
            "Parallel environments; defaults to 512 for recurrent PPO on the "
            "RTX 3090 fast profile and 256 otherwise."
        ),
    )
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
    parser.add_argument("--ppo-target-kl", type=float, default=None)
    parser.add_argument("--ppo-value-clip-range", type=float, default=None)
    parser.add_argument("--recurrent-hidden-size", type=int, default=None)
    parser.add_argument("--recurrent-sequence-length", type=int, default=None)
    parser.add_argument("--recurrent-sequence-minibatch-size", type=int, default=None)
    parser.add_argument("--rnd-reward-coef", type=float, default=None)
    parser.add_argument("--rnd-reward-clip", type=float, default=None)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--dashboard-every", type=int, default=None)
    parser.add_argument("--curriculum-promotion-rate", type=float, default=None)
    parser.add_argument("--curriculum-promotion-evals", type=int, default=None)
    parser.add_argument("--curriculum-previous-fraction", type=float, default=None)
    parser.add_argument("--curriculum-uniform-fraction", type=float, default=None)
    parser.add_argument("--curriculum-eval-episodes", type=int, default=None)
    parser.add_argument(
        "--hard-maze-fraction",
        type=float,
        default=None,
        help=(
            "Opt-in maximum fraction of recurrent tasks sampled from hard variants; "
            "ramps from zero at 90%% validation solve rate to full at 99%%."
        ),
    )
    parser.add_argument(
        "--hard-maze-pool-size",
        type=int,
        default=None,
        help="Maximum transformed evaluation failures retained for hard replay.",
    )
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
    parser.add_argument("--target-solve-rate", type=float, default=None)
    parser.add_argument(
        "--target-solve-evals",
        type=int,
        default=None,
        help=(
            "Number of deterministic, seed-separated suites that one frozen "
            "policy must pass before target-only training stops."
        ),
    )
    parser.add_argument(
        "--precision-recovery",
        dest="precision_recovery_enabled",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Opt in to rollback-based post-budget precision recovery.",
    )
    parser.add_argument(
        "--precision-plateau-evals",
        type=int,
        default=None,
        help="Post-budget evaluations without improvement before recovery.",
    )
    parser.add_argument(
        "--precision-recovery-steps",
        type=int,
        default=None,
        help="Transitions trained in one frozen-best recovery window.",
    )
    parser.add_argument(
        "--precision-recovery-lr-fraction",
        type=float,
        default=None,
        help="Fraction of the initial learning rate used during recovery.",
    )
    parser.add_argument(
        "--latest-checkpoint-every-steps",
        type=int,
        default=None,
        help="Transition cadence for the resumable .latest.pth sidecar.",
    )
    parser.add_argument(
        "--performance-profile",
        choices=PERFORMANCE_PROFILES,
        default=None,
    )
    parser.add_argument("--maze-workers", type=int, default=None)
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
    parser.add_argument(
        "--inference-mazes",
        type=_inference_count,
        default=DEFAULT_INFERENCE_MAZES,
        metavar="COUNT",
        help=(
            "Number of fresh mazes to render during inference; use 0 or "
            "'infinite' to continue until the window is closed."
        ),
    )
    return parser.parse_args(argv)


def _train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    if (
        args.train_updates_per_step is not None
        and args.updates_per_transition is not None
    ):
        raise ValueError(
            "--train-updates-per-step and --updates-per-transition cannot be "
            "used together"
        )
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
        "max_env_steps",
        "seed",
        "eval_seed",
        "algorithm",
        "network_type",
        "resume",
        "target_only_stop",
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
        "epsilon_decay_steps",
        "epsilon_final_steps",
        "num_envs",
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
        "ppo_target_kl",
        "ppo_value_clip_range",
        "recurrent_hidden_size",
        "recurrent_sequence_length",
        "recurrent_sequence_minibatch_size",
        "rnd_reward_coef",
        "rnd_reward_clip",
        "eval_episodes",
        "eval_every_steps",
        "dashboard_every",
        "curriculum_promotion_rate",
        "curriculum_promotion_evals",
        "curriculum_previous_fraction",
        "curriculum_uniform_fraction",
        "curriculum_eval_episodes",
        "hard_maze_fraction",
        "hard_maze_pool_size",
        "vectorized_envs",
        "checkpoint_replay",
        "expert_pretrain_mazes",
        "expert_pretrain_epochs",
        "target_solve_rate",
        "target_solve_evals",
        "precision_plateau_evals",
        "precision_recovery_steps",
        "precision_recovery_lr_fraction",
        "precision_recovery_enabled",
        "latest_checkpoint_every_steps",
        "performance_profile",
        "maze_workers",
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
    if args.target_only_stop is None and args.algorithm is not None:
        values["target_only_stop"] = None
    return TrainConfig(**values)


def _apply_legacy_cli_aliases(
    config: TrainConfig,
    args: argparse.Namespace,
) -> TrainConfig:
    """Resolve aliases that depend on finalized runtime configuration."""

    if args.train_updates_per_step is None:
        return config
    if args.updates_per_transition is not None:
        raise ValueError(
            "--train-updates-per-step and --updates-per-transition cannot be "
            "used together"
        )
    resolved_num_envs = resolve_num_envs(config)
    values = {field_.name: getattr(config, field_.name) for field_ in fields(config)}
    values["num_envs"] = resolved_num_envs
    values["updates_per_transition"] = (
        args.train_updates_per_step / resolved_num_envs
    )
    return TrainConfig(**values)


def _optional_bool(value: bool | None, default: bool) -> bool:
    return default if value is None else value


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = _train_config_from_args(args)
    run_timestamp = invocation_timestamp()
    config = resolve_cli_artifacts(config, args, run_timestamp)
    device = select_device(config.device, config.require_cuda)
    config.performance_profile = configure_performance(config.performance_profile, device)
    config.num_envs = resolve_num_envs(config)
    config = _apply_legacy_cli_aliases(config, args)
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
        train(agent=agent, config=config, run_timestamp=run_timestamp)
    elif config.save_path and os.path.exists(config.save_path):
        checkpoint_algo = checkpoint_algorithm(config.save_path)
        if checkpoint_algo != agent.algorithm:
            values = {field_.name: getattr(config, field_.name) for field_ in fields(config)}
            values["algorithm"] = checkpoint_algo
            config = TrainConfig(**values)
            agent = create_agent(config, expected_shape, device)
        agent.load(config.save_path)

    if _optional_bool(args.infer_flag, DEFAULT_INFER_FLAG):
        inference_policy: Agent | Planner = BfsPlanner() if config.planner_fallback else agent
        run_inference_loop(inference_policy, config, args.inference_mazes)


if __name__ == "__main__":
    main()
