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
import queue
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
from functools import lru_cache
from typing import Any, Callable, Protocol

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from gen_maze import generate_random_maze


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
# Size of the centered local observation window; it must remain odd.
VIEW_SIZE = 5
# Rows and columns used when no maze size is supplied on the command line.
DEFAULT_MAZE_SIZE = (21, 21)
# Observation choices: the full map or a centered local window.
OBSERVATION_MODES = ("full", "local")
# Algorithms supported by the trainer and checkpoint loader.
ALGORITHMS = ("dqn", "ppo", "recurrent_ppo")
# Q-network layouts: spatial convolutions or a flattened MLP input.
NETWORK_TYPES = ("spatial", "flat")
# Reward-shaping choices for distance-to-goal progress.
DISTANCE_SHAPING_MODES = ("progress", "potential", "fractional", "none")
VISIT_COUNT_ENCODINGS = ("clipped", "episode_log")
CONTINUOUS_DISTANCE_MODES = ("graph", "start_path")
ACTION_SPACES = ("discrete", "continuous")
# Hardware/reproducibility tuning profiles exposed by the CLI.
PERFORMANCE_PROFILES = ("auto", "rtx3090-fast", "strict", "portable")
CURRICULUM_MODES = ("auto", "manual")
# Checkpoint format version written by the current agent implementation.
CHECKPOINT_SCHEMA_VERSION = 11
SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS = (10, CHECKPOINT_SCHEMA_VERSION)
# Compatibility episode limit used when training is enabled without an override.
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
DEFAULT_RESUME = False
# Whether recurrent PPO ignores episode and transition caps until it reaches its target.
DEFAULT_TARGET_ONLY_STOP: bool | None = False

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
N_STEP_RETURNS = 10
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
RTX3090_RECURRENT_NUM_ENVS = 768
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
# Optional hard cap on steps in one maze episode; None uses the resolved budget.
MAX_EPISODE_STEPS: int | None = None
# Multiplier used to provide a maze-size-aware recovery budget when enabled.
DEFAULT_TIMEOUT_STEP_FACTOR = 4.0
# Minimum episode length allowing local agents to recover from short wrong turns.
DEFAULT_MIN_EPISODE_STEPS = 20
# Complete depth-first traversal allowance for partially observed mazes.
DEFAULT_EXPLORATION_STEP_FACTOR = 2.0
# Number of episodes in each standard evaluation pass during training.
NUM_EVAL_EPISODES = 500
# Episode-based dashboard/evaluation cadence compatibility default.
EVAL_PERIOD = 100
# Transition interval between evaluation passes during training.
EVAL_PERIOD_STEPS = 50_000
# Less frequent final-distribution evaluation after curriculum completion.
DEFAULT_POST_CURRICULUM_EVAL_STEPS = 500_000
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
# Action space used when no CLI override is provided.
DEFAULT_ACTION_SPACE = "continuous"
# Launching the script without CLI flags starts a fresh training run.
DEFAULT_TRAIN_FLAG = False
# Inference is enabled by default after optional training/checkpoint loading.
DEFAULT_INFER_FLAG = True
# Number of fresh mazes rendered for inference; zero means run indefinitely.
DEFAULT_INFERENCE_MAZES = 0
# Maximum inference frames rendered per second.
DEFAULT_INFERENCE_FPS = 30
# Whether inference initially shows the model's spatial input channels.
DEFAULT_SHOW_INPUT_CHANNELS = True
# Whether observations include the normalized remaining-episode-time channel.
DEFAULT_REMAINING_TIME_CHANNEL = False
# Whether observations include a normalized per-cell visit-count channel.
DEFAULT_VISIT_COUNT_CHANNEL = True
# Visit count represented as 1.0 after this many visits.
DEFAULT_VISIT_COUNT_CLIP = 5
# Whether local observations hide cells behind walls.
DEFAULT_WALL_OCCLUSION = True

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, "models")
LOG_RESULTS_DIR = os.path.join(RESULTS_DIR, "logs")
ARTIFACT_BASENAME = "mousemaze"

# Small per-step reward encouraging shorter solutions.
STEP_PENALTY = -0.01
# Additional penalty for attempting to move into a wall or outside the maze.
INVALID_MOVE_PENALTY = -0.2
# Reward granted when the mouse reaches the cheese.
GOAL_REWARD = 10.0
# Penalty applied when an episode times out.
TIMEOUT_PENALTY = -2.0
# Multiplier applied to the selected distance-shaping reward.
DISTANCE_SHAPING_SCALE = 1.0
# Distance-shaping strategy used unless explicitly overridden.
DEFAULT_DISTANCE_SHAPING_MODE = "progress"

# Whether curriculum sampling changes maze difficulty during training.
DEFAULT_CURRICULUM_ENABLED = True
# Empirical size-and-complexity curriculum used by new runs.
DEFAULT_CURRICULUM_MODE = "auto"
# Candidate mazes used to resolve deterministic complexity quantiles per size.
DEFAULT_CURRICULUM_PROBE_MAZES = 2_048
# Odd-cell dimension increment used to construct the automatic size ladder.
DEFAULT_CURRICULUM_SIZE_STEP = 4
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
DEFAULT_CURRICULUM_PROMOTION_RATE = 0.70
# Number of validation evaluations required for a curriculum promotion.
DEFAULT_CURRICULUM_PROMOTION_EVALS = 3
# Fraction of curriculum tasks sampled from previously successful stages.
DEFAULT_CURRICULUM_PREVIOUS_FRACTION = 0.20
# Fraction of curriculum tasks sampled uniformly across available stages.
DEFAULT_CURRICULUM_UNIFORM_FRACTION = 0.10
# Fraction of the finite budget guaranteed to the unrestricted final stage.
DEFAULT_CURRICULUM_FINAL_STAGE_FRACTION = 0.20
# Fraction of training tasks drawn from timed-out training mazes.
DEFAULT_HARD_MAZE_FRACTION = 0.05
# Maximum number of transformed hard-maze variants retained per grid shape.
DEFAULT_HARD_MAZE_POOL_SIZE = 256
# Validation range over which hard replay ramps to its configured maximum.
HARD_MAZE_RAMP_START = 0.60
HARD_MAZE_RAMP_END = 0.80
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
DEFAULT_LOCAL_MAX_ENV_STEPS = 100_000_000
# Target validation solve rate for full-map training.
DEFAULT_FULL_TARGET_SOLVE_RATE = 1.00
# Target validation solve rate for local-observation training.
DEFAULT_LOCAL_TARGET_SOLVE_RATE = 1.00
# Number of deterministic suites used to confirm one frozen target candidate.
DEFAULT_TARGET_SOLVE_EVALS = 3
# Final-stage evaluations without improvement before a guarded recovery pass.
DEFAULT_PRECISION_PLATEAU_EVALS = 20
# Transitions in one guarded precision-recovery pass.
DEFAULT_PRECISION_RECOVERY_STEPS = 1_000_000
# Fraction of the initial learning rate used during precision recovery.
DEFAULT_PRECISION_RECOVERY_LR_FRACTION = 0.05
# Whether rollback-based precision recovery is active.
DEFAULT_PRECISION_RECOVERY_ENABLED = True
# Transition cadence for the resumable latest-training sidecar.
DEFAULT_LATEST_CHECKPOINT_EVERY_STEPS = 1_000_000
# Number of mazes used to evaluate a curriculum stage.
DEFAULT_CURRICULUM_EVAL_EPISODES = 200
# Worker processes used for deterministic background maze generation.
DEFAULT_MAZE_WORKERS = 8
# Maze grids returned by one process-pool future.
DEFAULT_MAZE_GENERATION_BATCH_SIZE = 64
# Submitted generation batches retained per worker.
DEFAULT_MAZE_PREFETCH_BATCHES_PER_WORKER = 4

# Number of transitions collected in one PPO rollout.
PPO_ROLLOUT_STEPS = 256
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
# Epoch-mean approximate-KL threshold that can skip later PPO epochs.
PPO_TARGET_KL = 0.025
# Clipping threshold for PPO value predictions.
PPO_VALUE_CLIP_RANGE = 0.2
# Bounds applied to the state-dependent continuous-action log standard deviation.
CONTINUOUS_LOG_STD_MIN = -2.5
CONTINUOUS_LOG_STD_MAX = 2.0
# Size of the recurrent GRU hidden state.
RECURRENT_HIDDEN_SIZE = 512
# Number of consecutive steps in a recurrent PPO training sequence.
RECURRENT_SEQUENCE_LENGTH = 128
# Number of recurrent sequences processed in one PPO minibatch.
RECURRENT_SEQUENCE_MINIBATCH_SIZE = 16
# Coefficient for random-network-distillation intrinsic reward.
RND_REWARD_COEF = 0.05
# Maximum absolute intrinsic reward contribution from RND.
RND_REWARD_CLIP = 5.0
# Fraction of the finite budget reserved for precision annealing.
DEFAULT_PRECISION_FRACTION = 0.20
# Final learning-rate multiplier relative to the configured initial rate.
DEFAULT_PRECISION_LEARNING_RATE_FRACTION = 0.01
# Final entropy multiplier relative to the configured initial coefficient.
DEFAULT_PRECISION_ENTROPY_FRACTION = 0.10

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
    action_space: str | None = None
    continuous_step_scale: float = 0.25
    continuous_distance_mode: str | None = None
    view_size: int = VIEW_SIZE
    remaining_time_channel: bool = DEFAULT_REMAINING_TIME_CHANNEL
    visit_count_channel: bool = DEFAULT_VISIT_COUNT_CHANNEL
    visit_count_clip: int = DEFAULT_VISIT_COUNT_CLIP
    visit_count_encoding: str | None = None
    wall_occlusion: bool = DEFAULT_WALL_OCCLUSION
    max_episode_steps: int | None = MAX_EPISODE_STEPS
    timeout_step_factor: float | None = DEFAULT_TIMEOUT_STEP_FACTOR
    min_episode_steps: int = DEFAULT_MIN_EPISODE_STEPS
    exploration_step_factor: float = DEFAULT_EXPLORATION_STEP_FACTOR
    step_penalty: float = STEP_PENALTY
    invalid_move_penalty: float = INVALID_MOVE_PENALTY
    goal_reward: float = GOAL_REWARD
    timeout_penalty: float = TIMEOUT_PENALTY
    distance_shaping_scale: float = DISTANCE_SHAPING_SCALE
    distance_shaping_mode: str = DEFAULT_DISTANCE_SHAPING_MODE
    curriculum_enabled: bool = DEFAULT_CURRICULUM_ENABLED
    curriculum_mode: str = DEFAULT_CURRICULUM_MODE
    curriculum_probe_mazes: int = DEFAULT_CURRICULUM_PROBE_MAZES
    curriculum_size_step: int = DEFAULT_CURRICULUM_SIZE_STEP
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
    continuous_entropy_coef: float = 0.003
    ppo_entropy_floor: float = 0.2
    ppo_max_grad_norm: float = PPO_MAX_GRAD_NORM
    ppo_target_kl: float = PPO_TARGET_KL
    ppo_value_clip_range: float = PPO_VALUE_CLIP_RANGE
    recurrent_hidden_size: int = RECURRENT_HIDDEN_SIZE
    recurrent_sequence_length: int = RECURRENT_SEQUENCE_LENGTH
    recurrent_sequence_minibatch_size: int = RECURRENT_SEQUENCE_MINIBATCH_SIZE
    rnd_reward_coef: float = RND_REWARD_COEF
    rnd_reward_clip: float = RND_REWARD_CLIP
    eval_every_steps: int = EVAL_PERIOD_STEPS
    post_curriculum_eval_every_steps: int = DEFAULT_POST_CURRICULUM_EVAL_STEPS
    eval_episodes: int = NUM_EVAL_EPISODES
    curriculum_promotion_rate: float = DEFAULT_CURRICULUM_PROMOTION_RATE
    curriculum_promotion_evals: int = DEFAULT_CURRICULUM_PROMOTION_EVALS
    curriculum_previous_fraction: float = DEFAULT_CURRICULUM_PREVIOUS_FRACTION
    curriculum_uniform_fraction: float = DEFAULT_CURRICULUM_UNIFORM_FRACTION
    curriculum_final_stage_fraction: float = DEFAULT_CURRICULUM_FINAL_STAGE_FRACTION
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
    precision_fraction: float = DEFAULT_PRECISION_FRACTION
    precision_learning_rate_fraction: float = DEFAULT_PRECISION_LEARNING_RATE_FRACTION
    precision_entropy_fraction: float = DEFAULT_PRECISION_ENTROPY_FRACTION
    precision_phase_min_steps: int = 50_000_000
    latest_checkpoint_every_steps: int = DEFAULT_LATEST_CHECKPOINT_EVERY_STEPS
    performance_profile: str = DEFAULT_PERFORMANCE_PROFILE
    maze_workers: int = DEFAULT_MAZE_WORKERS
    maze_generation_batch_size: int = DEFAULT_MAZE_GENERATION_BATCH_SIZE
    maze_prefetch_batches_per_worker: int = DEFAULT_MAZE_PREFETCH_BATCHES_PER_WORKER
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
        if self.action_space is None:
            self.action_space = (
                DEFAULT_ACTION_SPACE
                if self.algorithm == "recurrent_ppo"
                else "discrete"
            )
        if self.visit_count_encoding is None:
            self.visit_count_encoding = (
                "episode_log" if self.action_space == "continuous" else "clipped"
            )
        if self.continuous_distance_mode is None:
            self.continuous_distance_mode = "graph"
        if self.target_only_stop is None:
            self.target_only_stop = False
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
            if self.observation_mode == "local":
                scaled_budget = (
                    DEFAULT_LOCAL_MAX_ENV_STEPS
                    * max(self.maze_size)
                    / max(DEFAULT_MAZE_SIZE)
                )
                self.max_env_steps = int(round(scaled_budget / 1_000_000)) * 1_000_000
            else:
                self.max_env_steps = DEFAULT_FULL_MAX_ENV_STEPS
        if self.algorithm == "recurrent_ppo" and self.episodes == DEFAULT_EPISODES:
            # Keep the transition budget authoritative even if every episode
            # happens to terminate after a single step.
            self.episodes = max(
                self.episodes,
                self.max_env_steps,
            )
        if self.target_solve_rate is None:
            self.target_solve_rate = (
                DEFAULT_LOCAL_TARGET_SOLVE_RATE
                if self.observation_mode == "local"
                else DEFAULT_FULL_TARGET_SOLVE_RATE
            )
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
        if self.action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space must be one of {ACTION_SPACES}; got {self.action_space!r}"
            )
        if self.action_space == "continuous" and self.algorithm != "recurrent_ppo":
            raise ValueError("continuous action space is supported only by recurrent_ppo")
        if not 0.0 < self.continuous_step_scale <= 1.0:
            raise ValueError("continuous_step_scale must be in (0, 1]")
        if self.continuous_distance_mode not in CONTINUOUS_DISTANCE_MODES:
            raise ValueError(
                "continuous_distance_mode must be one of "
                f"{CONTINUOUS_DISTANCE_MODES}; got {self.continuous_distance_mode!r}"
            )
        if len(self.maze_size) != 2 or min(self.maze_size) < 3:
            raise ValueError("maze_size must contain two dimensions >= 3")
        if any(size % 2 == 0 for size in self.maze_size):
            raise ValueError("maze_size dimensions must be odd for the maze generator")
        if self.view_size < 1 or self.view_size % 2 == 0:
            raise ValueError("view_size must be odd so the agent has a center cell")
        if self.visit_count_clip < 1:
            raise ValueError("visit_count_clip must be >= 1")
        if self.visit_count_encoding not in VISIT_COUNT_ENCODINGS:
            raise ValueError(
                "visit_count_encoding must be one of "
                f"{VISIT_COUNT_ENCODINGS}; got {self.visit_count_encoding!r}"
            )
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
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError("max_episode_steps must be >= 1")
        if self.timeout_step_factor is not None and self.timeout_step_factor <= 0:
            raise ValueError("timeout_step_factor must be positive or None")
        if self.min_episode_steps < 1:
            raise ValueError("min_episode_steps must be >= 1")
        if self.exploration_step_factor <= 0:
            raise ValueError("exploration_step_factor must be positive")
        if self.curriculum_mode not in CURRICULUM_MODES:
            raise ValueError(f"curriculum_mode must be one of {CURRICULUM_MODES}")
        if self.curriculum_probe_mazes < 1:
            raise ValueError("curriculum_probe_mazes must be >= 1")
        if self.curriculum_size_step < 2 or self.curriculum_size_step % 2:
            raise ValueError("curriculum_size_step must be a positive even increment")
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
        if not 0 <= self.curriculum_final_stage_fraction <= 1:
            raise ValueError("curriculum_final_stage_fraction must be in [0, 1]")
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
        if self.post_curriculum_eval_every_steps < 1:
            raise ValueError("post_curriculum_eval_every_steps must be >= 1")
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
        if not 0 < self.precision_fraction < 1:
            raise ValueError("precision_fraction must be in (0, 1)")
        if not 0 < self.precision_learning_rate_fraction <= 0.1:
            raise ValueError("precision_learning_rate_fraction must be in (0, 0.1]")
        if not 0 < self.precision_entropy_fraction <= 1:
            raise ValueError("precision_entropy_fraction must be in (0, 1]")
        if self.continuous_entropy_coef < 0:
            raise ValueError("continuous_entropy_coef must be >= 0")
        if self.precision_phase_min_steps < 0:
            raise ValueError("precision_phase_min_steps must be >= 0")
        if self.latest_checkpoint_every_steps < 1:
            raise ValueError("latest_checkpoint_every_steps must be >= 1")
        if self.maze_workers < 0:
            raise ValueError("maze_workers must be >= 0")
        if self.maze_generation_batch_size < 1:
            raise ValueError("maze_generation_batch_size must be >= 1")
        if self.maze_prefetch_batches_per_worker < 1:
            raise ValueError("maze_prefetch_batches_per_worker must be >= 1")
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
    requested_action_mean: tuple[float, float] | None = None
    requested_action_std: tuple[float, float] | None = None
    executed_displacement_mean: tuple[float, float] | None = None
    executed_displacement_std: tuple[float, float] | None = None
    action_saturation_rate: float = 0.0
    no_motion_rate: float = 0.0
    low_action_rate: float = 0.0
    visit_saturation_rate: float = 0.0
    failed_visit_saturation_rate: float = 0.0
    max_same_cell_dwell_mean: float = 0.0
    failed_max_same_cell_dwell_mean: float = 0.0
    reward_mean: float = 0.0
    distance_shaping_reward_mean: float = 0.0
    collision_cause_rates: dict[str, float] | None = None
    within_cell_offset_mean: tuple[float, float] | None = None
    goal_cell_outside_radius_rate: float = 0.0
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
    continuous_entropy_per_dimension: tuple[float, float] | None = None
    action_std_mean: float | None = None
    action_std_min: float | None = None
    action_std_max: float | None = None


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


@dataclass(slots=True)
class DashboardCharts:
    """Serializable chart data sent to the dashboard process."""

    reward_history: list[ChartPoint]
    loss_history: list[ChartPoint]
    greedy_solve_history: list[ChartPoint]


RawTransition = tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]
ReplayTransition = tuple[np.ndarray, int, float, np.ndarray, bool, np.ndarray]


def _append_chart_point(history: list[ChartPoint], point: ChartPoint) -> None:
    """Append one point while retaining a bounded whole-run representation.

    When *history* exceeds :data:`DASHBOARD_MAX_HISTORY_POINTS` it is reduced by
    collapsing non-overlapping windows into their median value (with the window's
    midpoint episode as the x-co-ordinate).  This keeps roughly half the points so
    that new data can accumulate before the next reduction, and avoids anchoring
    the chart axis at episode 1.
    """

    history.append(point)
    if len(history) <= DASHBOARD_MAX_HISTORY_POINTS:
        return

    target = max(2, DASHBOARD_MAX_HISTORY_POINTS // 2)
    n = len(history)
    window_size = max(1, n // target)

    latest = history[-1]
    collapsed: list[ChartPoint] = []
    for start in range(0, n - 1, window_size):
        end = min(start + window_size, n - 1)
        window = history[start:end]
        median_value = sorted(v for _, v in window)[len(window) // 2]
        mid_episode = (window[0][0] + window[-1][0]) / 2
        collapsed.append((int(mid_episode), median_value))

    collapsed.append(latest)
    history[:] = collapsed


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
    remaining_time_channel: bool = DEFAULT_REMAINING_TIME_CHANNEL,
    visit_count_channel: bool = DEFAULT_VISIT_COUNT_CHANNEL,
    action_space: str = "discrete",
) -> tuple[int, int, int]:
    """Return the channel-first observation shape for a maze configuration."""

    if observation_mode == "full":
        channels = 3 + int(remaining_time_channel) + int(visit_count_channel)
        return (channels, maze_size[0], maze_size[1])
    if observation_mode == "local":
        channels = 2 + int(remaining_time_channel) + int(visit_count_channel)
        return (channels, view_size, view_size)
    raise ValueError(f"unsupported observation_mode: {observation_mode!r}")


def config_observation_shape(config: TrainConfig) -> tuple[int, int, int]:
    """Return the observation shape selected by one training configuration."""

    return observation_shape(
        config.maze_size,
        config.observation_mode,
        config.view_size,
        config.remaining_time_channel,
        config.visit_count_channel,
        config.action_space,
    )


def observation_settings_payload(config: TrainConfig) -> dict[str, object]:
    """Return checkpoint metadata that defines observation semantics."""

    return {
        "remaining_time_channel": config.remaining_time_channel,
        "visit_count_channel": config.visit_count_channel,
        "visit_count_clip": config.visit_count_clip,
        "visit_count_encoding": config.visit_count_encoding,
        "wall_occlusion": config.wall_occlusion,
        "action_space": config.action_space,
        "continuous_features": (
            "separate_within_cell_row_col_and_previous_collision"
            if config.action_space == "continuous"
            else "none"
        ),
        "proprioception_size": 3 if config.action_space == "continuous" else 0,
        "continuous_step_scale": config.continuous_step_scale,
    }


def _validate_checkpoint_observation_settings(
    payload: dict[str, object],
    config: TrainConfig,
    path: str,
) -> None:
    expected = observation_settings_payload(config)
    actual = payload.get("observation_settings")
    if isinstance(actual, dict) and "visit_count_encoding" not in actual:
        actual = {**actual, "visit_count_encoding": "clipped"}
    if actual != expected:
        raise ValueError(
            f"checkpoint {path!r} uses observation settings {actual!r}; "
            f"current configuration uses {expected!r}. Pass matching channel "
            "and wall-occlusion options."
        )


def _validate_checkpoint_reward_settings(
    payload: dict[str, object],
    config: TrainConfig,
    path: str,
) -> None:
    actual = str(payload.get("distance_shaping_mode", "potential"))
    if actual != config.distance_shaping_mode:
        raise ValueError(
            f"checkpoint {path!r} uses distance shaping {actual!r}; current "
            f"configuration uses {config.distance_shaping_mode!r}"
        )
    actual_distance = str(payload.get("continuous_distance_mode", "start_path"))
    if actual_distance != config.continuous_distance_mode:
        raise ValueError(
            f"checkpoint {path!r} uses continuous distance {actual_distance!r}; "
            f"current configuration uses {config.continuous_distance_mode!r}"
        )


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
@lru_cache(maxsize=None)
def _local_visibility_blockers(
    view_size: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return conservative wall blockers for every cell in a local view."""

    half = view_size // 2
    blockers: list[tuple[tuple[int, int], ...]] = []
    for target_row in range(view_size):
        for target_col in range(view_size):
            delta_row = target_row - half
            delta_col = target_col - half
            row_step = 1 if delta_row > 0 else -1
            col_step = 1 if delta_col > 0 else -1
            row_distance = abs(delta_row)
            col_distance = abs(delta_col)
            row_progress = 0
            col_progress = 0
            row = 0
            col = 0
            ray_cells: list[tuple[int, int]] = []
            while row_progress < row_distance or col_progress < col_distance:
                decision = (
                    (1 + 2 * col_progress) * row_distance
                    - (1 + 2 * row_progress) * col_distance
                )
                if decision == 0:
                    side_cells = (
                        (row, col + col_step),
                        (row + row_step, col),
                    )
                    for side_row, side_col in side_cells:
                        if (side_row, side_col) != (delta_row, delta_col):
                            ray_cells.append((side_row + half, side_col + half))
                    row += row_step
                    col += col_step
                    row_progress += 1
                    col_progress += 1
                elif decision < 0:
                    col += col_step
                    col_progress += 1
                else:
                    row += row_step
                    row_progress += 1
                if (row, col) != (delta_row, delta_col):
                    ray_cells.append((row + half, col + half))
            blockers.append(tuple(dict.fromkeys(ray_cells)))
    return tuple(blockers)


def local_visibility_mask(walls: np.ndarray) -> np.ndarray:
    """Return cells visible from the center without looking through walls."""

    if walls.ndim < 2 or walls.shape[-1] != walls.shape[-2]:
        raise ValueError("local wall maps must be square")
    view_size = walls.shape[-1]
    if view_size % 2 == 0:
        raise ValueError("local wall maps must have odd dimensions")
    visible = np.ones(walls.shape, dtype=np.bool_)
    leading_slices = (slice(None),) * (walls.ndim - 2)
    for flat_index, blockers in enumerate(_local_visibility_blockers(view_size)):
        target_row, target_col = divmod(flat_index, view_size)
        if not blockers:
            continue
        blocker_rows, blocker_cols = np.asarray(blockers, dtype=np.int64).T
        blocked = np.any(
            walls[(*leading_slices, blocker_rows, blocker_cols)],
            axis=-1,
        )
        visible[(*leading_slices, target_row, target_col)] = ~blocked
    return visible


CONTINUOUS_GOAL_RADIUS = 0.6
CONTINUOUS_COLLISION_SUBSTEP = 0.1
CONTINUOUS_LOW_ACTION_THRESHOLD = 0.02


def continuous_position_to_cell(
    position: tuple[float, float] | np.ndarray,
    grid_shape: tuple[int, int],
) -> tuple[int, int] | None:
    """Map center-based float coordinates to a containing grid cell."""

    row, col = float(position[0]), float(position[1])
    if not (-0.5 <= row < grid_shape[0] - 0.5 and -0.5 <= col < grid_shape[1] - 0.5):
        return None
    return math.floor(row + 0.5), math.floor(col + 0.5)


def continuous_position_is_solved(
    position: tuple[float, float] | np.ndarray,
    goal: tuple[int, int] | np.ndarray,
) -> bool:
    """Return whether a float position is inside the goal's success radius."""

    return math.hypot(
        float(position[0]) - float(goal[0]),
        float(position[1]) - float(goal[1]),
    ) < CONTINUOUS_GOAL_RADIUS


def continuous_geodesic_distance(
    position: tuple[float, float] | np.ndarray,
    path: list[tuple[int, int]],
) -> float:
    """Return lateral distance plus remaining arclength on an ordered path."""

    if not path:
        return 0.0
    if len(path) == 1:
        return math.hypot(
            float(position[0]) - path[0][0],
            float(position[1]) - path[0][1],
        )
    remaining = [0.0] * len(path)
    for index in range(len(path) - 2, -1, -1):
        remaining[index] = remaining[index + 1] + math.dist(path[index], path[index + 1])
    point = np.asarray(position, dtype=np.float64)
    best = float("inf")
    for index, (start, end) in enumerate(zip(path, path[1:])):
        segment_start = np.asarray(start, dtype=np.float64)
        segment = np.asarray(end, dtype=np.float64) - segment_start
        length_squared = float(np.dot(segment, segment))
        fraction = float(np.dot(point - segment_start, segment) / length_squared)
        fraction = min(max(fraction, 0.0), 1.0)
        projection = segment_start + fraction * segment
        lateral = float(np.linalg.norm(point - projection))
        distance = lateral + (1.0 - fraction) * math.sqrt(length_squared) + remaining[index + 1]
        best = min(best, distance)
    return best


def graph_continuous_goal_distance(
    position: tuple[float, float] | np.ndarray,
    cell: tuple[int, int],
    goal: tuple[int, int],
    bfs_distances: np.ndarray,
    bfs_parent: dict[tuple[int, int], tuple[int, int] | None],
) -> float:
    """Return wall-aware continuous progress through the cell graph."""

    point = np.asarray(position, dtype=np.float64)
    if cell == goal:
        return float(np.linalg.norm(point - np.asarray(goal, dtype=np.float64)))
    successor = bfs_parent.get(cell)
    if successor is None:
        return float("inf")
    return float(
        np.linalg.norm(point - np.asarray(successor, dtype=np.float64))
        + bfs_distances[successor]
    )


class Maze:
    """Grid maze environment with full-map and local observation modes.

    Grid values are 0 for open cells, 1 for walls, 2 for start, and 3 for goal.
    Observations always contain walls, agent, and goal channels. Remaining-time
    and visit-count channels are configurable; local observations can hide cells
    behind walls.
    """

    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # R, L, D, U
    ACTION_NAMES = [("Right", ">"), ("Left", "<"), ("Down", "v"), ("Up", "^")]

    def __init__(
        self,
        grid: np.ndarray,
        observation_mode: str = "full",
        view_size: int = VIEW_SIZE,
        remaining_time_channel: bool = DEFAULT_REMAINING_TIME_CHANNEL,
        visit_count_channel: bool = DEFAULT_VISIT_COUNT_CHANNEL,
        visit_count_clip: int = DEFAULT_VISIT_COUNT_CLIP,
        visit_count_encoding: str | None = None,
        wall_occlusion: bool = DEFAULT_WALL_OCCLUSION,
        max_episode_steps: int | None = MAX_EPISODE_STEPS,
        timeout_step_factor: float | None = None,
        min_episode_steps: int = 1,
        exploration_step_factor: float = DEFAULT_EXPLORATION_STEP_FACTOR,
        step_penalty: float = STEP_PENALTY,
        invalid_move_penalty: float = INVALID_MOVE_PENALTY,
        goal_reward: float = GOAL_REWARD,
        timeout_penalty: float = TIMEOUT_PENALTY,
        distance_shaping_scale: float = DISTANCE_SHAPING_SCALE,
        distance_shaping_mode: str = DEFAULT_DISTANCE_SHAPING_MODE,
        gamma: float = GAMMA,
        action_space: str = "discrete",
        continuous_step_scale: float = 0.25,
        continuous_distance_mode: str = "graph",
    ):
        if observation_mode not in OBSERVATION_MODES:
            raise ValueError(f"unsupported observation_mode: {observation_mode!r}")
        if distance_shaping_mode not in DISTANCE_SHAPING_MODES:
            raise ValueError(f"unsupported distance_shaping_mode: {distance_shaping_mode!r}")
        if action_space not in ACTION_SPACES:
            raise ValueError(f"unsupported action_space: {action_space!r}")
        if not 0.0 < continuous_step_scale <= 1.0:
            raise ValueError("continuous_step_scale must be in (0, 1]")
        if continuous_distance_mode not in CONTINUOUS_DISTANCE_MODES:
            raise ValueError(
                f"unsupported continuous_distance_mode: {continuous_distance_mode!r}"
            )
        self.grid = grid
        self.observation_mode = observation_mode
        self.view_size = view_size
        self.remaining_time_channel = bool(remaining_time_channel)
        self.visit_count_channel = bool(visit_count_channel)
        self.visit_count_clip = int(visit_count_clip)
        self.visit_count_encoding = visit_count_encoding or (
            "episode_log" if action_space == "continuous" else "clipped"
        )
        self.wall_occlusion = bool(wall_occlusion)
        if self.visit_count_clip < 1:
            raise ValueError("visit_count_clip must be >= 1")
        if self.visit_count_encoding not in VISIT_COUNT_ENCODINGS:
            raise ValueError(
                f"unsupported visit_count_encoding: {self.visit_count_encoding!r}"
            )
        self.episode_step_cap = (
            None if max_episode_steps is None else int(max_episode_steps)
        )
        self.timeout_step_factor = timeout_step_factor
        self.min_episode_steps = int(min_episode_steps)
        self.exploration_step_factor = float(exploration_step_factor)
        self.step_penalty = float(step_penalty)
        self.invalid_move_penalty = float(invalid_move_penalty)
        self.goal_reward = float(goal_reward)
        self.timeout_penalty = float(timeout_penalty)
        self.distance_shaping_scale = float(distance_shaping_scale)
        self.distance_shaping_mode = distance_shaping_mode
        self.gamma = float(gamma)
        self.action_space = action_space
        self.continuous_step_scale = float(continuous_step_scale)
        self.continuous_distance_mode = continuous_distance_mode
        self.previous_collision = False
        self.start = tuple(np.argwhere(self.grid == 2)[0])
        self.goal = tuple(np.argwhere(self.grid == 3)[0])
        self.current_position = self.start
        self.continuous_position: tuple[float, float] = (float(self.start[0]), float(self.start[1]))
        self.visit_counts = np.zeros(self.grid.shape, dtype=np.uint32)
        self.visit_counts[self.start] = 1
        self.steps = 0
        self.invalid_moves = 0
        self.total_reward = 0.0
        self._compute_bfs_distances()
        self._centerline = self._compute_centerline()
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
            remaining_time_channel=self.remaining_time_channel,
            visit_count_channel=self.visit_count_channel,
            action_space=self.action_space,
        )

    def _compute_bfs_distances(self) -> None:
        """Compute shortest-path distances from every reachable cell to goal."""

        dist = np.full(self.grid.shape, -1, dtype=np.float32)
        goal_r, goal_c = self.goal
        dist[goal_r, goal_c] = 0
        queue = deque([(goal_r, goal_c)])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {self.goal: None}
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
                    parent[(nr, nc)] = (r, c)
                    queue.append((nr, nc))
        self.bfs_distances = dist
        self._bfs_parent = parent

    def _compute_centerline(self) -> list[tuple[int, int]]:
        """Extract shortest-path waypoint polyline from start to goal."""

        path: list[tuple[int, int]] = [self.start]
        current = self.start
        while current != self.goal:
            nxt = self._bfs_parent.get(current)
            if nxt is None:
                break
            path.append(nxt)
            current = nxt
        return path

    def centerline_distance(self, position: tuple[float, float]) -> float:
        """Return wall-aware continuous distance from ``position`` to the goal."""

        if self.continuous_distance_mode == "start_path":
            return continuous_geodesic_distance(position, self._centerline)
        cell = continuous_position_to_cell(position, self.grid.shape)
        if cell is None or self.grid[cell] == 1:
            return float("inf")
        return graph_continuous_goal_distance(
            position,
            cell,
            self.goal,
            self.bfs_distances,
            self._bfs_parent,
        )

    def _resolve_max_episode_steps(self) -> int:
        control_optimal_steps = self.optimal_start_steps
        if self.action_space == "continuous":
            control_optimal_steps = int(
                math.ceil(self.optimal_start_steps / self.continuous_step_scale)
            )
        if self.observation_mode == "full" and self.timeout_step_factor is None:
            if self.episode_step_cap is not None:
                return self.episode_step_cap
            return max(self.min_episode_steps, control_optimal_steps)
        candidates = [self.min_episode_steps, control_optimal_steps]
        if self.timeout_step_factor is not None:
            candidates.append(
                int(math.ceil(control_optimal_steps * self.timeout_step_factor))
            )
        if self.observation_mode == "local":
            traversable_cells = int(np.count_nonzero(self.grid != 1))
            step_scale = (
                self.continuous_step_scale if self.action_space == "continuous" else 1.0
            )
            candidates.append(
                int(
                    math.ceil(
                        traversable_cells * self.exploration_step_factor / step_scale
                    )
                )
            )
        resolved = max(candidates)
        if self.episode_step_cap is not None:
            resolved = min(resolved, self.episode_step_cap)
        return resolved

    def reset(self) -> np.ndarray:
        self.steps = 0
        self.invalid_moves = 0
        self.total_reward = 0.0
        self.current_position = self.start
        self.continuous_position = (float(self.start[0]), float(self.start[1]))
        self.previous_collision = False
        self.visit_counts.fill(0)
        self.visit_counts[self.start] = 1
        return self.observation()

    def observation(self) -> np.ndarray:
        if self.observation_mode == "full":
            return self._full_observation()
        return self._local_observation(self.current_position)

    def proprioception(self) -> np.ndarray:
        """Return within-cell offsets and previous collision for continuous control."""

        if self.action_space != "continuous":
            return np.empty(0, dtype=np.float32)
        offsets = 2.0 * (
            np.asarray(self.continuous_position, dtype=np.float32)
            - np.asarray(self.current_position, dtype=np.float32)
        )
        return np.asarray(
            (offsets[0], offsets[1], float(self.previous_collision)),
            dtype=np.float32,
        )

    @property
    def effective_visit_count_clip(self) -> int:
        """Return the occupancy-sample clip used by the observation channel."""

        if self.action_space == "continuous":
            return int(math.ceil(self.visit_count_clip / self.continuous_step_scale))
        return self.visit_count_clip

    def _full_observation(self) -> np.ndarray:
        walls = (self.grid == 1).astype(np.float32)
        agent = np.zeros(self.grid.shape, dtype=np.float32)
        agent[self.current_position] = 1.0
        goal = np.zeros(self.grid.shape, dtype=np.float32)
        goal[self.goal] = 1.0
        channels = [walls, agent, goal]
        if self.remaining_time_channel:
            channels.append(
                np.full(self.grid.shape, self.remaining_time_fraction, dtype=np.float32)
            )
        if self.visit_count_channel:
            channels.append(self._normalized_visit_counts())
        return np.stack(channels)

    def _local_observation(self, position: tuple[int, int]) -> np.ndarray:
        half = self.view_size // 2
        walls = np.zeros((self.view_size, self.view_size), dtype=np.float32)
        goal = np.zeros_like(walls)
        visit_counts = np.zeros_like(walls)
        for di in range(-half, half + 1):
            for dj in range(-half, half + 1):
                r, c = position[0] + di, position[1] + dj
                vi, vj = di + half, dj + half
                if not (0 <= r < self.grid.shape[0] and 0 <= c < self.grid.shape[1]):
                    walls[vi, vj] = 1.0
                    continue
                cell = self.grid[r, c]
                walls[vi, vj] = float(cell == 1)
                goal[vi, vj] = float((r, c) == self.goal)
                visit_counts[vi, vj] = self._normalized_visit_count(
                    int(self.visit_counts[r, c])
                )
        if self.wall_occlusion:
            visible = local_visibility_mask(walls.astype(np.bool_))
            walls *= visible
            goal *= visible
            visit_counts *= visible
        channels = [walls, goal]
        if self.remaining_time_channel:
            channels.append(
                np.full(
                    (self.view_size, self.view_size),
                    self.remaining_time_fraction,
                    dtype=np.float32,
                )
            )
        if self.visit_count_channel:
            channels.append(visit_counts)
        return np.stack(channels)

    def _normalized_visit_counts(self) -> np.ndarray:
        if self.visit_count_encoding == "episode_log":
            denominator = math.log1p(self.max_episode_steps + 1)
            return np.minimum(
                np.log1p(self.visit_counts.astype(np.float32)) / denominator,
                1.0,
            )
        return np.minimum(self.visit_counts, self.effective_visit_count_clip).astype(
            np.float32
        ) / self.effective_visit_count_clip

    def _normalized_visit_count(self, count: int) -> float:
        if self.visit_count_encoding == "episode_log":
            return min(
                math.log1p(count) / math.log1p(self.max_episode_steps + 1),
                1.0,
            )
        return (
            min(count, self.effective_visit_count_clip)
            / self.effective_visit_count_clip
        )

    @property
    def remaining_time_fraction(self) -> float:
        """Return the observable fraction of the episode budget remaining."""

        return max(self.max_episode_steps - self.steps, 0) / max(self.max_episode_steps, 1)

    def step(
        self, action: int | tuple[float, float] | np.ndarray
    ) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        if self.action_space == "continuous":
            action_array = np.asarray(action, dtype=np.float32)
            if action_array.shape != (2,):
                raise ValueError(
                    "continuous action must have shape (2,), "
                    f"got {action_array.shape}"
                )
            return self._step_continuous((float(action_array[0]), float(action_array[1])))
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
        self.visit_counts[self.current_position] += 1

        new_distance = float(self.bfs_distances[self.current_position])
        solved = self.current_position == self.goal

        reward = self.step_penalty
        if invalid:
            reward += self.invalid_move_penalty
        shaping_reward = self.distance_shaping_scale * self._distance_shaping(
            old_distance,
            new_distance,
        )
        reward += shaping_reward
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
            "distance_shaping_reward": shaping_reward,
        }
        return self.observation(), reward, done, info

    def _step_continuous(
        self, action: tuple[float, float]
    ) -> tuple[np.ndarray, float, bool, dict[str, object]]:
        requested = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        displacement = requested * self.continuous_step_scale
        old_pos = self.continuous_position
        old_cl_dist = self.centerline_distance(old_pos)
        substeps = max(
            1,
            int(
                math.ceil(
                    float(np.linalg.norm(displacement))
                    / CONTINUOUS_COLLISION_SUBSTEP
                )
            ),
        )
        collision_cause = "none"
        destination_cell: tuple[int, int] | None = self.current_position
        for fraction in np.linspace(1.0 / substeps, 1.0, substeps):
            point = np.asarray(old_pos) + displacement * fraction
            destination_cell = continuous_position_to_cell(point, self.grid.shape)
            if destination_cell is None:
                collision_cause = "boundary"
                break
            if self.grid[destination_cell] == 1:
                collision_cause = "wall"
                break
        invalid = collision_cause != "none"
        if invalid:
            self.invalid_moves += 1
            executed = np.zeros(2, dtype=np.float32)
        else:
            next_position = np.asarray(old_pos, dtype=np.float32) + displacement
            self.continuous_position = (float(next_position[0]), float(next_position[1]))
            assert destination_cell is not None
            self.current_position = destination_cell
            executed = displacement.astype(np.float32)
        self.visit_counts[self.current_position] += 1
        self.previous_collision = invalid

        new_cl_dist = self.centerline_distance(self.continuous_position)
        goal_r, goal_c = self.goal
        solved = continuous_position_is_solved(self.continuous_position, (goal_r, goal_c))

        reward = self.step_penalty
        if invalid:
            reward += self.invalid_move_penalty
        shaping_reward = (
            self.distance_shaping_scale
            * self._centerline_distance_shaping(old_cl_dist, new_cl_dist)
        )
        reward += shaping_reward
        if solved:
            reward += self.goal_reward

        self.steps += 1
        timeout = (not solved) and self.steps >= self.max_episode_steps
        if timeout:
            reward += self.timeout_penalty

        done = solved or timeout
        self.total_reward += reward
        info = {
            "moved": not invalid,
            "invalid": invalid,
            "solved": solved,
            "timeout": timeout,
            "distance": new_cl_dist,
            "optimal_steps": self.optimal_start_steps,
            "steps": self.steps,
            "invalid_moves": self.invalid_moves,
            "requested_action": requested.copy(),
            "executed_displacement": executed.copy(),
            "collision_cause": collision_cause,
            "distance_shaping_reward": shaping_reward,
        }
        return self.observation(), reward, done, info

    def _centerline_distance_shaping(
        self, old_distance: float, new_distance: float
    ) -> float:
        if self.distance_shaping_mode == "none" or old_distance <= 0:
            return 0.0
        if self.distance_shaping_mode == "progress":
            return old_distance - new_distance
        if self.distance_shaping_mode == "fractional":
            return (old_distance - new_distance) / old_distance
        old_potential = -old_distance
        new_potential = -new_distance
        return self.gamma * new_potential - old_potential

    def _distance_shaping(self, old_distance: float, new_distance: float) -> float:
        if self.distance_shaping_mode == "none" or old_distance <= 0:
            return 0.0
        if self.distance_shaping_mode == "progress":
            return old_distance - new_distance
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


@dataclass(frozen=True, slots=True)
class CurriculumStage:
    """One fixed-shape sampling stage in the resolved curriculum."""

    maze_size: tuple[int, int]
    name: str
    distance_range: tuple[int, int] | None = None
    complexity_high: float | None = None

    def payload(self) -> dict[str, object]:
        """Return a checkpoint- and log-safe stage description."""

        return {
            "maze_size": list(self.maze_size),
            "name": self.name,
            "distance_range": (
                list(self.distance_range) if self.distance_range is not None else None
            ),
            "complexity_high": self.complexity_high,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, object]) -> "CurriculumStage":
        """Restore one validated stage from checkpoint metadata."""

        maze_size = tuple(int(value) for value in payload["maze_size"])
        distance_value = payload.get("distance_range")
        distance_range = (
            tuple(int(value) for value in distance_value)
            if distance_value is not None
            else None
        )
        return cls(
            maze_size=maze_size,
            name=str(payload["name"]),
            distance_range=distance_range,
            complexity_high=(
                None
                if payload.get("complexity_high") is None
                else float(payload["complexity_high"])
            ),
        )


def automatic_curriculum_sizes(
    final_size: tuple[int, int],
    step: int = DEFAULT_CURRICULUM_SIZE_STEP,
) -> list[tuple[int, int]]:
    """Build an odd, synchronized size ladder ending at ``final_size``."""

    current = tuple(min(size, 11) for size in final_size)
    sizes = [current]
    while current != final_size:
        current = tuple(min(target, value + step) for value, target in zip(current, final_size))
        if current != sizes[-1]:
            sizes.append(current)
    return sizes


def maze_exploration_complexity(environment: Maze) -> float:
    """Estimate goal-discovery cost for a visited-map depth-first explorer."""

    base = list(Maze.ACTIONS)
    orders = []
    for rotation in range(4):
        rotated = base[rotation:] + base[:rotation]
        orders.extend((rotated, list(reversed(rotated))))

    costs: list[int] = []
    for order in orders:
        visited = {environment.start}
        stack: list[tuple[tuple[int, int], int]] = [(environment.start, 0)]
        transitions = 0
        while stack and stack[-1][0] != environment.goal:
            position, next_index = stack[-1]
            if next_index >= len(order):
                stack.pop()
                if stack:
                    transitions += 1
                continue
            stack[-1] = (position, next_index + 1)
            dr, dc = order[next_index]
            neighbor = (position[0] + dr, position[1] + dc)
            if neighbor in visited or not environment._is_valid(neighbor):
                continue
            visited.add(neighbor)
            stack.append((neighbor, 0))
            transitions += 1
        costs.append(transitions)
    return float(sum(costs) / len(costs))


_AUTO_CURRICULUM_CACHE: dict[tuple[object, ...], tuple[CurriculumStage, ...]] = {}


def resolve_curriculum_stages(config: TrainConfig) -> list[CurriculumStage]:
    """Resolve deterministic manual or empirical automatic curriculum stages."""

    if not config.curriculum_enabled:
        return [CurriculumStage(config.maze_size, "unrestricted")]
    if config.curriculum_mode == "manual":
        scale = max(config.maze_size) / max(DEFAULT_MAZE_SIZE)
        return [
            CurriculumStage(
                config.maze_size,
                "manual-easy",
                distance_range=_scale_distance_range(config.curriculum_easy_range, scale),
            ),
            CurriculumStage(
                config.maze_size,
                "manual-medium",
                distance_range=_scale_distance_range(config.curriculum_medium_range, scale),
            ),
            CurriculumStage(config.maze_size, "unrestricted"),
        ]

    key = (
        config.maze_size,
        config.seed,
        config.curriculum_probe_mazes,
        config.curriculum_size_step,
        config.observation_mode,
        config.view_size,
    )
    cached = _AUTO_CURRICULUM_CACHE.get(key)
    if cached is not None:
        return list(cached)

    stages: list[CurriculumStage] = []
    root_rng = random.Random(config.seed + 17 * EVAL_SEED_OFFSET)
    maze_sizes = (
        automatic_curriculum_sizes(config.maze_size, config.curriculum_size_step)
        if config.algorithm == "recurrent_ppo" and config.observation_mode == "local"
        else [config.maze_size]
    )
    for maze_size in maze_sizes:
        stage_config = replace(config, maze_size=maze_size)
        complexities: list[float] = []
        for _ in range(config.curriculum_probe_mazes):
            grid = generate_random_maze(*maze_size, rng=root_rng)
            complexities.append(
                maze_exploration_complexity(_maze_from_grid(stage_config, grid))
            )
        lower, upper = np.quantile(complexities, (1.0 / 3.0, 2.0 / 3.0))
        label = f"{maze_size[0]}x{maze_size[1]}"
        stages.extend(
            (
                CurriculumStage(maze_size, f"{label}-easy", complexity_high=float(lower)),
                CurriculumStage(maze_size, f"{label}-medium", complexity_high=float(upper)),
                CurriculumStage(maze_size, f"{label}-unrestricted"),
            )
        )
    _AUTO_CURRICULUM_CACHE[key] = tuple(stages)
    return stages


class CurriculumController:
    """Promote task difficulty only after repeated validation success."""

    def __init__(
        self,
        config: TrainConfig,
        stages: list[CurriculumStage] | None = None,
    ):
        self.config = config
        self.stages = stages or resolve_curriculum_stages(config)
        self.level = 0
        self.success_streak = 0

    @property
    def complete(self) -> bool:
        return self.level >= len(self.stages) - 1

    @property
    def current_stage(self) -> CurriculumStage:
        return self.stages[min(self.level, len(self.stages) - 1)]

    def target_range(self) -> tuple[int, int] | None:
        """Return the current maze path-length range, or uniform sampling."""

        return self.current_stage.distance_range

    def target_complexity(self) -> float | None:
        """Return the active automatic complexity ceiling."""

        return self.current_stage.complexity_high

    def previous_range(self) -> tuple[int, int] | None:
        """Return the immediately previous curriculum range when available."""

        if self.level < 1:
            return None
        previous = self.stages[self.level - 1]
        if previous.maze_size != self.current_stage.maze_size:
            return None
        return previous.distance_range

    def previous_stage(self) -> CurriculumStage | None:
        """Return the prior stage only when its grid shape is compatible."""

        if self.level < 1:
            return None
        previous = self.stages[self.level - 1]
        return previous if previous.maze_size == self.current_stage.maze_size else None

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

    def force_complete(self) -> bool:
        """Enter the unrestricted final stage and report whether state changed."""

        if self.complete:
            return False
        self.level = len(self.stages) - 1
        self.success_streak = 0
        return True


class MazeTaskSampler:
    """Difficulty-aware MouseMaze sampler conforming to ``TaskSampler``."""

    def __init__(
        self,
        config: TrainConfig,
        rng: random.Random,
        stages: list[CurriculumStage] | None = None,
    ):
        self.config = config
        self.rng = rng
        self.curriculum = CurriculumController(config, stages=stages)
        self.hard_grids_by_shape: dict[tuple[int, int], list[np.ndarray]] = {}
        self._seen_hard_grid_keys: dict[tuple[int, int], set[bytes]] = {}
        self.hard_candidates_seen_by_shape: dict[tuple[int, int], int] = {}
        self.validation_solve_rate: float | None = None

    @property
    def current_size(self) -> tuple[int, int]:
        return self.curriculum.current_stage.maze_size

    @property
    def hard_grids(self) -> list[np.ndarray]:
        """Return the active shape's hard-maze reservoir."""

        return self.hard_grids_by_shape.setdefault(self.current_size, [])

    @property
    def hard_candidates_seen(self) -> int:
        return self.hard_candidates_seen_by_shape.get(self.current_size, 0)

    def sample(self) -> Maze:
        """Sample from current, previous, and uniform task distributions."""

        hard_grid = self.sample_hard_grid()
        if hard_grid is not None:
            return _maze_from_grid(
                replace(self.config, maze_size=tuple(hard_grid.shape)),
                hard_grid,
            )
        stage = self.sample_stage()
        stage_config = replace(self.config, maze_size=stage.maze_size)
        try:
            return make_maze(
                stage_config,
                rng=self.rng,
                target_range=stage.distance_range,
                target_complexity=stage.complexity_high,
            )
        except TypeError as exc:
            if "unexpected keyword argument" not in str(exc):
                raise
            return make_maze(stage_config, rng=self.rng)

    def add_failed_grids(self, failed_grids: list[np.ndarray]) -> int:
        """Reservoir-sample training failures and their unique symmetries."""

        added = 0
        for failed_grid in failed_grids:
            shape = tuple(int(value) for value in failed_grid.shape)
            reservoir = self.hard_grids_by_shape.setdefault(shape, [])
            seen = self._seen_hard_grid_keys.setdefault(shape, set())
            for variant in _hard_maze_variants(failed_grid, include_original=True):
                key = variant.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                candidates_seen = self.hard_candidates_seen_by_shape.get(shape, 0) + 1
                self.hard_candidates_seen_by_shape[shape] = candidates_seen
                if len(reservoir) < self.config.hard_maze_pool_size:
                    reservoir.append(variant)
                    added += 1
                    continue
                replacement = self.rng.randrange(candidates_seen)
                if replacement >= self.config.hard_maze_pool_size:
                    continue
                reservoir[replacement] = variant
                added += 1
        return added

    def restore_hard_grids(
        self,
        grids: object,
        seen_keys: object = None,
        candidates_seen: object = None,
        validation_solve_rate: object = None,
    ) -> None:
        """Restore shape-separated replay reservoir state."""

        if isinstance(grids, (list, tuple)):
            shape = self.current_size
            self.hard_grids_by_shape[shape] = [
                np.ascontiguousarray(value, dtype=np.uint8)
                for value in grids[-self.config.hard_maze_pool_size :]
                if np.asarray(value).shape == shape
            ]
        elif isinstance(grids, dict):
            for shape_key, values in grids.items():
                shape = tuple(int(value) for value in shape_key)
                reservoir = self.hard_grids_by_shape.setdefault(shape, [])
                for value in list(values)[-self.config.hard_maze_pool_size :]:
                    grid = np.ascontiguousarray(value, dtype=np.uint8)
                    if grid.shape == shape:
                        reservoir.append(grid)
        if isinstance(seen_keys, (list, tuple, set)):
            self._seen_hard_grid_keys[self.current_size] = {
                bytes(value)
                for value in seen_keys
                if isinstance(value, (bytes, bytearray))
            }
        elif isinstance(seen_keys, dict):
            for shape_key, values in seen_keys.items():
                shape = tuple(int(value) for value in shape_key)
                self._seen_hard_grid_keys[shape] = {
                    bytes(value)
                    for value in values
                    if isinstance(value, (bytes, bytearray))
                }
        for shape, reservoir in self.hard_grids_by_shape.items():
            self._seen_hard_grid_keys.setdefault(shape, set()).update(
                grid.tobytes() for grid in reservoir
            )
        if isinstance(candidates_seen, (int, float)):
            self.hard_candidates_seen_by_shape[self.current_size] = int(candidates_seen)
        elif isinstance(candidates_seen, dict):
            self.hard_candidates_seen_by_shape = {
                tuple(int(value) for value in shape): int(count)
                for shape, count in candidates_seen.items()
            }
        for shape, seen in self._seen_hard_grid_keys.items():
            self.hard_candidates_seen_by_shape[shape] = max(
                self.hard_candidates_seen_by_shape.get(shape, 0),
                len(seen),
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

    def sample_stage(self) -> CurriculumStage:
        """Choose current, compatible previous, or unrestricted task sampling."""

        current = self.curriculum.current_stage
        if self.curriculum.complete or not self.config.curriculum_enabled:
            return current
        draw = self.rng.random()
        if draw < self.config.curriculum_uniform_fraction:
            return CurriculumStage(current.maze_size, f"{current.name}-uniform")
        previous = self.curriculum.previous_stage()
        if previous is not None and draw < (
            self.config.curriculum_uniform_fraction
            + self.config.curriculum_previous_fraction
        ):
            return previous
        return current

    def sample_target_range(self) -> tuple[int, int] | None:
        """Choose a curriculum range without turning a missing previous stage into uniform."""

        return self.sample_stage().distance_range

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
            if self.curriculum.previous_stage() is not None
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
            "exploration_complexity": maze_exploration_complexity(environment),
            "bucket": difficulty_bucket(distance),
        }


def _hard_maze_variants(
    grid: np.ndarray,
    include_original: bool = False,
) -> list[np.ndarray]:
    """Return unique shape-preserving task symmetries."""

    original = np.asarray(grid, dtype=np.uint8)
    candidates = ([original] if include_original else []) + [
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
        if (not include_original and key == original_key) or key in keys:
            continue
        variants.append(variant)
        keys.add(key)
    return variants


def _generate_prefetched_grid(
    config: TrainConfig,
    seed: int,
    target_range: tuple[int, int] | None,
    target_complexity: float | None = None,
) -> np.ndarray:
    """Generate one deterministically seeded task in a worker process."""

    return make_maze(
        config,
        rng=random.Random(seed),
        target_range=target_range,
        target_complexity=target_complexity,
    ).grid


def _generate_prefetched_grid_batch(
    config: TrainConfig,
    requests: list[tuple[int, CurriculumStage]],
) -> list[np.ndarray]:
    """Generate one ordered batch of independently seeded maze grids."""

    return [
        _generate_prefetched_grid(
            replace(config, maze_size=stage.maze_size),
            seed,
            stage.distance_range,
            stage.complexity_high,
        )
        for seed, stage in requests
    ]


class DeterministicMazePrefetcher:
    """Ordered process-backed maze generation with reproducible task seeds."""

    def __init__(self, sampler: MazeTaskSampler, workers: int):
        self.sampler = sampler
        self.workers = workers
        self.executor: ProcessPoolExecutor | None = None
        self.futures: deque[Future[list[np.ndarray]]] = deque()
        self.ready_grids: deque[np.ndarray] = deque()
        self.generated_grids = 0
        self.started_at = time.perf_counter()
        if workers > 0:
            self.executor = ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("spawn"),
            )
            self._fill()

    def _fill(self) -> None:
        if self.executor is None:
            return
        target_batches = (
            self.workers * self.sampler.config.maze_prefetch_batches_per_worker
        )
        while len(self.futures) < target_batches:
            requests: list[tuple[int, CurriculumStage]] = []
            for _ in range(self.sampler.config.maze_generation_batch_size):
                stage = self.sampler.sample_stage()
                seed = self.sampler.rng.randrange(0, 2**63)
                requests.append((seed, stage))
            self.futures.append(
                self.executor.submit(
                    _generate_prefetched_grid_batch,
                    self.sampler.config,
                    requests,
                )
            )

    def next(self) -> Maze:
        if self.executor is None:
            return self.sampler.sample()
        hard_grid = self.sampler.sample_hard_grid()
        if hard_grid is not None:
            return _maze_from_grid(
                replace(self.sampler.config, maze_size=tuple(hard_grid.shape)),
                hard_grid,
            )
        if not self.ready_grids:
            self.ready_grids.extend(self.futures.popleft().result())
            self.generated_grids += len(self.ready_grids)
        grid = self.ready_grids.popleft()
        self._fill()
        return _maze_from_grid(
            replace(self.sampler.config, maze_size=tuple(grid.shape)),
            grid,
        )

    def telemetry(self) -> dict[str, float | int]:
        """Return prepared queue depth and process generation throughput."""

        elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        return {
            "ready_grids": len(self.ready_grids),
            "queued_batches": len(self.futures),
            "generated_grids": self.generated_grids,
            "generated_grids_per_second": self.generated_grids / elapsed,
        }

    def reset(self) -> None:
        """Discard queued tasks after curriculum promotion."""

        if self.executor is None:
            return
        for future in self.futures:
            future.cancel()
        self.futures.clear()
        self.ready_grids.clear()
        self._fill()

    def close(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            self.futures.clear()
            self.ready_grids.clear()


def _maze_from_grid(config: TrainConfig, grid: np.ndarray) -> Maze:
    return Maze(
        grid.copy(),
        observation_mode=config.observation_mode,
        view_size=config.view_size,
        remaining_time_channel=config.remaining_time_channel,
        visit_count_channel=config.visit_count_channel,
        visit_count_clip=config.visit_count_clip,
        visit_count_encoding=config.visit_count_encoding,
        wall_occlusion=config.wall_occlusion,
        max_episode_steps=config.max_episode_steps,
        timeout_step_factor=config.timeout_step_factor,
        min_episode_steps=config.min_episode_steps,
        exploration_step_factor=config.exploration_step_factor,
        step_penalty=config.step_penalty,
        invalid_move_penalty=config.invalid_move_penalty,
        goal_reward=config.goal_reward,
        timeout_penalty=config.timeout_penalty,
        distance_shaping_scale=config.distance_shaping_scale,
        distance_shaping_mode=config.distance_shaping_mode,
        gamma=config.gamma,
        action_space=config.action_space,
        continuous_step_scale=config.continuous_step_scale,
        continuous_distance_mode=config.continuous_distance_mode,
    )


def make_maze(
    config: TrainConfig,
    rng: random.Random | None = None,
    episode: int | None = None,
    target_range: tuple[int, int] | None = None,
    target_complexity: float | None = None,
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
            distance_matches = True
        else:
            low, high = target_range
            distance_matches = low <= env.optimal_start_steps <= high
        complexity_matches = (
            target_complexity is None
            or maze_exploration_complexity(env) <= target_complexity
        )
        if distance_matches and complexity_matches:
            return env

    if target_range is not None or target_complexity is not None:
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
    """Results of stepping a same-shaped batch of maze tasks."""

    states: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    action_masks: np.ndarray
    invalid: np.ndarray
    solved: np.ndarray
    timeout: np.ndarray
    distance_shaping_rewards: np.ndarray | None = None
    requested_actions: np.ndarray | None = None
    executed_displacements: np.ndarray | None = None
    collision_causes: np.ndarray | None = None
    proprioception: np.ndarray | None = None


class MazeBatch:
    """Vectorized Maze adapter used by DQN and PPO collection.

    Individual ``Maze`` objects remain the source of task construction and
    rendering. This adapter only batches their fixed-shape transition logic.
    """

    def __init__(self, environments: list[Maze], continuous: bool | None = None):
        if not environments:
            raise ValueError("MazeBatch requires at least one environment")
        first = environments[0]
        grid_shape = first.grid.shape
        if any(
            env.observation_mode != first.observation_mode
            or env.view_size != first.view_size
            or env.remaining_time_channel != first.remaining_time_channel
            or env.visit_count_channel != first.visit_count_channel
            or env.visit_count_clip != first.visit_count_clip
            or env.visit_count_encoding != first.visit_count_encoding
            or env.wall_occlusion != first.wall_occlusion
            or env.grid.shape != grid_shape
            or env.action_space != first.action_space
            or env.continuous_distance_mode != first.continuous_distance_mode
            for env in environments
        ):
            raise ValueError("all batched mazes must use one observation and grid shape")
        self.size = len(environments)
        self.grid_shape = grid_shape
        self.observation_mode = first.observation_mode
        self.view_size = first.view_size
        self.remaining_time_channel = first.remaining_time_channel
        self.visit_count_channel = first.visit_count_channel
        self.visit_count_clip = first.visit_count_clip
        self.visit_count_encoding = first.visit_count_encoding
        self.wall_occlusion = first.wall_occlusion
        self.continuous = (
            first.action_space == "continuous" if continuous is None else bool(continuous)
        )
        if self.continuous != (first.action_space == "continuous"):
            raise ValueError("MazeBatch action mode must match its environments")
        self.continuous_step_scale = first.continuous_step_scale
        self.continuous_distance_mode = first.continuous_distance_mode
        self.effective_visit_count_clip = (
            int(math.ceil(self.visit_count_clip / self.continuous_step_scale))
            if self.continuous
            else self.visit_count_clip
        )
        self.grids = np.empty((self.size, *grid_shape), dtype=np.uint8)
        self.bfs_distances = np.empty((self.size, *grid_shape), dtype=np.float32)
        self.goal_successors = np.empty((self.size, *grid_shape, 2), dtype=np.int64)
        self.starts = np.empty((self.size, 2), dtype=np.int64)
        self.goals = np.empty((self.size, 2), dtype=np.int64)
        self.positions = np.empty((self.size, 2), dtype=np.int64)
        self.continuous_positions = np.empty((self.size, 2), dtype=np.float32)
        self.previous_collisions = np.empty(self.size, dtype=np.bool_)
        self.visit_counts = np.empty((self.size, *grid_shape), dtype=np.uint32)
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
        self._centerlines: dict[int, list[tuple[int, int]]] = {}
        max_centerline_segments = max(math.prod(grid_shape) - 1, 1)
        geometry_shape = (self.size, max_centerline_segments)
        self._centerline_starts = np.zeros((*geometry_shape, 2), dtype=np.float64)
        self._centerline_deltas = np.zeros_like(self._centerline_starts)
        self._centerline_lengths = np.zeros(geometry_shape, dtype=np.float64)
        self._centerline_remaining = np.zeros(geometry_shape, dtype=np.float64)
        self._centerline_segment_counts = np.zeros(self.size, dtype=np.int64)
        self._current_centerline_distances = np.zeros(self.size, dtype=np.float64)
        for index, environment in enumerate(environments):
            self.replace(index, environment)

    def replace(self, index: int, environment: Maze) -> None:
        """Replace a completed slot with a freshly generated equivalent task."""

        if (
            environment.observation_mode != self.observation_mode
            or environment.view_size != self.view_size
            or environment.remaining_time_channel != self.remaining_time_channel
            or environment.visit_count_channel != self.visit_count_channel
            or environment.visit_count_clip != self.visit_count_clip
            or environment.visit_count_encoding != self.visit_count_encoding
            or environment.wall_occlusion != self.wall_occlusion
            or environment.grid.shape != self.grid_shape
            or environment.continuous_step_scale != self.continuous_step_scale
            or environment.continuous_distance_mode != self.continuous_distance_mode
        ):
            raise ValueError("replacement maze must match the batch observation shape")
        self.grids[index] = environment.grid
        self.bfs_distances[index] = environment.bfs_distances
        self.goal_successors[index].fill(-1)
        for cell, successor in environment._bfs_parent.items():
            if successor is not None:
                self.goal_successors[index, cell[0], cell[1]] = successor
        self.starts[index] = environment.start
        self.goals[index] = environment.goal
        self.positions[index] = environment.current_position
        self.continuous_positions[index] = environment.continuous_position
        self.previous_collisions[index] = environment.previous_collision
        self.visit_counts[index] = environment.visit_counts
        self.steps[index] = environment.steps
        self.invalid_moves[index] = environment.invalid_moves
        self.total_rewards[index] = environment.total_reward
        self.optimal_steps[index] = environment.optimal_start_steps
        self.max_episode_steps[index] = environment.max_episode_steps
        centerline = list(environment._centerline)
        self._centerlines[index] = centerline
        self._set_centerline_geometry(index, centerline)
        self._current_centerline_distances[index] = environment.centerline_distance(
            environment.continuous_position
        )

    def _set_centerline_geometry(
        self,
        index: int,
        centerline: list[tuple[int, int]],
    ) -> None:
        """Cache one centerline in padded arrays for batched distance queries."""

        self._centerline_starts[index].fill(0.0)
        self._centerline_deltas[index].fill(0.0)
        self._centerline_lengths[index].fill(0.0)
        self._centerline_remaining[index].fill(0.0)
        segment_count = max(len(centerline) - 1, 0)
        self._centerline_segment_counts[index] = segment_count
        if segment_count == 0:
            return
        points = np.asarray(centerline, dtype=np.float64)
        deltas = points[1:] - points[:-1]
        lengths = np.linalg.norm(deltas, axis=1)
        self._centerline_starts[index, :segment_count] = points[:-1]
        self._centerline_deltas[index, :segment_count] = deltas
        self._centerline_lengths[index, :segment_count] = lengths
        remaining = np.cumsum(lengths[::-1])[::-1] - lengths
        self._centerline_remaining[index, :segment_count] = remaining

    def _centerline_distances(
        self,
        indices: np.ndarray,
        positions: np.ndarray,
    ) -> np.ndarray:
        """Return graph-aware continuous distances for a selected batch."""

        selected = np.asarray(indices, dtype=np.int64)
        points = np.asarray(positions, dtype=np.float64)
        if selected.shape != (len(points),):
            raise ValueError("indices and positions must have matching leading dimensions")
        if self.continuous_distance_mode == "start_path":
            segment_counts = self._centerline_segment_counts[selected]
            max_segments = int(segment_counts.max(initial=0))
            distances = np.linalg.norm(points - self.goals[selected], axis=1)
            if max_segments == 0:
                return distances
            starts = self._centerline_starts[selected, :max_segments]
            deltas = self._centerline_deltas[selected, :max_segments]
            lengths = self._centerline_lengths[selected, :max_segments]
            remaining = self._centerline_remaining[selected, :max_segments]
            valid = (
                np.arange(max_segments)[np.newaxis, :]
                < segment_counts[:, np.newaxis]
            )
            relative = points[:, np.newaxis, :] - starts
            length_squared = lengths * lengths
            fractions = np.divide(
                np.sum(relative * deltas, axis=2),
                length_squared,
                out=np.zeros_like(length_squared),
                where=valid,
            )
            np.clip(fractions, 0.0, 1.0, out=fractions)
            projections = starts + fractions[:, :, np.newaxis] * deltas
            lateral = np.linalg.norm(points[:, np.newaxis, :] - projections, axis=2)
            candidates = lateral + (1.0 - fractions) * lengths + remaining
            candidates[~valid] = np.inf
            has_segments = segment_counts > 0
            distances[has_segments] = np.min(candidates[has_segments], axis=1)
            return distances
        cells = np.floor(points + 0.5).astype(np.int64)
        rows = np.clip(cells[:, 0], 0, self.grid_shape[0] - 1)
        cols = np.clip(cells[:, 1], 0, self.grid_shape[1] - 1)
        successors = self.goal_successors[selected, rows, cols]
        at_goal = np.all(cells == self.goals[selected], axis=1)
        targets = np.where(at_goal[:, np.newaxis], self.goals[selected], successors)
        base = np.where(
            at_goal,
            0.0,
            self.bfs_distances[selected, targets[:, 0], targets[:, 1]],
        )
        return np.linalg.norm(points - targets, axis=1) + base

    def _normalize_visit_counts(
        self,
        counts: np.ndarray,
        selected: np.ndarray,
    ) -> np.ndarray:
        """Normalize visit counts using the configured observation encoding."""

        if self.visit_count_encoding == "episode_log":
            denominator = np.log1p(self.max_episode_steps[selected] + 1).astype(
                np.float32
            )
            denominator = denominator.reshape(
                (len(selected),) + (1,) * (counts.ndim - 1)
            )
            return np.minimum(
                np.log1p(counts.astype(np.float32)) / denominator,
                1.0,
            )
        return (
            np.minimum(counts, self.effective_visit_count_clip).astype(np.float32)
            / self.effective_visit_count_clip
        )

    def observations(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Build batched full or centered local observations."""

        selected = (
            np.arange(self.size, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if self.observation_mode == "local":
            return self._local_observations(selected)
        walls = (self.grids[selected] == 1).astype(np.float32)
        agent = np.zeros((len(selected), *self.grid_shape), dtype=np.float32)
        local_indices = np.arange(len(selected))
        agent[
            local_indices,
            self.positions[selected, 0],
            self.positions[selected, 1],
        ] = 1.0
        goal = np.zeros_like(agent)
        goal[
            local_indices,
            self.goals[selected, 0],
            self.goals[selected, 1],
        ] = 1.0
        channels = [walls, agent, goal]
        if self.remaining_time_channel:
            remaining = np.maximum(
                self.max_episode_steps[selected] - self.steps[selected],
                0,
            ) / np.maximum(
                self.max_episode_steps[selected],
                1,
            )
            channels.append(
                np.broadcast_to(
                    remaining[:, np.newaxis, np.newaxis],
                    walls.shape,
                )
            )
        if self.visit_count_channel:
            channels.append(
                self._normalize_visit_counts(self.visit_counts[selected], selected)
            )
        return np.stack(channels, axis=1).astype(np.float32, copy=False)

    def _local_observations(self, selected: np.ndarray) -> np.ndarray:
        half = self.view_size // 2
        offsets = np.arange(-half, half + 1, dtype=np.int64)
        rows = (
            self.positions[selected, 0, np.newaxis, np.newaxis]
            + offsets[np.newaxis, :, np.newaxis]
        )
        cols = (
            self.positions[selected, 1, np.newaxis, np.newaxis]
            + offsets[np.newaxis, np.newaxis, :]
        )
        observation_shape = (len(selected), self.view_size, self.view_size)
        rows = np.broadcast_to(rows, observation_shape)
        cols = np.broadcast_to(cols, observation_shape)
        in_bounds = (
            (rows >= 0)
            & (rows < self.grid_shape[0])
            & (cols >= 0)
            & (cols < self.grid_shape[1])
        )
        clipped_rows = np.clip(rows, 0, self.grid_shape[0] - 1)
        clipped_cols = np.clip(cols, 0, self.grid_shape[1] - 1)
        batch_indices = selected[:, np.newaxis, np.newaxis]
        cells = self.grids[batch_indices, clipped_rows, clipped_cols]
        walls = ((~in_bounds) | (cells == 1)).astype(np.float32)
        goal = (
            in_bounds
            & (
                (rows == self.goals[selected, 0, np.newaxis, np.newaxis])
                & (cols == self.goals[selected, 1, np.newaxis, np.newaxis])
            )
        ).astype(np.float32)
        visit_counts = self._normalize_visit_counts(
            self.visit_counts[batch_indices, clipped_rows, clipped_cols],
            selected,
        )
        visit_counts *= in_bounds
        if self.wall_occlusion:
            visible = local_visibility_mask(walls.astype(np.bool_))
            walls *= visible
            goal *= visible
            visit_counts *= visible
        channels = [walls, goal]
        if self.remaining_time_channel:
            remaining = np.maximum(
                self.max_episode_steps[selected] - self.steps[selected],
                0,
            ) / np.maximum(
                self.max_episode_steps[selected],
                1,
            )
            channels.append(
                np.broadcast_to(
                    remaining[:, np.newaxis, np.newaxis],
                    walls.shape,
                )
            )
        if self.visit_count_channel:
            channels.append(visit_counts)
        return np.stack(channels, axis=1).astype(np.float32, copy=False)

    def proprioception(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Build batched continuous within-cell offsets and collision flags."""

        if not self.continuous:
            selected_count = self.size if indices is None else len(indices)
            return np.empty((selected_count, 0), dtype=np.float32)
        selected = (
            np.arange(self.size, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        offsets = 2.0 * (
            self.continuous_positions[selected].astype(np.float32)
            - self.positions[selected].astype(np.float32)
        )
        values = np.empty((len(selected), 3), dtype=np.float32)
        values[:, :2] = offsets
        values[:, 2] = self.previous_collisions[selected]
        return values

    def valid_action_masks(self, indices: np.ndarray | None = None) -> np.ndarray:
        """Return legal action masks for every current batch position."""

        selected = (
            np.arange(self.size, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        deltas = np.asarray(Maze.ACTIONS, dtype=np.int64)
        candidates = self.positions[selected, np.newaxis, :] + deltas[np.newaxis, :, :]
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
            selected[:, np.newaxis],
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

        if self.continuous:
            return self._step_continuous(actions, indices)

        action_array = np.asarray(actions, dtype=np.int64)
        selected = (
            np.arange(self.size, dtype=np.int64)
            if indices is None
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
        action_masks = self.valid_action_masks(selected)
        local_indices = np.arange(len(selected))
        valid = action_masks[local_indices, action_array]
        candidates = old_positions + deltas[action_array]
        self.positions[selected[valid]] = candidates[valid]
        self.visit_counts[
            selected,
            self.positions[selected, 0],
            self.positions[selected, 1],
        ] += 1
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
        shaping_rewards = np.zeros(len(selected), dtype=np.float32)
        if self.distance_shaping_mode == "progress":
            shaping = old_distance - new_distance
            shaping_rewards = self.distance_shaping_scale * shaping
        elif self.distance_shaping_mode == "potential":
            shaping = self.gamma * (-new_distance) - (-old_distance)
            shaping_rewards = self.distance_shaping_scale * np.where(
                old_distance > 0, shaping, 0.0
            )
        elif self.distance_shaping_mode == "fractional":
            shaping = np.divide(
                old_distance - new_distance,
                old_distance,
                out=np.zeros_like(old_distance),
                where=old_distance > 0,
            )
            shaping_rewards = self.distance_shaping_scale * shaping
        rewards += shaping_rewards
        rewards += solved.astype(np.float32) * self.goal_reward
        self.steps[selected] += 1
        timeout = ~solved & (
            self.steps[selected] >= self.max_episode_steps[selected]
        )
        rewards += timeout.astype(np.float32) * self.timeout_penalty
        dones = solved | timeout
        self.total_rewards[selected] += rewards
        next_states = self.observations(selected)
        next_masks = self.valid_action_masks(selected)
        return BatchStep(
            states=next_states,
            rewards=rewards,
            dones=dones,
            action_masks=next_masks,
            invalid=invalid,
            solved=solved,
            timeout=timeout,
            distance_shaping_rewards=shaping_rewards,
        )

    def _step_continuous(
        self,
        actions: np.ndarray,
        indices: np.ndarray | None = None,
    ) -> BatchStep:
        action_array = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
        selected = (
            np.arange(self.size, dtype=np.int64)
            if indices is None
            else np.asarray(indices, dtype=np.int64)
        )
        if selected.ndim != 1 or np.any((selected < 0) | (selected >= self.size)):
            raise ValueError("indices must be a one-dimensional in-range array")
        if len(np.unique(selected)) != len(selected):
            raise ValueError("indices must not contain duplicates")
        if action_array.shape != (len(selected), 2):
            raise ValueError(
                f"continuous actions must have shape ({len(selected)}, 2), got {action_array.shape}"
            )

        old_positions = self.continuous_positions[selected].copy()
        displacements = action_array * self.continuous_step_scale
        executed = np.zeros_like(displacements)
        substeps = np.maximum(
            1,
            np.ceil(
                np.linalg.norm(displacements, axis=1)
                / CONTINUOUS_COLLISION_SUBSTEP
            ).astype(np.int64),
        )
        max_substeps = int(substeps.max(initial=1))
        fractions = np.ones((len(selected), max_substeps), dtype=np.float64)
        active_substeps = (
            np.arange(max_substeps)[np.newaxis, :] < substeps[:, np.newaxis]
        )
        for count in np.unique(substeps):
            matching = substeps == count
            fractions[matching, :count] = np.linspace(1.0 / count, 1.0, count)
        swept_points = (
            old_positions[:, np.newaxis, :]
            + displacements[:, np.newaxis, :] * fractions[:, :, np.newaxis]
        )
        rows, cols = self.grid_shape
        in_bounds = (
            (swept_points[:, :, 0] >= -0.5)
            & (swept_points[:, :, 0] < rows - 0.5)
            & (swept_points[:, :, 1] >= -0.5)
            & (swept_points[:, :, 1] < cols - 0.5)
        )
        swept_cells = np.floor(swept_points + 0.5).astype(np.int64)
        clipped_rows = np.clip(swept_cells[:, :, 0], 0, rows - 1)
        clipped_cols = np.clip(swept_cells[:, :, 1], 0, cols - 1)
        walls = self.grids[
            selected[:, np.newaxis],
            clipped_rows,
            clipped_cols,
        ] == 1
        collisions = active_substeps & (~in_bounds | walls)
        invalid = np.any(collisions, axis=1)
        first_collision = np.argmax(collisions, axis=1)
        collision_causes = np.full(len(selected), "none", dtype=object)
        invalid_indices = np.flatnonzero(invalid)
        if len(invalid_indices):
            first_in_bounds = in_bounds[invalid_indices, first_collision[invalid_indices]]
            collision_causes[invalid_indices] = np.where(
                first_in_bounds,
                "wall",
                "boundary",
            )

        valid_indices = np.flatnonzero(~invalid)
        destination_cells = swept_cells[
            np.arange(len(selected)),
            substeps - 1,
        ]
        if len(valid_indices):
            valid_environments = selected[valid_indices]
            self.continuous_positions[valid_environments] = (
                old_positions[valid_indices] + displacements[valid_indices]
            )
            self.positions[valid_environments] = destination_cells[valid_indices]
            executed[valid_indices] = displacements[valid_indices]
        self.visit_counts[
            selected,
            self.positions[selected, 0],
            self.positions[selected, 1],
        ] += 1
        self.previous_collisions[selected] = invalid
        self.invalid_moves[selected] += invalid.astype(np.int64)

        old_cl_dist = self._current_centerline_distances[selected].copy()
        new_cl_dist = self._centerline_distances(
            selected,
            self.continuous_positions[selected],
        )
        self._current_centerline_distances[selected] = new_cl_dist

        goal_offsets = self.continuous_positions[selected] - self.goals[selected]
        solved = np.linalg.norm(goal_offsets, axis=1) < CONTINUOUS_GOAL_RADIUS

        rewards = np.full(len(selected), self.step_penalty, dtype=np.float32)
        rewards += invalid.astype(np.float32) * self.invalid_move_penalty
        shaping_rewards = np.zeros(len(selected), dtype=np.float32)
        if self.distance_shaping_mode == "progress":
            shaping = old_cl_dist - new_cl_dist
            shaping_rewards = self.distance_shaping_scale * shaping
        elif self.distance_shaping_mode == "potential":
            shaping = self.gamma * (-new_cl_dist) - (-old_cl_dist)
            shaping_rewards = self.distance_shaping_scale * np.where(
                old_cl_dist > 0, shaping, 0.0
            )
        elif self.distance_shaping_mode == "fractional":
            shaping = np.divide(
                old_cl_dist - new_cl_dist,
                old_cl_dist,
                out=np.zeros_like(old_cl_dist),
                where=old_cl_dist > 0,
            )
            shaping_rewards = self.distance_shaping_scale * shaping
        rewards += shaping_rewards
        rewards += solved.astype(np.float32) * self.goal_reward
        self.steps[selected] += 1
        timeout = ~solved & (self.steps[selected] >= self.max_episode_steps[selected])
        rewards += timeout.astype(np.float32) * self.timeout_penalty
        dones = solved | timeout
        self.total_rewards[selected] += rewards
        next_states = self.observations(selected)
        next_masks = self.valid_action_masks(selected)
        return BatchStep(
            states=next_states,
            rewards=rewards,
            dones=dones,
            action_masks=next_masks,
            invalid=invalid,
            solved=solved,
            timeout=timeout,
            distance_shaping_rewards=shaping_rewards,
            requested_actions=action_array.copy(),
            executed_displacements=executed,
            collision_causes=collision_causes,
            proprioception=self.proprioception(selected),
        )

    def _centerline_distance_batch(
        self, env_idx: int, position: tuple[float, float] | np.ndarray
    ) -> float:
        return float(
            self._centerline_distances(
                np.asarray([env_idx], dtype=np.int64),
                np.asarray(position, dtype=np.float64)[np.newaxis, :],
            )[0]
        )

    def _get_centerline(self, env_idx: int) -> list[tuple[int, int]]:
        return self._centerlines[env_idx]

    def episode_stats(self, index: int) -> EpisodeStats:
        """Return terminal statistics for one slot before replacing it."""

        if self.continuous:
            solved = continuous_position_is_solved(
                self.continuous_positions[index], self.goals[index]
            )
        else:
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
    """Fully convolutional trunk over spatial observation channels."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        channels: int = 64,
        residual_blocks: int = 6,
        has_agent_channel: bool = True,
    ):
        super().__init__()
        input_channels, rows, cols = input_shape
        self.has_agent_channel = has_agent_channel
        row_coords = torch.linspace(-1.0, 1.0, rows)
        col_coords = torch.linspace(-1.0, 1.0, cols)
        row_plane = row_coords.view(1, rows, 1).expand(1, rows, cols)
        col_plane = col_coords.view(1, 1, cols).expand(1, rows, cols)
        self.register_buffer(
            "coordinate_planes",
            torch.cat((row_plane, col_plane), dim=0),
            persistent=False,
        )
        network_input_channels = input_channels - int(has_agent_channel) + 2
        layers: list[nn.Module] = [
            nn.Conv2d(network_input_channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
        ]
        layers.extend(ResidualBlock(channels) for _ in range(residual_blocks))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.has_agent_channel:
            observable_features = torch.cat((x[:, 0:1], x[:, 2:]), dim=1)
        else:
            observable_features = x
        coordinates = self.coordinate_planes.to(dtype=x.dtype).expand(x.shape[0], -1, -1, -1)
        return self.net(torch.cat((observable_features, coordinates), dim=1))


def _gather_agent_cell(cell_values: torch.Tensor, observations: torch.Tensor) -> torch.Tensor:
    agent_mask = observations[:, 1:2]
    return (cell_values * agent_mask).flatten(2).sum(dim=2)


def _gather_policy_cell(
    cell_values: torch.Tensor,
    observations: torch.Tensor,
    has_agent_channel: bool,
) -> torch.Tensor:
    """Gather spatial outputs at the agent cell or local-view center."""

    if has_agent_channel:
        return _gather_agent_cell(cell_values, observations)
    center_row = cell_values.shape[-2] // 2
    center_col = cell_values.shape[-1] // 2
    return cell_values[..., center_row, center_col]


class SpatialQNetwork(nn.Module):
    """Dueling Q-map network that gathers at the agent or center cell."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_size: int = 4,
        has_agent_channel: bool = True,
    ):
        super().__init__()
        _channels, _rows, _cols = input_shape
        hidden_channels = 64
        self.has_agent_channel = has_agent_channel
        self.trunk = SpatialTrunk(
            input_shape,
            hidden_channels,
            has_agent_channel=has_agent_channel,
        )
        self.value_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)
        self.advantage_head = nn.Conv2d(hidden_channels, output_size, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.trunk(x)
        value_map = self.value_head(features)
        advantage_map = self.advantage_head(features)
        q_map = value_map + advantage_map - advantage_map.mean(dim=1, keepdim=True)
        return _gather_policy_cell(q_map, x, self.has_agent_channel)


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
    has_agent_channel: bool = True,
) -> nn.Module:
    if network_type == "spatial":
        return SpatialQNetwork(input_shape, has_agent_channel=has_agent_channel)
    if network_type == "flat":
        return FlatQNetwork(input_shape)
    raise ValueError(f"unsupported network_type: {network_type!r}")


def infer_q_network_type(state_dict: dict[str, torch.Tensor]) -> str:
    if any(key.startswith("trunk.") for key in state_dict):
        return "spatial"
    return "flat"


class MaskedActorCriticNetwork(nn.Module):
    """Spatial actor-critic gathering at the agent or center cell."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        output_size: int = 4,
        has_agent_channel: bool = True,
    ):
        super().__init__()
        _channels, _rows, _cols = input_shape
        hidden_channels = 64
        self.has_agent_channel = has_agent_channel
        self.trunk = SpatialTrunk(
            input_shape,
            hidden_channels,
            has_agent_channel=has_agent_channel,
        )
        self.policy_head = nn.Conv2d(hidden_channels, output_size, kernel_size=1)
        self.value_head = nn.Conv2d(hidden_channels, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.trunk(x)
        logits = _gather_policy_cell(self.policy_head(features), x, self.has_agent_channel)
        value = _gather_policy_cell(
            self.value_head(features),
            x,
            self.has_agent_channel,
        ).squeeze(1)
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
        self.observation_shape = observation_shape_ or config_observation_shape(
            self.config
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
        has_agent_channel = self.observation_mode == "full"
        self.online_net = build_q_network(
            self.observation_shape,
            network_type,
            has_agent_channel=has_agent_channel,
        ).to(self.device)
        self.target_net = build_q_network(
            self.observation_shape,
            network_type,
            has_agent_channel=has_agent_channel,
        ).to(self.device)
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
            "observation_settings": observation_settings_payload(self.config),
            "distance_shaping_mode": self.config.distance_shaping_mode,
            "continuous_distance_mode": self.config.continuous_distance_mode,
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
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
        ):
            raise ValueError(
                f"checkpoint {path!r} is not a supported MouseMaze checkpoint"
            )
        _validate_checkpoint_observation_settings(payload, self.config, path)
        _validate_checkpoint_reward_settings(payload, self.config, path)
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
        self.observation_shape = observation_shape_ or config_observation_shape(
            self.config
        )
        self.policy_net = MaskedActorCriticNetwork(
            self.observation_shape,
            has_agent_channel=self.observation_mode == "full",
        ).to(self.device)
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
            "observation_settings": observation_settings_payload(self.config),
            "distance_shaping_mode": self.config.distance_shaping_mode,
            "continuous_distance_mode": self.config.continuous_distance_mode,
            "maze_size": self.config.maze_size,
            "update_count": self.update_count,
            "total_env_steps": self.total_env_steps,
            "best_greedy_solve_rate": self.best_greedy_solve_rate,
            "training_state": self.training_state,
        }
        _atomic_torch_save(payload, path)

    def load(self, path: str) -> None:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
        ):
            raise ValueError(
                f"checkpoint {path!r} is not a supported MouseMaze checkpoint"
            )
        _validate_checkpoint_observation_settings(payload, self.config, path)
        _validate_checkpoint_reward_settings(payload, self.config, path)
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
        has_agent_channel: bool = True,
        continuous: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.has_agent_channel = has_agent_channel
        self.continuous = continuous
        self.action_dim = 2 if continuous else output_size
        self.proprioception_size = 3 if continuous else 0
        self.trunk = SpatialTrunk(
            input_shape,
            channels=64,
            has_agent_channel=has_agent_channel,
        )
        action_feature_dim = self.action_dim
        self.gru = nn.GRUCell(
            64 + self.proprioception_size + action_feature_dim + 1,
            hidden_size,
        )
        if continuous:
            self.mean_head = nn.Linear(hidden_size, self.action_dim)
            self.log_std_head = nn.Linear(hidden_size, self.action_dim)
            nn.init.zeros_(self.log_std_head.weight)
            nn.init.constant_(self.log_std_head.bias, -1.0)
        else:
            self.policy_head = nn.Linear(hidden_size, output_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def spatial_features(self, observations: torch.Tensor) -> torch.Tensor:
        return _gather_policy_cell(
            self.trunk(observations),
            observations,
            self.has_agent_channel,
        )

    def forward_step(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        hidden: torch.Tensor,
        proprioception: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = hidden * (~episode_starts).to(hidden.dtype).unsqueeze(1)
        features = self.spatial_features(observations)
        if self.continuous:
            valid_actions = ~episode_starts
            action_features = previous_actions.clamp(-1, 1)
            action_features *= valid_actions.to(features.dtype).unsqueeze(1)
        else:
            valid_actions = previous_actions >= 0
            safe_actions = previous_actions.clamp_min(0)
            action_features = nn.functional.one_hot(safe_actions, 4).to(features.dtype)
            action_features *= valid_actions.to(features.dtype).unsqueeze(1)
        input_features = [features]
        if self.continuous:
            if proprioception is None or proprioception.shape != (features.shape[0], 3):
                actual = None if proprioception is None else tuple(proprioception.shape)
                raise ValueError(
                    "continuous proprioception must have shape "
                    f"({features.shape[0]}, 3), got {actual}"
                )
            input_features.append(proprioception.to(features.dtype))
        input_features.extend(
            (action_features, previous_rewards.to(features.dtype).unsqueeze(1))
        )
        inputs = torch.cat(input_features, dim=1)
        next_hidden = self.gru(inputs, hidden.to(inputs.dtype))
        if self.continuous:
            mean = self.mean_head(next_hidden).float()
            log_std = self.log_std_head(next_hidden).float()
            log_std = torch.clamp(
                log_std,
                CONTINUOUS_LOG_STD_MIN,
                CONTINUOUS_LOG_STD_MAX,
            )
            values = self.value_head(next_hidden).squeeze(1).float()
            return mean, values, next_hidden.float(), log_std
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
        proprioception: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.continuous:
            return self._forward_sequence_continuous(
                observations,
                previous_actions,
                previous_rewards,
                episode_starts,
                initial_hidden,
                proprioception,
            )
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
                None,
            )
            logits.append(step_logits)
            values.append(step_values)
        return torch.stack(logits), torch.stack(values), hidden

    def _forward_sequence_continuous(
        self,
        observations: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        initial_hidden: torch.Tensor,
        proprioception: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means: list[torch.Tensor] = []
        values: list[torch.Tensor] = []
        log_stds: list[torch.Tensor] = []
        hidden = initial_hidden
        if proprioception is None or proprioception.shape != (*observations.shape[:2], 3):
            actual = None if proprioception is None else tuple(proprioception.shape)
            raise ValueError(
                "continuous sequence proprioception must have shape "
                f"{(*observations.shape[:2], 3)}, got {actual}"
            )
        for step in range(observations.shape[0]):
            step_mean, step_values, hidden, step_log_std = self.forward_step(
                observations[step],
                previous_actions[step],
                previous_rewards[step],
                episode_starts[step],
                hidden,
                proprioception[step],
            )
            means.append(step_mean)
            values.append(step_values)
            log_stds.append(step_log_std)
        return torch.stack(means), torch.stack(values), hidden, torch.stack(log_stds)


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


def _squashed_gaussian_log_prob(
    distribution: torch.distributions.Normal,
    latent_actions: torch.Tensor,
) -> torch.Tensor:
    """Return tanh-squashed joint log probabilities with Jacobian correction."""

    correction = 2.0 * (
        math.log(2.0)
        - latent_actions
        - nn.functional.softplus(-2.0 * latent_actions)
    )
    return (distribution.log_prob(latent_actions) - correction).sum(dim=-1)


def _bounded_action_log_prob(
    distribution: torch.distributions.Normal,
    actions: torch.Tensor,
) -> torch.Tensor:
    """Score already-bounded actions under a tanh-squashed Gaussian."""

    bounded = actions.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    latent = torch.atanh(bounded)
    return _squashed_gaussian_log_prob(distribution, latent)


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
        self.observation_shape = observation_shape_ or config_observation_shape(
            self.config
        )
        self.performance_profile = resolved_performance_profile(
            self.config.performance_profile,
            self.device,
        )
        self.policy_net = RecurrentActorCriticNetwork(
            self.observation_shape,
            self.config.recurrent_hidden_size,
            has_agent_channel=self.observation_mode == "full",
            continuous=self.config.action_space == "continuous",
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
        proprioception: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.policy_net.continuous:
            return self._step_continuous(
                states, action_masks, previous_actions,
                previous_rewards, episode_starts, hidden, deterministic,
                proprioception,
            )
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

    def _step_continuous(
        self,
        states: torch.Tensor,
        action_masks: torch.Tensor,
        previous_actions: torch.Tensor,
        previous_rewards: torch.Tensor,
        episode_starts: torch.Tensor,
        hidden: torch.Tensor,
        deterministic: bool,
        proprioception: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self.autocast():
            mean, values, next_hidden, log_std = self.forward_step(
                states,
                previous_actions,
                previous_rewards,
                episode_starts,
                hidden,
                proprioception,
            )
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        if deterministic:
            latent_actions = mean
        else:
            latent_actions = distribution.rsample()
        actions = torch.tanh(latent_actions)
        log_probs = _squashed_gaussian_log_prob(distribution, latent_actions)
        return actions, log_probs, values, next_hidden

    def get_actions_stateful(
        self,
        states: np.ndarray,
        action_masks: np.ndarray,
        previous_actions: np.ndarray,
        previous_rewards: np.ndarray,
        episode_starts: np.ndarray,
        hidden: torch.Tensor,
        proprioception: np.ndarray | None = None,
    ) -> tuple[np.ndarray, torch.Tensor]:
        with torch.no_grad():
            if self.policy_net.continuous:
                if proprioception is None:
                    raise ValueError("continuous stateful inference requires proprioception")
                previous_actions = np.asarray(previous_actions, dtype=np.float32)
                if previous_actions.shape != (len(states), 2):
                    raise ValueError(
                        "continuous previous_actions must have shape "
                        f"({len(states)}, 2), got {previous_actions.shape}"
                    )
                prev_act_tensor = torch.as_tensor(
                    previous_actions,
                    dtype=torch.float32, device=self.device,
                )
                actions, _log_probs, _values, next_hidden = self.step(
                    torch.as_tensor(states, dtype=torch.float32, device=self.device),
                    torch.as_tensor(action_masks, dtype=torch.bool, device=self.device),
                    prev_act_tensor,
                    torch.as_tensor(previous_rewards, dtype=torch.float32, device=self.device),
                    torch.as_tensor(episode_starts, dtype=torch.bool, device=self.device),
                    hidden,
                    deterministic=True,
                    proprioception=torch.as_tensor(
                        proprioception,
                        dtype=torch.float32,
                        device=self.device,
                    ),
                )
                return actions.cpu().numpy().astype(np.float32), next_hidden
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
        proprioception: np.ndarray | None = None,
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
            (
                np.zeros((len(states), 2), dtype=np.float32)
                if self.policy_net.continuous
                else np.full(len(states), -1, dtype=np.int64)
            ),
            np.zeros(len(states), dtype=np.float32),
            np.ones(len(states), dtype=np.bool_),
            self.initial_policy_state(len(states)),
            proprioception,
        )
        return actions

    def get_action(
        self,
        state: np.ndarray,
        epsilon: float = 0.0,
        action_mask: np.ndarray | None = None,
        proprioception: np.ndarray | None = None,
    ) -> int | np.ndarray:
        action = self.get_actions(state, epsilon, action_mask, proprioception)[0]
        if self.policy_net.continuous:
            return np.asarray(action, dtype=np.float32)
        return int(action)

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
            "action_space": self.config.action_space,
            "view_size": self.view_size,
            "observation_settings": observation_settings_payload(self.config),
            "distance_shaping_mode": self.config.distance_shaping_mode,
            "continuous_distance_mode": self.config.continuous_distance_mode,
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
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
        ):
            raise ValueError(
                f"checkpoint {path!r} is not a supported MouseMaze checkpoint"
            )
        if payload.get("algorithm") != self.algorithm:
            raise ValueError(f"checkpoint algorithm is {payload.get('algorithm')!r}, expected {self.algorithm!r}")
        if payload.get("action_space") != self.config.action_space:
            raise ValueError(
                f"checkpoint action space is {payload.get('action_space')!r}, "
                f"expected {self.config.action_space!r}"
            )
        _validate_checkpoint_observation_settings(payload, self.config, path)
        _validate_checkpoint_reward_settings(payload, self.config, path)
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
    solved = (
        continuous_position_is_solved(env.continuous_position, env.goal)
        if getattr(env, "action_space", "discrete") == "continuous"
        else env.current_position == env.goal
    )
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


def _in_precision_phase(total_steps: int, config: TrainConfig) -> bool:
    """Return True if training has entered the precision phase."""

    budget = max(config.max_env_steps, 1)
    precision_start = 1.0 - config.precision_fraction
    if config.precision_phase_min_steps > 0:
        min_threshold = 1.0 - config.precision_phase_min_steps / budget
        if 0 < min_threshold < precision_start:
            precision_start = min_threshold
    return total_steps >= budget * precision_start


def _curriculum_final_stage_start_step(config: TrainConfig) -> int | None:
    """Return the finite-budget boundary reserved for unrestricted training."""

    if (
        not config.curriculum_enabled
        or config.curriculum_final_stage_fraction <= 0.0
    ):
        return None
    budget = max(config.max_env_steps, 1)
    return int(budget * (1.0 - config.curriculum_final_stage_fraction))


def _recurrent_ppo_precision_schedule(
    total_steps: int,
    config: TrainConfig,
) -> tuple[float, float]:
    """Return exploration and final cosine-precision PPO schedules."""

    budget = max(config.max_env_steps, 1)
    progress = min(max(total_steps, 0) / budget, 1.0)
    precision_start = 1.0 - config.precision_fraction
    if config.precision_phase_min_steps > 0:
        min_threshold = 1.0 - config.precision_phase_min_steps / budget
        if 0 < min_threshold < precision_start:
            precision_start = min_threshold
    base_entropy = (
        config.continuous_entropy_coef
        if config.action_space == "continuous"
        else config.ppo_entropy_coef
    )
    if progress <= precision_start:
        phase_progress = progress / precision_start
        learning_rate = config.learning_rate * (1.0 - 0.6 * phase_progress)
        entropy = base_entropy
    else:
        precision_duration = max(1.0 - precision_start, 1e-9)
        phase_progress = min(
            max((progress - precision_start) / precision_duration, 0.0),
            1.0,
        )
        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * phase_progress))
        final_learning_rate = (
            config.learning_rate * config.precision_learning_rate_fraction
        )
        learning_rate = final_learning_rate + (
            config.learning_rate * 0.4 - final_learning_rate
        ) * cosine_weight
        final_entropy = base_entropy * config.precision_entropy_fraction
        entropy = final_entropy + (
            base_entropy - final_entropy
        ) * cosine_weight
    return learning_rate, entropy


def _rnd_coefficient(total_steps: int, config: TrainConfig) -> float:
    """Cosine-decay RND to zero before the final third of a finite budget."""

    progress = min(max(total_steps, 0) / max(config.max_env_steps, 1), 1.0)
    return config.rnd_reward_coef * 0.5 * (1.0 + math.cos(math.pi * min(progress * 1.5, 1.0)))


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
    action_sample_count = 0
    requested_action_sum = np.zeros(2, dtype=np.float64)
    requested_action_square_sum = np.zeros(2, dtype=np.float64)
    executed_displacement_sum = np.zeros(2, dtype=np.float64)
    executed_displacement_square_sum = np.zeros(2, dtype=np.float64)
    within_cell_offset_sum = np.zeros(2, dtype=np.float64)
    saturated_action_steps = 0
    no_motion_steps = 0
    low_action_steps = 0
    reward_sum = 0.0
    shaping_reward_sum = 0.0
    saturated_visit_steps = np.zeros(config.eval_episodes, dtype=np.int64)
    episode_step_counts = np.zeros(config.eval_episodes, dtype=np.int64)
    same_cell_dwell = np.zeros(config.eval_episodes, dtype=np.int64)
    max_same_cell_dwell = np.zeros(config.eval_episodes, dtype=np.int64)
    failed_episode = np.zeros(config.eval_episodes, dtype=np.bool_)
    collision_cause_counts: dict[str, int] = {}
    goal_cell_outside_radius_steps = 0
    bucket_totals: dict[str, int] = {}
    bucket_solves: dict[str, int] = {}
    eval_rng = random.Random(resolved_eval_seed(config))
    continuous_evaluation = (
        isinstance(agent, RecurrentPPOAgent) and agent.policy_net.continuous
    )
    evaluation_config = replace(
        config,
        action_space="continuous" if continuous_evaluation else "discrete",
    )
    make_env = maze_factory or (lambda: make_maze(evaluation_config, rng=eval_rng))

    envs = [make_env() for _ in range(config.eval_episodes)]
    buckets = [difficulty_bucket(env.optimal_start_steps) for env in envs]
    for bucket in buckets:
        bucket_totals[bucket] = bucket_totals.get(bucket, 0) + 1
    for env in envs:
        env.reset()
    environment_batch = MazeBatch(
        envs,
        continuous=isinstance(agent, RecurrentPPOAgent) and agent.policy_net.continuous,
    )
    states = environment_batch.observations()
    proprioception = environment_batch.proprioception()
    active = np.ones(len(envs), dtype=np.bool_)
    recurrent_hidden = (
        agent.initial_policy_state(len(envs))
        if isinstance(agent, RecurrentPPOAgent)
        else None
    )
    if isinstance(agent, RecurrentPPOAgent) and agent.policy_net.continuous:
        previous_actions = np.zeros((len(envs), 2), dtype=np.float32)
    else:
        previous_actions = np.full(len(envs), -1, dtype=np.int64)
    previous_rewards = np.zeros(len(envs), dtype=np.float32)
    episode_starts = np.ones(len(envs), dtype=np.bool_)
    continuous_evaluation = environment_batch.continuous
    previous_position = (
        environment_batch.continuous_positions.astype(np.float64)
        if continuous_evaluation
        else environment_batch.positions.astype(np.float64)
    )
    previous_cells = environment_batch.positions.copy()
    position_two_steps_ago = np.full_like(previous_position, np.nan)
    position_three_steps_ago = np.full_like(previous_position, np.nan)
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
                proprioception[active_indices],
            )
            recurrent_hidden.index_copy_(0, hidden_indices, next_hidden)
        else:
            actions = _agent_greedy_actions(agent, state_batch, action_masks)

        if not isinstance(agent, RecurrentPPOAgent) or not agent.policy_net.continuous:
            for batch_idx, env_idx in enumerate(active_indices):
                action = int(actions[batch_idx])
                position = tuple(int(value) for value in environment_batch.positions[env_idx])
                state_action = (position, action)
                if state_action in seen_state_actions[env_idx]:
                    has_repeated_state_action[env_idx] = True
                seen_state_actions[env_idx].add(state_action)

        step_result = environment_batch.step(actions, active_indices)
        if continuous_evaluation:
            assert step_result.requested_actions is not None
            assert step_result.executed_displacements is not None
            assert step_result.collision_causes is not None
            requested = step_result.requested_actions
            executed = step_result.executed_displacements
            offsets = 2.0 * (
                environment_batch.continuous_positions[active_indices]
                - environment_batch.positions[active_indices]
            )
            action_sample_count += len(requested)
            requested_action_sum += requested.sum(axis=0, dtype=np.float64)
            requested_action_square_sum += np.square(requested).sum(
                axis=0,
                dtype=np.float64,
            )
            executed_displacement_sum += executed.sum(axis=0, dtype=np.float64)
            executed_displacement_square_sum += np.square(executed).sum(
                axis=0,
                dtype=np.float64,
            )
            within_cell_offset_sum += offsets.sum(axis=0, dtype=np.float64)
            saturated_action_steps += int(
                np.count_nonzero(np.any(np.abs(requested) >= 0.999, axis=1))
            )
            no_motion_steps += int(
                np.count_nonzero(np.linalg.norm(executed, axis=1) < 1e-6)
            )
            low_action_steps += int(
                np.count_nonzero(
                    np.linalg.norm(requested, axis=1) < CONTINUOUS_LOW_ACTION_THRESHOLD
                )
            )
            causes, cause_counts = np.unique(
                step_result.collision_causes,
                return_counts=True,
            )
            for cause, count in zip(causes, cause_counts):
                collision_cause_counts[str(cause)] = (
                    collision_cause_counts.get(str(cause), 0) + int(count)
                )
            rounded_goal = np.all(
                environment_batch.positions[active_indices]
                == environment_batch.goals[active_indices],
                axis=1,
            )
            goal_cell_outside_radius_steps += int(
                np.count_nonzero(rounded_goal & ~step_result.solved)
            )
        states[active_indices] = step_result.states
        if continuous_evaluation:
            assert step_result.proprioception is not None
            proprioception[active_indices] = step_result.proprioception
        if continuous_evaluation:
            assert step_result.executed_displacements is not None
            previous_actions[active_indices] = step_result.executed_displacements
        else:
            previous_actions[active_indices] = actions
        previous_rewards[active_indices] = step_result.rewards
        reward_sum += float(step_result.rewards.sum())
        if step_result.distance_shaping_rewards is not None:
            shaping_reward_sum += float(step_result.distance_shaping_rewards.sum())
        episode_starts[active_indices] = False
        invalid_moves += int(step_result.invalid.sum())
        total_steps += len(active_indices)
        episode_step_counts[active_indices] += 1
        current_cells = environment_batch.positions[active_indices]
        remained = np.all(current_cells == previous_cells[active_indices], axis=1)
        same_cell_dwell[active_indices] = np.where(
            remained,
            same_cell_dwell[active_indices] + 1,
            1,
        )
        max_same_cell_dwell[active_indices] = np.maximum(
            max_same_cell_dwell[active_indices],
            same_cell_dwell[active_indices],
        )
        previous_cells[active_indices] = current_cells
        if environment_batch.visit_count_encoding == "clipped":
            counts = environment_batch.visit_counts[
                active_indices,
                current_cells[:, 0],
                current_cells[:, 1],
            ]
            saturated_visit_steps[active_indices] += (
                counts >= environment_batch.effective_visit_count_clip
            )

        current_positions = (
            environment_batch.continuous_positions[active_indices].astype(np.float64)
            if continuous_evaluation
            else environment_batch.positions[active_indices].astype(np.float64)
        )
        eligible_for_cycle = environment_batch.steps[active_indices] >= 4
        two_cycle = eligible_for_cycle & (
            np.linalg.norm(
                current_positions - position_two_steps_ago[active_indices],
                axis=1,
            )
            < 1e-3
        ) & (
            np.linalg.norm(
                previous_position[active_indices]
                - position_three_steps_ago[active_indices],
                axis=1,
            )
            < 1e-3
        )
        has_loop[active_indices[two_cycle]] = True
        position_three_steps_ago[active_indices] = position_two_steps_ago[active_indices]
        position_two_steps_ago[active_indices] = previous_position[active_indices]
        previous_position[active_indices] = current_positions

        for batch_idx, env_idx in enumerate(active_indices):

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
                    failed_episode[env_idx] = True
                    failed_grids.append(environment_batch.grids[env_idx].copy())
                    loop_episodes += int(has_loop[env_idx])
                    repeated_state_action_episodes += int(
                        has_repeated_state_action[env_idx]
                    )
                    if continuous_evaluation:
                        failed_final_distances.append(
                            environment_batch._centerline_distance_batch(
                                env_idx,
                                environment_batch.continuous_positions[env_idx],
                            )
                        )
                    else:
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
    timeout_indices = np.flatnonzero(failed_episode)
    requested_action_mean = requested_action_sum / max(action_sample_count, 1)
    requested_action_variance = np.maximum(
        requested_action_square_sum / max(action_sample_count, 1)
        - requested_action_mean**2,
        0.0,
    )
    executed_displacement_mean = (
        executed_displacement_sum / max(action_sample_count, 1)
    )
    executed_displacement_variance = np.maximum(
        executed_displacement_square_sum / max(action_sample_count, 1)
        - executed_displacement_mean**2,
        0.0,
    )
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
        requested_action_mean=(
            tuple(float(value) for value in requested_action_mean)
            if action_sample_count
            else None
        ),
        requested_action_std=(
            tuple(float(value) for value in np.sqrt(requested_action_variance))
            if action_sample_count
            else None
        ),
        executed_displacement_mean=(
            tuple(float(value) for value in executed_displacement_mean)
            if action_sample_count
            else None
        ),
        executed_displacement_std=(
            tuple(float(value) for value in np.sqrt(executed_displacement_variance))
            if action_sample_count
            else None
        ),
        action_saturation_rate=(
            saturated_action_steps / action_sample_count
            if action_sample_count
            else 0.0
        ),
        no_motion_rate=(
            no_motion_steps / action_sample_count
            if action_sample_count
            else 0.0
        ),
        low_action_rate=(
            low_action_steps / action_sample_count
            if action_sample_count
            else 0.0
        ),
        visit_saturation_rate=(
            float(saturated_visit_steps.sum()) / max(int(episode_step_counts.sum()), 1)
        ),
        failed_visit_saturation_rate=(
            float(saturated_visit_steps[timeout_indices].sum())
            / max(int(episode_step_counts[timeout_indices].sum()), 1)
        ),
        max_same_cell_dwell_mean=float(max_same_cell_dwell.mean()),
        failed_max_same_cell_dwell_mean=(
            float(max_same_cell_dwell[timeout_indices].mean())
            if len(timeout_indices)
            else 0.0
        ),
        reward_mean=reward_sum / max(total_steps, 1),
        distance_shaping_reward_mean=shaping_reward_sum / max(total_steps, 1),
        collision_cause_rates={
            cause: count / max(total_steps, 1)
            for cause, count in sorted(collision_cause_counts.items())
        },
        within_cell_offset_mean=(
            tuple(
                float(value)
                for value in within_cell_offset_sum / action_sample_count
            )
            if action_sample_count
            else None
        ),
        goal_cell_outside_radius_rate=(
            goal_cell_outside_radius_steps / max(total_steps, 1)
        ),
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
    benchmark_action_space = (
        "continuous"
        if isinstance(agent, RecurrentPPOAgent) and agent.policy_net.continuous
        else "discrete"
    )
    validation_config = replace(
        config,
        eval_seed=root_seed,
        eval_episodes=episodes,
        action_space=benchmark_action_space,
    )
    final_config = replace(
        config,
        eval_seed=root_seed + 1,
        eval_episodes=episodes,
        action_space=benchmark_action_space,
    )
    stress_config = replace(
        config,
        eval_seed=root_seed + 2,
        eval_episodes=episodes,
        action_space=benchmark_action_space,
    )
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
        self._last_tracker: MetricsTracker | DashboardCharts | None = None
        display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if not display:
            print("[dashboard] No DISPLAY/WAYLAND_DISPLAY; GUI disabled.")
            self.disabled = True
            return
        import pygame

        self.pygame = pygame
        pygame.init()
        try:
            self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            pygame.display.set_caption("MouseMaze Training")
        except pygame.error:
            print("[dashboard] pygame display failed; GUI disabled.")
            self.disabled = True
            self.screen = None

    def draw(
        self,
        state: DashboardState,
        tracker: MetricsTracker | DashboardCharts,
    ) -> None:
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

        repaint_requested = self._process_events()
        if not self.running:
            return
        if (
            repaint_requested
            and self._last_state is not None
            and self._last_tracker is not None
        ):
            self._render(self._last_state, self._last_tracker)

    def _process_events(self) -> bool:
        pg = self.pygame
        assert pg is not None
        repaint_requested = False
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.running = False
            if event.type == pg.KEYDOWN and event.key == pg.K_ESCAPE:
                self.running = False
            if event.type == pg.MOUSEMOTION:
                repaint_requested = True
            if event.type == getattr(pg, "VIDEORESIZE", None):
                resized = (max(320, event.w), max(240, event.h))
                self.screen = pg.display.set_mode(resized, pg.RESIZABLE)
                repaint_requested = True
            if event.type == getattr(pg, "WINDOWRESIZED", None):
                repaint_requested = True
        return repaint_requested

    def _render(
        self,
        state: DashboardState,
        tracker: MetricsTracker | DashboardCharts,
    ) -> None:
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

        if not data:
            return

        x_min = min(episode for episode, _value in data)
        x_span = max(x_max - x_min, 1)

        tick_font = pg.font.SysFont("arial", 11)
        for ratio, raw_value in ((0.0, y_min), (0.5, (y_min + y_max) / 2), (1.0, y_max)):
            ty = py + ph - 1 - int(ratio * (ph - 2))
            label_text = _format_pct(raw_value) if percent else _format_chart_tick(raw_value)
            tick = tick_font.render(label_text, True, (150, 160, 172))
            self.screen.blit(tick, (x + 7, ty - 7))
            pg.draw.line(self.screen, (52, 60, 70), (px, ty), (px + pw, ty))
        candidate_ticks = _episode_tick_values(x_span, pw)
        tick_labels = [str(tick + x_min) for tick in candidate_ticks]
        visible_ticks = _non_overlapping_episode_ticks(
            candidate_ticks,
            x_span,
            pw,
            [tick_font.size(label)[0] for label in tick_labels],
        )
        for offset_tick in visible_ticks:
            tx = px + int(offset_tick / x_span * (pw - 1))
            pg.draw.line(self.screen, (52, 60, 70), (tx, py), (tx, py + ph))
            tick_label = str(offset_tick + x_min)
            tick = tick_font.render(tick_label, True, (150, 160, 172))
            self.screen.blit(tick, (tx - tick.get_width() // 2, py + ph + 4))
        axis_label = tick_font.render("Episode", True, (150, 160, 172))
        self.screen.blit(axis_label, (px + (pw - axis_label.get_width()) // 2, y + h - 19))

        points = []
        for episode, value in data:
            clipped = max(y_min, min(y_max, value))
            nx = px + int((episode - x_min) / x_span * (pw - 1))
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


def _dashboard_process_main(
    messages: multiprocessing.Queue,
    width: int,
    height: int,
    parent_pid: int,
) -> None:
    """Own pygame and its event loop entirely inside the dashboard process."""

    dashboard = Dashboard(width=width, height=height)
    if dashboard.disabled:
        return
    try:
        while dashboard.running and os.getppid() == parent_pid:
            try:
                message = messages.get(timeout=0.05)
            except queue.Empty:
                dashboard.poll()
                continue
            if message is None:
                break
            state, charts = message
            dashboard.draw(state, charts)
    finally:
        dashboard.close()


class DashboardProcess:
    """Non-blocking training-side proxy for the pygame dashboard process."""

    def __init__(self, width: int = 1100, height: int = 720):
        self.running = True
        self.disabled = False
        self._messages: multiprocessing.Queue | None = None
        self._process: multiprocessing.Process | None = None
        display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if not display:
            print("[dashboard] No DISPLAY/WAYLAND_DISPLAY; GUI disabled.")
            self.disabled = True
            return
        context = multiprocessing.get_context("spawn")
        self._messages = context.Queue(maxsize=1)
        self._process = context.Process(
            target=_dashboard_process_main,
            args=(self._messages, width, height, os.getpid()),
            daemon=True,
        )
        self._process.start()

    def draw(self, state: DashboardState, tracker: MetricsTracker) -> None:
        """Publish the newest dashboard frame without delaying training."""

        if self.disabled or not self.running or self._messages is None:
            return
        charts = DashboardCharts(
            reward_history=list(tracker.reward_history),
            loss_history=list(tracker.loss_history),
            greedy_solve_history=list(tracker.greedy_solve_history),
        )
        message = (state, charts)
        try:
            self._messages.put_nowait(message)
        except queue.Full:
            try:
                self._messages.get_nowait()
            except queue.Empty:
                return
            try:
                self._messages.put_nowait(message)
            except queue.Full:
                pass
        self.poll()

    def poll(self) -> None:
        """Notice a closed or failed dashboard without blocking training."""

        if self._process is not None and not self._process.is_alive():
            self.running = False

    def close(self) -> None:
        self.running = False
        process = self._process
        messages = self._messages
        if process is None:
            return
        if process.is_alive() and messages is not None:
            try:
                messages.put_nowait(None)
            except queue.Full:
                try:
                    messages.get_nowait()
                except queue.Empty:
                    pass
                try:
                    messages.put_nowait(None)
                except queue.Full:
                    pass
            process.join(timeout=2.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=2.0)
        if messages is not None:
            messages.close()
            messages.join_thread()
        self._process = None
        self._messages = None


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
        "requested_action_mean": metrics.requested_action_mean,
        "requested_action_std": metrics.requested_action_std,
        "executed_displacement_mean": metrics.executed_displacement_mean,
        "executed_displacement_std": metrics.executed_displacement_std,
        "action_saturation_rate": metrics.action_saturation_rate,
        "no_motion_rate": metrics.no_motion_rate,
        "low_action_rate": metrics.low_action_rate,
        "visit_saturation_rate": metrics.visit_saturation_rate,
        "failed_visit_saturation_rate": metrics.failed_visit_saturation_rate,
        "max_same_cell_dwell_mean": metrics.max_same_cell_dwell_mean,
        "failed_max_same_cell_dwell_mean": metrics.failed_max_same_cell_dwell_mean,
        "reward_mean": metrics.reward_mean,
        "distance_shaping_reward_mean": metrics.distance_shaping_reward_mean,
        "collision_cause_rates": metrics.collision_cause_rates or {},
        "within_cell_offset_mean": metrics.within_cell_offset_mean,
        "goal_cell_outside_radius_rate": metrics.goal_cell_outside_radius_rate,
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
        or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"checkpoint {path!r} is not a supported MouseMaze checkpoint"
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
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"checkpoint {path!r} is not a supported MouseMaze checkpoint"
        )
    return str(payload.get("algorithm", "dqn"))


def checkpoint_schema_version(path: str) -> int:
    """Return the schema version from a supported checkpoint."""

    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path!r} is not a MouseMaze checkpoint")
    version = int(payload.get("schema_version", -1))
    if version not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS:
        raise ValueError(f"checkpoint {path!r} has unsupported schema v{version}")
    return version


def checkpoint_action_space(path: str) -> str:
    """Return and validate the action-space metadata stored in a checkpoint."""

    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise ValueError(
            f"checkpoint {path!r} is not a supported MouseMaze checkpoint"
        )
    action_space = str(payload.get("action_space", "discrete"))
    if action_space not in ACTION_SPACES:
        raise ValueError(f"checkpoint {path!r} has invalid action_space {action_space!r}")
    return action_space


def checkpoint_observation_settings(path: str) -> dict[str, object]:
    """Return normalized observation settings from a supported checkpoint."""

    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") not in SUPPORTED_CHECKPOINT_SCHEMA_VERSIONS
    ):
        raise ValueError(f"checkpoint {path!r} has an unsupported schema")
    settings = payload.get("observation_settings")
    if not isinstance(settings, dict):
        raise ValueError(f"checkpoint {path!r} has no observation settings")
    normalized = dict(settings)
    normalized.setdefault("visit_count_encoding", "clipped")
    return normalized


def checkpoint_distance_shaping_mode(path: str) -> str:
    """Return reward semantics needed by recurrent checkpoint inference."""

    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path!r} is not a MouseMaze checkpoint")
    mode = str(payload.get("distance_shaping_mode", "potential"))
    if mode not in DISTANCE_SHAPING_MODES:
        raise ValueError(f"checkpoint {path!r} has invalid distance shaping {mode!r}")
    return mode


def checkpoint_continuous_distance_mode(path: str) -> str:
    """Return the continuous distance semantics stored by a checkpoint."""

    payload = torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"checkpoint {path!r} is not a MouseMaze checkpoint")
    mode = str(payload.get("continuous_distance_mode", "start_path"))
    if mode not in CONTINUOUS_DISTANCE_MODES:
        raise ValueError(f"checkpoint {path!r} has invalid distance mode {mode!r}")
    return mode


def ensure_agent_matches_config(agent: Agent, config: TrainConfig) -> None:
    if agent.algorithm != config.algorithm:
        raise ValueError(
            "agent algorithm does not match TrainConfig. "
            f"agent={agent.algorithm}, config={config.algorithm}"
        )
    agent_action_space = getattr(getattr(agent, "config", None), "action_space", "discrete")
    if agent_action_space != config.action_space:
        raise ValueError(
            "agent action space does not match TrainConfig. "
            f"agent={agent_action_space}, config={config.action_space}"
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
    entropy_boost: float = 0.0,
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
    state["entropy_boost"] = entropy_boost
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
            "stages": [stage.payload() for stage in curriculum.stages],
        }
    if hard_sampler is not None and hard_sampler.config.hard_maze_fraction > 0.0:
        state["hard_maze_grids"] = {
            shape: [grid.copy() for grid in grids]
            for shape, grids in hard_sampler.hard_grids_by_shape.items()
        }
        state["hard_maze_seen_keys"] = {
            shape: sorted(keys)
            for shape, keys in hard_sampler._seen_hard_grid_keys.items()
        }
        state["hard_maze_candidates_seen"] = dict(
            hard_sampler.hard_candidates_seen_by_shape
        )
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
    eval_every_steps: int | None = None,
    evaluation_fn: Callable[[], EvalMetrics] | None = None,
    allow_new_best: bool = True,
) -> tuple[EvalMetrics | None, int, float, dict[str, torch.Tensor] | None]:
    should_eval = _should_run_eval(
        config,
        completed,
        total_steps,
        last_eval_step,
        eval_every_steps=eval_every_steps,
    )
    if not should_eval:
        return None, last_eval_step, best_eval_rate, best_weights

    evaluation_started = time.perf_counter()
    greedy = evaluation_fn() if evaluation_fn is not None else _eval_greedy(agent, config)
    evaluation_seconds = time.perf_counter() - evaluation_started
    tracker.record_eval(greedy, completed)
    last_eval_step = total_steps
    if on_evaluation is not None:
        on_evaluation(greedy)
    is_new_best = False
    if allow_new_best and greedy.solve_rate > best_eval_rate:
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
            evaluation_seconds=evaluation_seconds,
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
    eval_every_steps: int | None = None,
) -> bool:
    """Return whether evaluation is due under bounded or target-only training."""

    cadence = config.eval_every_steps if eval_every_steps is None else eval_every_steps
    return completed > 0 and (
        last_eval_step == 0
        or (not config.target_only_stop and completed >= config.episodes)
        or total_steps - last_eval_step >= cadence
    )


def _maybe_draw_dashboard(
    dashboard: Dashboard | DashboardProcess | None,
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
    dashboard: Dashboard | DashboardProcess | None,
    completed: int,
    total_steps: int,
    last_eval: EvalMetrics,
    best_eval_rate: float,
    best_weights: dict[str, torch.Tensor] | None,
    start_time: float,
    process_start_cpu: float,
    final_checkpoint_eligible: bool = True,
    best_optimizer_state: dict[str, object] | None = None,
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
        if best_optimizer_state is not None:
            restore_optimizer_state(agent, best_optimizer_state)
        agent.best_greedy_solve_rate = best_eval_rate
        print(f"[train] restored best greedy weights ({best_eval_rate:.1%}).")
    last_eval = _eval_greedy(agent, config)
    tracker.latest_eval = last_eval
    if final_checkpoint_eligible and best_weights is None:
        best_eval_rate = last_eval.solve_rate
        agent.best_greedy_solve_rate = best_eval_rate

    if config.save_path and final_checkpoint_eligible:
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
        final_checkpoint_path = (
            config.save_path
            if final_checkpoint_eligible
            else (
                latest_checkpoint_path(config.save_path)
                if config.save_path is not None
                else None
            )
        )
        logger.log(
            "train_end",
            final_checkpoint_path=final_checkpoint_path,
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
        print("[train] local observation uses visit-count history.")

    expected_shape = config_observation_shape(config)
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
        if checkpoint_schema_version(resume_checkpoint_path) < CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "schema-v10 checkpoints remain inference-compatible but cannot be "
                "resumed after reward and observation semantics changed"
            )
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
    dashboard = DashboardProcess() if config.dashboard_flag else None
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
    curriculum_state = saved_state.get("curriculum")
    restored_stages: list[CurriculumStage] | None = None
    if isinstance(curriculum_state, dict) and isinstance(
        curriculum_state.get("stages"), list
    ):
        restored_stages = [
            CurriculumStage.from_payload(payload)
            for payload in curriculum_state["stages"]
        ]
    sampler = MazeTaskSampler(config, train_rng, stages=restored_stages)
    if isinstance(curriculum_state, dict):
        sampler.curriculum.level = min(
            int(curriculum_state.get("level", 0)),
            len(sampler.curriculum.stages) - 1,
        )
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
    dashboard = DashboardProcess() if config.dashboard_flag else None
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


def _recurrent_sequence_chunks(
    values: torch.Tensor,
    sequence_length: int,
    padding_value: float | int | bool = 0,
) -> torch.Tensor:
    """Pack ``[time, env, ...]`` values into environment-major sequences."""

    time_steps, env_count = values.shape[:2]
    sequence_count = math.ceil(time_steps / sequence_length)
    padded_steps = sequence_count * sequence_length
    if padded_steps != time_steps:
        padded = torch.full(
            (padded_steps, env_count, *values.shape[2:]),
            padding_value,
            dtype=values.dtype,
            device=values.device,
        )
        padded[:time_steps].copy_(values)
        values = padded
    environment_major = values.transpose(0, 1)
    return environment_major.reshape(
        env_count * sequence_count,
        sequence_length,
        *values.shape[2:],
    )


def _recurrent_ppo_update(
    agent: RecurrentPPOAgent,
    config: TrainConfig,
    states: torch.Tensor,
    proprioception: torch.Tensor,
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

    is_continuous = agent.policy_net.continuous
    time_steps, env_count = actions.shape[:2]

    effective_entropy_coefficient = (
        (
            config.continuous_entropy_coef
            if is_continuous
            else config.ppo_entropy_coef
        )
        if entropy_coefficient is None
        else float(entropy_coefficient)
    )
    sequence_length = config.recurrent_sequence_length
    normalized_advantages = (advantages - advantages.mean()) / (
        advantages.std(unbiased=False) + 1e-8
    )
    state_chunks = _recurrent_sequence_chunks(states, sequence_length)
    proprioception_chunks = _recurrent_sequence_chunks(
        proprioception,
        sequence_length,
    )
    mask_chunks = _recurrent_sequence_chunks(
        action_masks,
        sequence_length,
        padding_value=True,
    )
    action_chunks = _recurrent_sequence_chunks(actions, sequence_length)
    previous_action_chunks = _recurrent_sequence_chunks(
        previous_actions,
        sequence_length,
        padding_value=-1.0 if is_continuous else -1,
    )
    previous_reward_chunks = _recurrent_sequence_chunks(
        previous_rewards,
        sequence_length,
    )
    episode_start_chunks = _recurrent_sequence_chunks(
        episode_starts,
        sequence_length,
        padding_value=True,
    )
    old_log_chunks = _recurrent_sequence_chunks(old_log_probs, sequence_length)
    old_value_chunks = _recurrent_sequence_chunks(old_values, sequence_length)
    advantage_chunks = _recurrent_sequence_chunks(
        normalized_advantages,
        sequence_length,
    )
    return_chunks = _recurrent_sequence_chunks(returns, sequence_length)
    valid_steps = torch.ones(
        time_steps,
        env_count,
        dtype=torch.bool,
        device=agent.device,
    )
    valid_chunks = _recurrent_sequence_chunks(
        valid_steps,
        sequence_length,
        padding_value=False,
    )
    chunk_starts = torch.arange(
        0,
        time_steps,
        sequence_length,
        device=agent.device,
    )
    initial_hidden_chunks = hidden_states.index_select(0, chunk_starts).transpose(0, 1)
    initial_hidden_chunks = initial_hidden_chunks.reshape(
        state_chunks.shape[0],
        hidden_states.shape[-1],
    )
    chunk_count = state_chunks.shape[0]
    totals = PPOUpdateMetrics()
    metric_sums = torch.zeros(5, dtype=torch.float64, device=agent.device)
    continuous_entropy_sum = torch.zeros(2, dtype=torch.float64, device=agent.device)
    action_std_sum = torch.zeros((), dtype=torch.float64, device=agent.device)
    action_std_min = torch.full((), torch.inf, dtype=torch.float64, device=agent.device)
    action_std_max = torch.zeros((), dtype=torch.float64, device=agent.device)
    update_count = 0

    for epoch in range(config.ppo_epochs):
        epoch_kl_total = torch.zeros((), dtype=torch.float64, device=agent.device)
        epoch_update_count = 0
        order = torch.randperm(chunk_count, device=agent.device)
        for offset in range(
            0,
            chunk_count,
            config.recurrent_sequence_minibatch_size,
        ):
            selected = order[
                offset : offset + config.recurrent_sequence_minibatch_size
            ]
            state_batch = state_chunks.index_select(0, selected).transpose(0, 1)
            proprioception_batch = proprioception_chunks.index_select(
                0,
                selected,
            ).transpose(0, 1)
            mask_batch = mask_chunks.index_select(0, selected).transpose(0, 1)
            if is_continuous:
                mask_batch = torch.ones_like(mask_batch)
            action_batch = action_chunks.index_select(0, selected).transpose(0, 1)
            previous_action_batch = previous_action_chunks.index_select(
                0,
                selected,
            ).transpose(0, 1)
            previous_reward_batch = previous_reward_chunks.index_select(
                0,
                selected,
            ).transpose(0, 1)
            start_batch = episode_start_chunks.index_select(0, selected).transpose(0, 1)
            old_log_batch = old_log_chunks.index_select(0, selected).transpose(0, 1)
            old_value_batch = old_value_chunks.index_select(0, selected).transpose(0, 1)
            advantage_batch = advantage_chunks.index_select(0, selected).transpose(0, 1)
            return_batch = return_chunks.index_select(0, selected).transpose(0, 1)
            valid = valid_chunks.index_select(0, selected).transpose(0, 1)
            initial_hidden = initial_hidden_chunks.index_select(0, selected)

            with agent.autocast():
                sequence_result = agent.forward_sequence(
                    state_batch,
                    previous_action_batch,
                    previous_reward_batch,
                    start_batch,
                    initial_hidden,
                    proprioception_batch if is_continuous else None,
                )
            if is_continuous:
                means, new_values, _hidden, log_stds = sequence_result
                stds = log_stds.exp()
                distribution = torch.distributions.Normal(means, stds)
                new_log_probs = _bounded_action_log_prob(distribution, action_batch)
            else:
                logits, new_values, _hidden = sequence_result
                logits = logits.masked_fill(~mask_batch, -torch.inf)
                distribution = torch.distributions.Categorical(logits=logits)
                new_log_probs = distribution.log_prob(action_batch)
            log_ratio = new_log_probs - old_log_batch
            ratio = torch.exp(log_ratio)
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
            if is_continuous:
                entropy_latent = distribution.rsample()
                sampled_entropy_per_dimension = -(
                    distribution.log_prob(entropy_latent)
                    - 2.0
                    * (
                        math.log(2.0)
                        - entropy_latent
                        - nn.functional.softplus(-2.0 * entropy_latent)
                    )
                )
                entropy = sampled_entropy_per_dimension.sum(dim=2)[valid].mean()
                per_dimension_entropy = sampled_entropy_per_dimension[valid].mean(dim=0)
                continuous_entropy_sum += per_dimension_entropy.detach().to(torch.float64)
                valid_stds = stds[valid]
                action_std_sum += valid_stds.mean().detach().to(torch.float64)
                action_std_min = torch.minimum(
                    action_std_min,
                    valid_stds.min().detach().to(torch.float64),
                )
                action_std_max = torch.maximum(
                    action_std_max,
                    valid_stds.max().detach().to(torch.float64),
                )
            else:
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
            approx_kl = ((ratio - 1.0) - log_ratio)[valid].mean().detach()
            metric_sums += torch.stack(
                (loss, policy_loss, value_loss, entropy, approx_kl)
            ).detach().to(torch.float64)
            epoch_kl_total += approx_kl.to(torch.float64)
            epoch_update_count += 1
            update_count += 1
        totals.epochs = epoch + 1
        epoch_mean_kl = float(
            (epoch_kl_total / max(epoch_update_count, 1)).item()
        )
        if epoch_mean_kl > config.ppo_target_kl:
            break
    if update_count:
        metric_means = (metric_sums / update_count).cpu().tolist()
        (
            totals.loss,
            totals.policy_loss,
            totals.value_loss,
            totals.entropy,
            totals.approx_kl,
        ) = (float(value) for value in metric_means)
        if is_continuous:
            continuous_metrics = torch.cat(
                (
                    continuous_entropy_sum / update_count,
                    (action_std_sum / update_count).reshape(1),
                    action_std_min.reshape(1),
                    action_std_max.reshape(1),
                )
            ).cpu().tolist()
            totals.continuous_entropy_per_dimension = tuple(
                float(value) for value in continuous_metrics[:2]
            )
            totals.action_std_mean = float(continuous_metrics[2])
            totals.action_std_min = float(continuous_metrics[3])
            totals.action_std_max = float(continuous_metrics[4])
    return totals


def _curriculum_stage_eval(
    agent: Agent,
    config: TrainConfig,
    curriculum: CurriculumController,
) -> EvalMetrics:
    stage = curriculum.current_stage
    stage_config = replace(
        config,
        maze_size=stage.maze_size,
        eval_episodes=config.curriculum_eval_episodes,
    )
    if stage.distance_range is None and stage.complexity_high is None:
        return _eval_greedy(
            agent,
            stage_config,
        )
    stage_rng = random.Random(resolved_eval_seed(config) + 100_000 * (curriculum.level + 1))
    return _eval_greedy(
        agent,
        stage_config,
        maze_factory=lambda: make_maze(
            stage_config,
            rng=stage_rng,
            target_range=stage.distance_range,
            target_complexity=stage.complexity_high,
        ),
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
    dashboard = DashboardProcess() if config.dashboard_flag else None
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
    entropy_boost = float(saved_state.get("entropy_boost", 0.0))

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
    curriculum_state = saved_state.get("curriculum")
    restored_stages: list[CurriculumStage] | None = None
    if isinstance(curriculum_state, dict) and isinstance(
        curriculum_state.get("stages"), list
    ):
        restored_stages = [
            CurriculumStage.from_payload(payload)
            for payload in curriculum_state["stages"]
        ]
    sampler = MazeTaskSampler(config, train_rng, stages=restored_stages)
    if config.hard_maze_fraction > 0.0:
        sampler.restore_hard_grids(
            saved_state.get("hard_maze_grids"),
            saved_state.get("hard_maze_seen_keys"),
            saved_state.get("hard_maze_candidates_seen"),
            saved_state.get("hard_maze_validation_solve_rate"),
        )
    if isinstance(curriculum_state, dict):
        sampler.curriculum.level = min(
            int(curriculum_state.get("level", 0)),
            len(sampler.curriculum.stages) - 1,
        )
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
    environment_batch = MazeBatch(environments, continuous=agent.policy_net.continuous)
    states = environment_batch.observations()
    proprioception = environment_batch.proprioception()
    action_masks_np = environment_batch.valid_action_masks()
    hidden = agent.initial_policy_state(env_count)
    if agent.policy_net.continuous:
        previous_actions_np = np.zeros((env_count, 2), dtype=np.float32)
    else:
        previous_actions_np = np.full(env_count, -1, dtype=np.int64)
    previous_rewards_np = np.zeros(env_count, dtype=np.float32)
    episode_starts_np = np.ones(env_count, dtype=np.bool_)

    def reset_training_batch() -> None:
        """Start fresh recurrent contexts after restoring frozen-best weights."""

        nonlocal states
        nonlocal proprioception
        nonlocal hidden
        nonlocal environment_batch
        nonlocal previous_actions_np
        nonlocal action_masks_np
        environment_batch = MazeBatch(
            [prefetcher.next() for _ in range(env_count)],
            continuous=agent.policy_net.continuous,
        )
        states = environment_batch.observations()
        proprioception = environment_batch.proprioception()
        action_masks_np = environment_batch.valid_action_masks()
        hidden = agent.initial_policy_state(env_count)
        if agent.policy_net.continuous:
            previous_actions_np.fill(0.0)
        else:
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
    proprioception_size = 3 if agent.policy_net.continuous else 0
    proprioception_staging = torch.empty(
        env_count,
        proprioception_size,
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    mask_staging = torch.empty(env_count, 4, dtype=torch.bool, pin_memory=pin_memory)
    last_eval = EvalMetrics()
    last_dashboard_episode = -config.dashboard_every
    rollout_steps = config.ppo_rollout_steps
    rollout_shape = (rollout_steps, env_count)
    is_continuous_mode = agent.policy_net.continuous
    rollout_states_buffer = torch.empty(
        *rollout_shape,
        *agent.observation_shape,
        dtype=torch.float32,
        device=agent.device,
    )
    rollout_next_states_buffer = torch.empty(
        *rollout_shape,
        *agent.observation_shape,
        dtype=torch.float32,
        pin_memory=pin_memory,
    )
    rollout_proprioception_buffer = torch.empty(
        *rollout_shape,
        proprioception_size,
        dtype=torch.float32,
        device=agent.device,
    )
    rollout_masks_buffer = torch.empty(
        *rollout_shape,
        4,
        dtype=torch.bool,
        device=agent.device,
    )
    if is_continuous_mode:
        rollout_actions_buffer = torch.empty(
            *rollout_shape,
            2,
            dtype=torch.float32,
            device=agent.device,
        )
    else:
        rollout_actions_buffer = torch.empty(
            *rollout_shape,
            dtype=torch.long,
            device=agent.device,
        )
    rollout_previous_actions_buffer = torch.empty_like(rollout_actions_buffer)
    rollout_previous_rewards_buffer = torch.empty(*rollout_shape, device=agent.device)
    rollout_episode_starts_buffer = torch.empty(
        *rollout_shape,
        dtype=torch.bool,
        device=agent.device,
    )
    rollout_hidden_buffer = torch.empty(
        *rollout_shape,
        config.recurrent_hidden_size,
        device=agent.device,
    )
    rollout_log_probs_buffer = torch.empty(*rollout_shape, device=agent.device)
    rollout_values_buffer = torch.empty(*rollout_shape, device=agent.device)
    rollout_rewards_buffer = torch.empty(*rollout_shape, device=agent.device)
    rollout_dones_buffer = torch.empty(*rollout_shape, device=agent.device)
    final_stage_start_step = _curriculum_final_stage_start_step(config)

    while (
        (not limits_enabled or total_steps < config.max_env_steps)
        and (not limits_enabled or completed < config.episodes)
        and not target_confirmed
    ):
        if (
            final_stage_start_step is not None
            and total_steps >= final_stage_start_step
            and sampler.curriculum.force_complete()
        ):
            prefetcher.reset()
            reset_training_batch()
            last_eval_step = 0
            _update_training_state(
                agent,
                completed,
                total_steps,
                last_eval_step,
                train_rng,
                curriculum=sampler.curriculum,
                hard_sampler=sampler,
                precision_recovery=precision_recovery_payload(),
                entropy_boost=entropy_boost,
            )
            if logger.enabled:
                logger.log(
                    "curriculum",
                    total_steps=total_steps,
                    level=sampler.curriculum.level,
                    stage=sampler.curriculum.current_stage.payload(),
                    promoted=True,
                    promotion_reason="budget_reserve",
                    stage_metrics=None,
                    sampling_mix=sampler.sampling_mix(),
                    hard_variants_added=0,
                    hard_maze_pool_size=len(sampler.hard_grids),
                    hard_maze_candidates_seen=sampler.hard_candidates_seen,
                    validation_solve_rate=sampler.validation_solve_rate,
                    active_eval_every_steps=config.post_curriculum_eval_every_steps,
                )
        collector_started = time.perf_counter()
        rollout_states = rollout_states_buffer
        rollout_next_states = rollout_next_states_buffer
        rollout_proprioception = rollout_proprioception_buffer
        rollout_masks = rollout_masks_buffer
        rollout_actions = rollout_actions_buffer
        rollout_previous_actions = rollout_previous_actions_buffer
        rollout_previous_rewards = rollout_previous_rewards_buffer
        rollout_episode_starts = rollout_episode_starts_buffer
        rollout_hidden = rollout_hidden_buffer
        rollout_log_probs = rollout_log_probs_buffer
        rollout_values = rollout_values_buffer
        rollout_rewards = rollout_rewards_buffer
        rollout_dones = rollout_dones_buffer
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
            proprioception_staging.copy_(torch.from_numpy(proprioception))
            proprioception_tensor = proprioception_staging.to(
                agent.device,
                non_blocking=pin_memory,
            )
            mask_staging.copy_(torch.from_numpy(action_masks_np))
            mask_tensor = mask_staging.to(agent.device, non_blocking=pin_memory)
            transfer_seconds += time.perf_counter() - transfer_started
            if is_continuous_mode:
                previous_action_tensor = torch.as_tensor(
                    previous_actions_np,
                    dtype=torch.float32,
                    device=agent.device,
                )
            else:
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
            rollout_proprioception[step].copy_(proprioception_tensor)
            rollout_masks[step].copy_(mask_tensor)
            rollout_previous_actions[step].copy_(previous_action_tensor)
            rollout_previous_rewards[step].copy_(previous_reward_tensor)
            rollout_episode_starts[step].copy_(episode_start_tensor)
            rollout_hidden[step].copy_(hidden)
            with torch.inference_mode():
                actions_t, log_probs_t, values_t, hidden = agent.step(
                    state_tensor,
                    mask_tensor,
                    previous_action_tensor,
                    previous_reward_tensor,
                    episode_start_tensor,
                    hidden,
                    deterministic=False,
                    proprioception=(
                        proprioception_tensor if is_continuous_mode else None
                    ),
                )
            if is_continuous_mode:
                actions_np = actions_t.cpu().numpy().astype(np.float32)
            else:
                actions_np = actions_t.cpu().numpy().astype(np.int64)
            environment_started = time.perf_counter()
            result = environment_batch.step(actions_np)
            environment_seconds += time.perf_counter() - environment_started
            rollout_next_states[step].copy_(torch.from_numpy(result.states))
            rollout_actions[step].copy_(actions_t)
            rollout_log_probs[step].copy_(log_probs_t)
            rollout_values[step].copy_(values_t)
            rollout_rewards[step].copy_(
                torch.as_tensor(result.rewards, device=agent.device)
            )
            rollout_dones[step].copy_(
                torch.as_tensor(result.dones, dtype=torch.float32, device=agent.device)
            )
            failed_training_grids = [
                environment_batch.grids[int(index)].copy()
                for index in np.flatnonzero(result.dones)
                if result.timeout[int(index)]
            ]
            if failed_training_grids and config.hard_maze_fraction > 0.0:
                sampler.add_failed_grids(failed_training_grids)
            replaced_indices: list[int] = []
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
                    replaced_indices.append(int(index))
            total_steps += env_count
            agent.total_env_steps += env_count
            actual_steps = step + 1
            states = result.states
            if is_continuous_mode:
                assert result.proprioception is not None
                proprioception = result.proprioception
            action_masks_np = result.action_masks
            if replaced_indices:
                replacement_indices = np.asarray(replaced_indices, dtype=np.int64)
                states[replacement_indices] = environment_batch.observations(
                    replacement_indices
                )
                if is_continuous_mode:
                    proprioception[replacement_indices] = environment_batch.proprioception(
                        replacement_indices
                    )
                action_masks_np[replacement_indices] = (
                    environment_batch.valid_action_masks(replacement_indices)
                )
            if is_continuous_mode:
                reset_val = np.zeros((env_count, 2), dtype=np.float32)
                assert result.executed_displacements is not None
                previous_actions_np = np.where(
                    result.dones[:, np.newaxis],
                    reset_val,
                    result.executed_displacements,
                )
            else:
                previous_actions_np = np.where(result.dones, -1, actions_np)
            previous_rewards_np = np.where(result.dones, 0.0, result.rewards).astype(np.float32)
            episode_starts_np = result.dones.copy()

        if actual_steps == 0:
            break
        collector_seconds = time.perf_counter() - collector_started
        rollout_states = rollout_states[:actual_steps]
        rollout_next_states = rollout_next_states[:actual_steps].to(
            agent.device,
            non_blocking=pin_memory,
        )
        rollout_proprioception = rollout_proprioception[:actual_steps]
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
        if agent.rnd is not None:
            rnd_coefficient = _rnd_coefficient(total_steps, config)
        if agent.rnd is not None and rnd_coefficient > 0.0:
            intrinsic_rewards = agent.rnd.bonus_and_update(
                rollout_next_states,
                config.rnd_reward_clip,
            )
        combined_rewards = rollout_rewards + rnd_coefficient * intrinsic_rewards
        next_state_tensor = torch.as_tensor(states, dtype=torch.float32, device=agent.device)
        next_proprioception_tensor = torch.as_tensor(
            proprioception,
            dtype=torch.float32,
            device=agent.device,
        )
        with torch.inference_mode():
            _actions, _log_probs, next_values, _next_hidden = agent.step(
                next_state_tensor,
                torch.as_tensor(
                    action_masks_np,
                    dtype=torch.bool,
                    device=agent.device,
                ),
                torch.as_tensor(
                    previous_actions_np,
                    dtype=torch.float32 if is_continuous_mode else torch.long,
                    device=agent.device,
                ),
                torch.as_tensor(previous_rewards_np, dtype=torch.float32, device=agent.device),
                torch.as_tensor(episode_starts_np, dtype=torch.bool, device=agent.device),
                hidden,
                deterministic=True,
                proprioception=(
                    next_proprioception_tensor if is_continuous_mode else None
                ),
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
        effective_entropy_coefficient = (
            (
                config.continuous_entropy_coef
                if is_continuous_mode
                else config.ppo_entropy_coef
            )
            + entropy_boost
        ) if entropy_boost > 0 else entropy_coefficient
        for group in agent.optimizer.param_groups:
            group["lr"] = learning_rate
        updates_before_rollout = agent.update_count
        update_metrics = _recurrent_ppo_update(
            agent,
            config,
            rollout_states,
            rollout_proprioception,
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
            entropy_coefficient=effective_entropy_coefficient,
        )
        tracker.record_loss(update_metrics.loss, completed)
        if not is_continuous_mode and update_metrics.entropy < config.ppo_entropy_floor:
            entropy_boost = min(entropy_boost + 0.01, 0.1)
        elif not is_continuous_mode and entropy_boost > 0:
            entropy_boost = max(entropy_boost - 0.005, 0.0)
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
                schedule={
                    "budget_steps": config.max_env_steps,
                    "progress": min(total_steps / max(config.max_env_steps, 1), 1.0),
                    "phase": (
                        "precision"
                        if _in_precision_phase(total_steps, config)
                        else "exploration"
                    ),
                },
                active_eval_every_steps=(
                    config.post_curriculum_eval_every_steps
                    if sampler.curriculum.complete
                    else config.eval_every_steps
                ),
                curriculum_stage=sampler.curriculum.current_stage.payload(),
                generation=prefetcher.telemetry(),
                extrinsic_reward_mean=float(rollout_rewards.mean().item()),
                intrinsic_reward_mean=float(intrinsic_rewards.mean().item()),
                ppo={
                    "loss": update_metrics.loss,
                    "policy_loss": update_metrics.policy_loss,
                    "value_loss": update_metrics.value_loss,
                    "entropy": update_metrics.entropy,
                    "continuous_entropy_per_dimension": (
                        update_metrics.continuous_entropy_per_dimension
                    ),
                    "action_std_mean": update_metrics.action_std_mean,
                    "action_std_min": update_metrics.action_std_min,
                    "action_std_max": update_metrics.action_std_max,
                    "entropy_boost": entropy_boost,
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
            entropy_boost=entropy_boost,
        )
        agent.training_state["target_confirmed"] = False
        _save_latest_checkpoint(
            agent,
            config,
            logger,
            total_steps,
            "periodic_latest",
        )

        became_complete = False

        def record_evaluation(metrics: EvalMetrics) -> None:
            nonlocal became_complete
            promoted = False
            stage_metrics = metrics
            if config.curriculum_enabled and not sampler.curriculum.complete:
                promoted = sampler.curriculum.record_validation(stage_metrics)
                if promoted:
                    became_complete = sampler.curriculum.complete
                    prefetcher.reset()
                    reset_training_batch()
            sampler.record_validation_solve_rate(metrics.solve_rate)
            hard_variants_added = 0
            _update_training_state(
                agent,
                completed,
                total_steps,
                total_steps,
                train_rng,
                curriculum=sampler.curriculum,
                hard_sampler=sampler,
                precision_recovery=precision_recovery_payload(),
                entropy_boost=entropy_boost,
            )
            agent.training_state["target_confirmed"] = False
            if logger.enabled:
                logger.log(
                    "curriculum",
                    level=sampler.curriculum.level,
                    stage=sampler.curriculum.current_stage.payload(),
                    promoted=promoted,
                    promotion_reason="performance" if promoted else None,
                    stage_metrics=_eval_metrics_payload(stage_metrics),
                    sampling_mix=sampler.sampling_mix(),
                    hard_variants_added=hard_variants_added,
                    hard_maze_pool_size=len(sampler.hard_grids),
                    hard_maze_candidates_seen=sampler.hard_candidates_seen,
                    validation_solve_rate=sampler.validation_solve_rate,
                    active_eval_every_steps=(
                        config.post_curriculum_eval_every_steps
                        if sampler.curriculum.complete
                        else config.eval_every_steps
                    ),
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
                entropy_boost=entropy_boost,
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
        curriculum_complete_before_eval = (
            not config.curriculum_enabled or sampler.curriculum.complete
        )
        active_eval_every_steps = (
            config.post_curriculum_eval_every_steps
            if curriculum_complete_before_eval
            else config.eval_every_steps
        )
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
            eval_every_steps=active_eval_every_steps,
            evaluation_fn=(
                None
                if curriculum_complete_before_eval
                else lambda: _curriculum_stage_eval(agent, config, sampler.curriculum)
            ),
            allow_new_best=curriculum_complete_before_eval,
        )
        if became_complete:
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
                0,
                start_time,
                process_start_cpu,
                on_evaluation=record_evaluation,
                on_new_best=record_new_best,
                eval_every_steps=config.post_curriculum_eval_every_steps,
                allow_new_best=True,
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
                confirmation_variants_added = 0
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
                        entropy_boost=entropy_boost,
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
                and _in_precision_phase(total_steps, config)
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
                    entropy_boost=entropy_boost,
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
        entropy_boost=entropy_boost,
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
        final_checkpoint_eligible=sampler.curriculum.complete,
        best_optimizer_state=best_optimizer_state,
    )


def _train_ppo(
    agent: MaskedPPOAgent,
    config: TrainConfig,
    logger: _TrainingLogger,
    start_time: float,
    process_start_cpu: float,
) -> MetricsTracker:
    tracker = MetricsTracker()
    dashboard = DashboardProcess() if config.dashboard_flag else None
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
    channel_panel_rect: Rect | None = None


@dataclass(slots=True)
class InferenceSession:
    """Persistent pygame resources shared across inference mazes."""

    pygame: Any
    screen: Any
    clock: Any
    fps: int
    show_input_channels: bool


INFERENCE_CHANNEL_PANEL_WIDTH = 300
INFERENCE_CHANNEL_PANEL_GAP = 12


def inference_layout(
    window_size: tuple[int, int],
    maze_shape: tuple[int, int],
    show_input_channels: bool = False,
) -> InferenceLayout:
    """Fit the maze and optional input-channel panel above the status HUD."""

    width, height = (max(1, int(value)) for value in window_size)
    rows, cols = (int(value) for value in maze_shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("maze dimensions must be positive")

    margin = max(4, min(16, min(width, height) // 30))
    hud_height = max(44, min(68, height // 7))
    hud_y = max(margin, height - hud_height - margin)
    content_width = max(1, width - margin * 2)
    available_height = max(1, hud_y - margin * 2)
    panel_width = 0
    panel_gap = 0
    if show_input_channels:
        panel_width = min(
            INFERENCE_CHANNEL_PANEL_WIDTH,
            max(1, content_width // 2),
        )
        panel_gap = min(
            INFERENCE_CHANNEL_PANEL_GAP,
            max(0, content_width - panel_width - 1),
        )
    maze_available_width = max(1, content_width - panel_width - panel_gap)
    cell_size = max(
        1,
        min(80, maze_available_width // cols, available_height // rows),
    )
    maze_width = cols * cell_size
    maze_height = rows * cell_size
    maze_x = margin + max(0, (maze_available_width - maze_width) // 2)
    maze_y = margin + max(0, (available_height - maze_height) // 2)
    channel_panel_rect = None
    if show_input_channels:
        channel_panel_rect = (
            margin + maze_available_width + panel_gap,
            margin,
            panel_width,
            available_height,
        )
    return InferenceLayout(
        maze_rect=(maze_x, maze_y, maze_width, maze_height),
        hud_rect=(margin, hud_y, max(1, width - margin * 2), hud_height),
        cell_size=cell_size,
        channel_panel_rect=channel_panel_rect,
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
    center_x = round(center_x)
    center_y = round(center_y)
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


def _initial_inference_window(
    pg,
    rows: int,
    cols: int,
    *,
    show_input_channels: bool = False,
) -> tuple[int, int]:
    """Choose a useful initial size while staying within the active display."""

    info = pg.display.Info()
    display_width = info.current_w if info.current_w > 0 else 1280
    display_height = info.current_h if info.current_h > 0 else 720
    target_cell = max(16, min(48, (display_height // 2 - 96) // max(rows, 1)))
    panel_width = (
        INFERENCE_CHANNEL_PANEL_WIDTH + INFERENCE_CHANNEL_PANEL_GAP
        if show_input_channels
        else 0
    )
    width = max(360, cols * target_cell + 32 + panel_width)
    height = max(260, rows * target_cell + 100)
    return min(width, max(320, display_width - 80)), min(
        height,
        max(240, display_height - 80),
    )


def _create_inference_session(
    maze_shape: tuple[int, int],
    fps: int,
    show_input_channels: bool,
) -> InferenceSession:
    """Initialize the pygame resources used for one inference invocation."""

    if fps < 1:
        raise ValueError("inference FPS must be positive")

    import pygame

    os.environ["SDL_VIDEO_CENTERED"] = "1"
    pygame.init()
    try:
        rows, cols = maze_shape
        window_size = _initial_inference_window(
            pygame,
            rows,
            cols,
            show_input_channels=show_input_channels,
        )
        screen = pygame.display.set_mode(window_size, pygame.RESIZABLE)
        pygame.display.set_caption("MouseMaze Inference")
        return InferenceSession(
            pygame=pygame,
            screen=screen,
            clock=pygame.time.Clock(),
            fps=fps,
            show_input_channels=show_input_channels,
        )
    except BaseException:
        pygame.quit()
        raise


def _close_inference_session(session: InferenceSession) -> None:
    """Release a persistent inference session."""

    session.pygame.quit()


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


def inference_channel_specs(
    state: np.ndarray,
    observation_mode: str,
    remaining_time_channel: bool = DEFAULT_REMAINING_TIME_CHANNEL,
    visit_count_channel: bool = DEFAULT_VISIT_COUNT_CHANNEL,
    action_space: str = "discrete",
) -> tuple[tuple[str, np.ndarray, tuple[int, int, int]], ...]:
    """Return labels, maps, and display colors for one model observation."""

    channel_colors = {
        "Walls": (37, 43, 50),
        "Mouse": (75, 167, 232),
        "Goal": (255, 201, 45),
        "Time remaining": (64, 178, 154),
        "Visit count": (225, 116, 86),
    }
    channel_names = ["Walls"]
    if observation_mode == "full":
        channel_names.append("Mouse")
    channel_names.append("Goal")
    if remaining_time_channel:
        channel_names.append("Time remaining")
    if visit_count_channel:
        channel_names.append("Visit count")
    expected_channels = len(channel_names)
    if state.ndim != 3 or state.shape[0] != expected_channels:
        raise ValueError(
            f"expected {expected_channels} observation channels for "
            f"{observation_mode!r} mode, got shape {state.shape}"
        )
    return tuple(
        (name, state[index], channel_colors[name])
        for index, name in enumerate(channel_names)
    )


def _channel_map_surface(
    pg,
    channel: np.ndarray,
    active_color: tuple[int, int, int],
):
    """Build a small RGB surface for a normalized observation channel."""

    rows, cols = channel.shape
    surface = pg.Surface((cols, rows))
    background = np.asarray((239, 242, 237), dtype=np.float32)
    foreground = np.asarray(active_color, dtype=np.float32)
    values = np.nan_to_num(channel, nan=0.0, posinf=1.0, neginf=0.0)
    values = np.clip(values, 0.0, 1.0)
    colors = np.rint(
        background[np.newaxis, np.newaxis, :]
        + values[:, :, np.newaxis]
        * (foreground - background)[np.newaxis, np.newaxis, :]
    ).astype(np.uint8)
    pg.surfarray.blit_array(surface, np.transpose(colors, (1, 0, 2)))
    return surface


def _continuous_input_specs(
    proprioception: np.ndarray | None,
) -> tuple[tuple[str, float], tuple[str, float], tuple[str, float]]:
    """Return the labeled scalar inputs used by a continuous policy."""

    if proprioception is None or np.asarray(proprioception).shape != (3,):
        actual = None if proprioception is None else np.asarray(proprioception).shape
        raise ValueError(
            "continuous input panel expected proprioception shape (3,), "
            f"got {actual}"
        )
    row_offset, col_offset, previous_collision = np.asarray(
        proprioception,
        dtype=np.float32,
    )
    return (
        ("Within-cell row", float(row_offset)),
        ("Within-cell col", float(col_offset)),
        ("Previous collision", float(previous_collision)),
    )


def _draw_signed_input_gauge(
    pg,
    screen,
    label_font,
    label: str,
    value: float,
    rect: Rect,
) -> None:
    """Draw one signed ``[-1, 1]`` scalar with its exact numeric value."""

    x, y, width, height = rect
    text = label_font.render(f"{label}: {value:+.2f}", True, (205, 213, 222))
    screen.blit(text, (x, y))
    track_y = y + max(14, height - 9)
    track_rect = (x, track_y, max(1, width), 7)
    pg.draw.rect(screen, (73, 84, 96), track_rect, border_radius=3)
    center_x = x + width // 2
    pg.draw.line(
        screen,
        (151, 161, 171),
        (center_x, track_y),
        (center_x, track_y + 6),
    )
    normalized = (float(np.clip(value, -1.0, 1.0)) + 1.0) / 2.0
    marker_x = x + round(normalized * max(0, width - 1))
    pg.draw.circle(screen, (75, 167, 232), (marker_x, track_y + 3), 4)


def _draw_collision_input(
    pg,
    screen,
    label_font,
    value: float,
    rect: Rect,
) -> None:
    """Draw the binary previous-collision policy input."""

    x, y, width, height = rect
    active = value >= 0.5
    color = (203, 78, 91) if active else (86, 166, 112)
    radius = max(4, min(7, height // 3))
    center = (x + radius, y + height // 2)
    pg.draw.circle(screen, color, center, radius)
    status = "1 / Yes" if active else "0 / No"
    text = label_font.render(
        f"Previous collision: {status}",
        True,
        color,
    )
    text_x = min(x + radius * 2 + 7, x + max(0, width - text.get_width()))
    screen.blit(text, (text_x, y + max(0, (height - text.get_height()) // 2)))


def _draw_input_channel_panel(
    pg,
    screen,
    panel_rect: Rect,
    state: np.ndarray,
    observation_mode: str,
    remaining_time_channel: bool,
    visit_count_channel: bool,
    action_space: str = "discrete",
    proprioception: np.ndarray | None = None,
) -> None:
    """Draw spatial input thumbnails and continuous scalar proprioception."""

    panel_x, panel_y, panel_width, panel_height = panel_rect
    pg.draw.rect(screen, (34, 40, 48), panel_rect, border_radius=8)
    title_font = pg.font.SysFont("arial", 16)
    label_font = pg.font.SysFont("arial", 13)
    title = title_font.render("Model inputs", True, (236, 240, 244))
    screen.blit(title, (panel_x + 12, panel_y + 10))

    specs = inference_channel_specs(
        state,
        observation_mode,
        remaining_time_channel,
        visit_count_channel,
        action_space,
    )
    columns = 2
    rows = (len(specs) + columns - 1) // columns
    outer_pad = 12
    gap = 8
    title_height = 34
    label_height = 20
    scalar_specs = (
        _continuous_input_specs(proprioception)
        if action_space == "continuous"
        else ()
    )
    minimum_tile_height = label_height + 1
    scalar_height = (
        min(
            108,
            max(
                68,
                panel_height
                - title_height
                - outer_pad
                - gap * (rows - 1)
                - rows * minimum_tile_height,
            ),
        )
        if scalar_specs
        else 0
    )
    tile_width = max(1, (panel_width - outer_pad * 2 - gap) // columns)
    tile_height = max(
        1,
        (
            panel_height
            - title_height
            - outer_pad
            - scalar_height
            - gap * (rows - 1)
        )
        // rows,
    )

    for index, (label, channel, color) in enumerate(specs):
        column = index % columns
        row = index // columns
        tile_x = panel_x + outer_pad + column * (tile_width + gap)
        tile_y = panel_y + title_height + row * (tile_height + gap)
        label_surface = label_font.render(label, True, (205, 213, 222))
        screen.blit(label_surface, (tile_x, tile_y))
        map_x = tile_x
        map_y = tile_y + label_height
        map_width = tile_width
        map_height = max(1, tile_height - label_height)
        channel_rows, channel_cols = channel.shape
        scale = min(map_width / channel_cols, map_height / channel_rows)
        scaled_width = max(1, int(channel_cols * scale))
        scaled_height = max(1, int(channel_rows * scale))
        map_surface = _channel_map_surface(pg, channel, color)
        map_surface = pg.transform.scale(map_surface, (scaled_width, scaled_height))
        map_x += (map_width - scaled_width) // 2
        map_y += (map_height - scaled_height) // 2
        screen.blit(map_surface, (map_x, map_y))
        pg.draw.rect(
            screen,
            (104, 116, 128),
            (map_x, map_y, scaled_width, scaled_height),
            width=1,
        )

    if scalar_specs:
        scalar_x = panel_x + outer_pad
        scalar_y = panel_y + panel_height - scalar_height
        scalar_width = max(1, panel_width - outer_pad * 2)
        scalar_font = _fitted_font(
            pg,
            "Previous collision: 1 / Yes",
            scalar_width,
            13,
        )
        collision_height = max(18, min(25, scalar_height // 4))
        gauge_height = max(
            20,
            min(31, (scalar_height - collision_height - 5) // 2),
        )
        for index, (label, value) in enumerate(scalar_specs[:2]):
            _draw_signed_input_gauge(
                pg,
                screen,
                scalar_font,
                label,
                value,
                (
                    scalar_x,
                    scalar_y + index * gauge_height,
                    scalar_width,
                    gauge_height,
                ),
            )
        _collision_label, collision_value = scalar_specs[2]
        _draw_collision_input(
            pg,
            screen,
            scalar_font,
            collision_value,
            (
                scalar_x,
                scalar_y + gauge_height * 2 + 3,
                scalar_width,
                collision_height,
            ),
        )


def _toggle_input_channel_panel(pg, event, visible: bool) -> bool:
    """Toggle input-channel visibility for the inference ``I`` shortcut."""

    if event.type == pg.KEYDOWN and event.key == pg.K_i:
        return not visible
    return visible


def _next_blocked_streak(blocked_streak: int, moved: bool) -> int:
    """Update the consecutive blocked-step count shown during inference."""

    return 0 if moved else blocked_streak + 1


def _inference_status_line(env: Maze, observation_mode: str, done: bool) -> str:
    """Build the inference HUD line with visible timeout progress."""

    outcome = ""
    if done:
        solved = (
            continuous_position_is_solved(env.continuous_position, env.goal)
            if env.action_space == "continuous"
            else env.current_position == env.goal
        )
        outcome = " | SOLVED" if solved else " | TIMEOUT"
    view_status = f"{observation_mode} observation"
    if observation_mode == "local":
        view_status += f" ({env.view_size}x{env.view_size} highlighted)"
    return f"Steps {env.steps:>3}/{env.max_episode_steps:<3} | {view_status}{outcome}"


def visualize_inference(
    agent: Agent | Planner,
    maze_grid: np.ndarray,
    fps: int = DEFAULT_INFERENCE_FPS,
    observation_mode: str | None = None,
    config: TrainConfig | None = None,
    show_input_channels: bool = DEFAULT_SHOW_INPUT_CHANNELS,
    session: InferenceSession | None = None,
) -> bool:
    """Render one maze and return whether it completed without window closure."""

    owns_session = session is None
    active_session = session or _create_inference_session(
        tuple(int(value) for value in maze_grid.shape),
        fps,
        show_input_channels,
    )
    try:
        return _visualize_inference_episode(
            agent,
            maze_grid,
            observation_mode=observation_mode,
            config=config,
            session=active_session,
        )
    finally:
        if owns_session:
            _close_inference_session(active_session)


def _visualize_inference_episode(
    agent: Agent | Planner,
    maze_grid: np.ndarray,
    *,
    observation_mode: str | None,
    config: TrainConfig | None,
    session: InferenceSession,
) -> bool:
    """Render one maze using an existing inference session."""

    agent_config = config or getattr(agent, "config", TrainConfig())
    mode = observation_mode or agent_config.observation_mode
    env = Maze(
        maze_grid.copy(),
        observation_mode=mode,
        view_size=getattr(agent, "view_size", agent_config.view_size),
        remaining_time_channel=agent_config.remaining_time_channel,
        visit_count_channel=agent_config.visit_count_channel,
        visit_count_clip=agent_config.visit_count_clip,
        wall_occlusion=agent_config.wall_occlusion,
        max_episode_steps=agent_config.max_episode_steps,
        timeout_step_factor=agent_config.timeout_step_factor,
        min_episode_steps=agent_config.min_episode_steps,
        exploration_step_factor=agent_config.exploration_step_factor,
        step_penalty=agent_config.step_penalty,
        invalid_move_penalty=agent_config.invalid_move_penalty,
        goal_reward=agent_config.goal_reward,
        timeout_penalty=agent_config.timeout_penalty,
        distance_shaping_scale=agent_config.distance_shaping_scale,
        distance_shaping_mode=agent_config.distance_shaping_mode,
        gamma=agent_config.gamma,
        action_space=agent_config.action_space,
        continuous_step_scale=agent_config.continuous_step_scale,
    )

    pygame = session.pygame
    screen = session.screen
    clock = session.clock
    rows, cols = env.grid.shape

    trail = [env.continuous_position]
    state = env.reset()
    proprioception = env.proprioception()
    action_mask = env.valid_action_mask()
    recurrent_hidden = (
        agent.initial_policy_state(1) if isinstance(agent, RecurrentPPOAgent) else None
    )
    previous_action: int | tuple[float, float] = (
        (0.0, 0.0) if agent_config.action_space == "continuous" else -1
    )
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
        proprioception,
    )
    done = False
    blocked_streak = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
            session.show_input_channels = _toggle_input_channel_panel(
                pygame,
                event,
                session.show_input_channels,
            )
            if event.type == pygame.VIDEORESIZE:
                resized = (max(320, event.w), max(240, event.h))
                screen = pygame.display.set_mode(resized, pygame.RESIZABLE)
                session.screen = screen

        window_w, window_h = screen.get_size()
        layout = inference_layout(
            (window_w, window_h),
            (rows, cols),
            show_input_channels=session.show_input_channels,
        )
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

        is_continuous_action = env.action_space == "continuous"
        trail_color = (91, 142, 222)
        if is_continuous_action and len(trail) >= 2:
            pixel_pts = [
                (maze_x + pt[1] * cell_size + cell_size / 2, maze_y + pt[0] * cell_size + cell_size / 2)
                for pt in trail[:-1]
            ]
            if len(pixel_pts) >= 2:
                pygame.draw.lines(screen, trail_color, False, pixel_pts, max(2, cell_size // 8))
            elif len(pixel_pts) == 1:
                pygame.draw.circle(screen, trail_color, pixel_pts[0], max(1, cell_size // 6))
        else:
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
                    trail_color,
                    trail_rect,
                    max(1, cell_size // 24),
                )

        if is_continuous_action:
            mouse_cell_x = maze_x + env.continuous_position[1] * cell_size
            mouse_cell_y = maze_y + env.continuous_position[0] * cell_size
        else:
            mouse_cell_x = maze_x + env.current_position[1] * cell_size
            mouse_cell_y = maze_y + env.current_position[0] * cell_size
        _draw_mouse_icon(
            pygame,
            screen,
            mouse_cell_x,
            mouse_cell_y,
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
        if layout.channel_panel_rect is not None:
            _draw_input_channel_panel(
                pygame,
                screen,
                layout.channel_panel_rect,
                state,
                mode,
                agent_config.remaining_time_channel,
                agent_config.visit_count_channel,
                agent_config.action_space,
                proprioception,
            )

        hud_x, hud_y, hud_width, hud_height = layout.hud_rect
        pygame.draw.rect(
            screen,
            (231, 235, 240),
            layout.hud_rect,
            border_radius=max(3, min(8, hud_height // 7)),
        )
        blocked = f" | Blocked streak {blocked_streak}" if blocked_streak else ""
        status_line = _inference_status_line(env, mode, done)
        if is_continuous_action:
            action_label = f"({action[0]:+.2f}, {action[1]:+.2f})"
        else:
            action_name, arrow = Maze.ACTION_NAMES[action]
            action_label = f"{action_name} {arrow}"
        if q_vals is None:
            action_line = f"Planner action: {action_label}{blocked}"
        else:
            masked_q_vals = q_vals.copy()
            masked_q_vals[~action_mask] = -np.inf
            best_action = int(np.argmax(masked_q_vals))
            best_name, best_arrow = Maze.ACTION_NAMES[best_action]
            action_line = (
                f"Last action: {action_label}{blocked} | "
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
        clock.tick(session.fps)

        if done:
            break

        trail.append(env.continuous_position)
        next_state, step_reward, done, step_info = env.step(action)
        blocked_streak = _next_blocked_streak(
            blocked_streak,
            bool(step_info["moved"]),
        )
        state = next_state
        proprioception = env.proprioception()
        action_mask = env.valid_action_mask()
        if is_continuous_action:
            executed = np.asarray(step_info["executed_displacement"], dtype=np.float32)
            previous_action = (float(executed[0]), float(executed[1]))
        else:
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
            proprioception,
        )

    label = (
        "SOLVED"
        if (
            continuous_position_is_solved(env.continuous_position, env.goal)
            if env.action_space == "continuous"
            else env.current_position == env.goal
        )
        else "TIMEOUT"
    )
    print(f"Steps: {env.steps} -- {label}")
    return True


def _inference_action(
    policy: Agent | Planner,
    state: np.ndarray,
    action_mask: np.ndarray,
    recurrent_hidden: torch.Tensor | None = None,
    previous_action: int = -1,
    previous_reward: float = 0.0,
    episode_start: bool = True,
    proprioception: np.ndarray | None = None,
) -> tuple[int, np.ndarray | None, torch.Tensor | None]:
    """Choose a legal action and optionally return learned action values."""

    if isinstance(policy, RecurrentPPOAgent):
        if recurrent_hidden is None:
            recurrent_hidden = policy.initial_policy_state(1)
        is_continuous = policy.policy_net.continuous
        if is_continuous:
            if proprioception is None or np.asarray(proprioception).shape != (3,):
                actual = None if proprioception is None else np.asarray(proprioception).shape
                raise ValueError(
                    "continuous inference requires proprioception shape (3,), "
                    f"got {actual}"
                )
            if isinstance(previous_action, int):
                prev_actions_arr = np.array([[-1.0, -1.0]], dtype=np.float32)
            else:
                prev_actions_arr = np.array(
                    [[previous_action[0], previous_action[1]]],
                    dtype=np.float32,
                )
        else:
            prev_actions_arr = np.array([previous_action], dtype=np.int64)
        actions, next_hidden = policy.get_actions_stateful(
            state[np.newaxis],
            action_mask[np.newaxis],
            prev_actions_arr,
            np.array([previous_reward], dtype=np.float32),
            np.array([episode_start], dtype=np.bool_),
            recurrent_hidden,
            (
                None
                if not is_continuous
                else np.asarray(proprioception, dtype=np.float32)[np.newaxis]
            ),
        )
        if is_continuous:
            return tuple(float(v) for v in actions[0]), None, next_hidden
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


def _positive_int(value: str) -> int:
    """Parse a strictly positive integer CLI value."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def run_inference_loop(
    agent: Agent | Planner,
    config: TrainConfig,
    maze_count: int = DEFAULT_INFERENCE_MAZES,
    show_input_channels: bool = DEFAULT_SHOW_INPUT_CHANNELS,
    fps: int = DEFAULT_INFERENCE_FPS,
) -> int:
    """Run inference on fresh mazes until a count or window close stops it.

    A ``maze_count`` of zero means that new mazes are generated indefinitely.
    The returned value is the number of mazes that were started.
    """

    if maze_count < 0:
        raise ValueError("maze_count must be non-negative")
    if fps < 1:
        raise ValueError("inference FPS must be positive")

    completed = 0
    session: InferenceSession | None = None
    try:
        while maze_count == 0 or completed < maze_count:
            completed += 1
            maze_grid = make_maze(config).grid
            if session is None:
                session = _create_inference_session(
                    tuple(int(value) for value in maze_grid.shape),
                    fps,
                    show_input_channels,
                )
            limit = "infinite" if maze_count == 0 else str(maze_count)
            print(f"Inference on fresh maze {completed}/{limit}:")
            if not visualize_inference(
                agent,
                maze_grid.copy(),
                observation_mode=config.observation_mode,
                config=config,
                session=session,
            ):
                break
    finally:
        if session is not None:
            _close_inference_session(session)
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
            "Resume the selected/latest schema-v11 checkpoint; fresh training "
            "is the default."
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
    parser.add_argument("--action-space", choices=ACTION_SPACES, default=None)
    parser.add_argument("--continuous-step-scale", type=float, default=None)
    parser.add_argument(
        "--continuous-distance-mode",
        choices=CONTINUOUS_DISTANCE_MODES,
        default=None,
    )
    parser.add_argument("--view-size", type=int, default=None)
    parser.add_argument(
        "--remaining-time-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include normalized remaining episode time in observations.",
    )
    parser.add_argument(
        "--visit-count-channel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Include normalized per-cell visit counts in observations.",
    )
    parser.add_argument(
        "--visit-count-clip",
        type=int,
        default=None,
        help="Visit count represented as 1.0 in the visit-count channel.",
    )
    parser.add_argument(
        "--visit-count-encoding",
        choices=VISIT_COUNT_ENCODINGS,
        default=None,
        help="Encode visits with a legacy clip or episode-scaled logarithm.",
    )
    parser.add_argument(
        "--wall-occlusion",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Hide local-observation cells that are behind walls.",
    )
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--timeout-step-factor", type=float, default=None)
    parser.add_argument("--min-episode-steps", type=int, default=None)
    parser.add_argument("--exploration-step-factor", type=float, default=None)
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
    parser.add_argument("--curriculum-mode", choices=CURRICULUM_MODES, default=None)
    parser.add_argument("--curriculum-probe-mazes", type=int, default=None)
    parser.add_argument("--curriculum-size-step", type=int, default=None)
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
            "Parallel environments; defaults to 768 for recurrent PPO on the "
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
    parser.add_argument("--continuous-entropy-coef", type=float, default=None)
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
    parser.add_argument("--post-curriculum-eval-every-steps", type=int, default=None)
    parser.add_argument("--dashboard-every", type=int, default=None)
    parser.add_argument("--curriculum-promotion-rate", type=float, default=None)
    parser.add_argument("--curriculum-promotion-evals", type=int, default=None)
    parser.add_argument("--curriculum-previous-fraction", type=float, default=None)
    parser.add_argument("--curriculum-uniform-fraction", type=float, default=None)
    parser.add_argument(
        "--curriculum-final-stage-fraction",
        type=float,
        default=None,
        help=(
            "Fraction of a finite recurrent-PPO budget reserved for the "
            "unrestricted final curriculum stage; zero disables forced promotion."
        ),
    )
    parser.add_argument("--curriculum-eval-episodes", type=int, default=None)
    parser.add_argument(
        "--hard-maze-fraction",
        type=float,
        default=None,
        help=(
            "Maximum fraction of recurrent tasks sampled from hard variants; "
            "ramps from zero at 60%% validation solve rate to full at 80%%."
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
        help="Enable rollback-based final-stage precision recovery.",
    )
    parser.add_argument("--precision-fraction", type=float, default=None)
    parser.add_argument("--precision-learning-rate-fraction", type=float, default=None)
    parser.add_argument("--precision-entropy-fraction", type=float, default=None)
    parser.add_argument(
        "--precision-plateau-evals",
        type=int,
        default=None,
        help="Final-stage evaluations without improvement before recovery.",
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
    parser.add_argument("--maze-generation-batch-size", type=int, default=None)
    parser.add_argument("--maze-prefetch-batches-per-worker", type=int, default=None)
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
    parser.add_argument(
        "--inference-fps",
        type=_positive_int,
        default=DEFAULT_INFERENCE_FPS,
        metavar="FPS",
        help="Maximum inference frames per second (default: 30).",
    )
    parser.add_argument(
        "--show-input-channels",
        dest="show_input_channels",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Show the observation channels supplied to the model during "
            "inference; the I key toggles the panel live."
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
    if args.max_env_steps is None and (
        args.maze_size is not None or args.observation_mode is not None
    ):
        values["max_env_steps"] = None
    if args.algorithm in {"dqn", "ppo"} and args.action_space is None:
        values["action_space"] = None
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
        "action_space",
        "continuous_step_scale",
        "continuous_distance_mode",
        "view_size",
        "remaining_time_channel",
        "visit_count_channel",
        "visit_count_clip",
        "visit_count_encoding",
        "wall_occlusion",
        "max_episode_steps",
        "timeout_step_factor",
        "min_episode_steps",
        "exploration_step_factor",
        "step_penalty",
        "invalid_move_penalty",
        "goal_reward",
        "timeout_penalty",
        "distance_shaping_scale",
        "distance_shaping_mode",
        "curriculum_enabled",
        "curriculum_mode",
        "curriculum_probe_mazes",
        "curriculum_size_step",
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
        "continuous_entropy_coef",
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
        "post_curriculum_eval_every_steps",
        "dashboard_every",
        "curriculum_promotion_rate",
        "curriculum_promotion_evals",
        "curriculum_previous_fraction",
        "curriculum_uniform_fraction",
        "curriculum_final_stage_fraction",
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
        "precision_fraction",
        "precision_learning_rate_fraction",
        "precision_entropy_fraction",
        "latest_checkpoint_every_steps",
        "performance_profile",
        "maze_workers",
        "maze_generation_batch_size",
        "maze_prefetch_batches_per_worker",
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
    should_load_checkpoint = (
        config.save_path
        and os.path.exists(config.save_path)
        and (
            args.benchmark
            or config.resume
            or not _optional_bool(args.train_flag, DEFAULT_TRAIN_FLAG)
        )
    )
    if should_load_checkpoint:
        checkpoint_schema = checkpoint_schema_version(config.save_path)
        if config.resume and checkpoint_schema < CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                "schema-v10 checkpoints remain inference-compatible but cannot be "
                "resumed after reward and observation semantics changed"
            )
        checkpoint_algo = checkpoint_algorithm(config.save_path)
        checkpoint_actions = checkpoint_action_space(config.save_path)
        checkpoint_settings = checkpoint_observation_settings(config.save_path)
        checkpoint_shaping = checkpoint_distance_shaping_mode(config.save_path)
        checkpoint_distance = checkpoint_continuous_distance_mode(config.save_path)
        if args.algorithm is not None and args.algorithm != checkpoint_algo:
            raise ValueError(
                f"checkpoint algorithm is {checkpoint_algo!r}, but --algorithm is "
                f"{args.algorithm!r}"
            )
        if args.action_space is not None and args.action_space != checkpoint_actions:
            raise ValueError(
                f"checkpoint action space is {checkpoint_actions!r}, but "
                f"--action-space is {args.action_space!r}"
            )
        values = {field_.name: getattr(config, field_.name) for field_ in fields(config)}
        values["algorithm"] = checkpoint_algo
        values["action_space"] = checkpoint_actions
        if (
            args.distance_shaping_mode is not None
            and args.distance_shaping_mode != checkpoint_shaping
        ):
            raise ValueError(
                f"checkpoint distance shaping is {checkpoint_shaping!r}, but the CLI "
                f"requested {args.distance_shaping_mode!r}"
            )
        values["distance_shaping_mode"] = checkpoint_shaping
        if (
            args.continuous_distance_mode is not None
            and args.continuous_distance_mode != checkpoint_distance
        ):
            raise ValueError(
                f"checkpoint continuous distance is {checkpoint_distance!r}, but "
                f"the CLI requested {args.continuous_distance_mode!r}"
            )
        values["continuous_distance_mode"] = checkpoint_distance
        observation_arguments = {
            "remaining_time_channel": args.remaining_time_channel,
            "visit_count_channel": args.visit_count_channel,
            "visit_count_clip": args.visit_count_clip,
            "visit_count_encoding": args.visit_count_encoding,
            "wall_occlusion": args.wall_occlusion,
            "continuous_step_scale": args.continuous_step_scale,
        }
        for name, explicit_value in observation_arguments.items():
            if name not in checkpoint_settings:
                continue
            checkpoint_value = checkpoint_settings[name]
            if explicit_value is not None and explicit_value != checkpoint_value:
                raise ValueError(
                    f"checkpoint {name} is {checkpoint_value!r}, but the CLI "
                    f"requested {explicit_value!r}"
                )
            values[name] = checkpoint_value
        config = TrainConfig(**values)
    expected_shape = config_observation_shape(config)
    agent = create_agent(config, expected_shape, device)

    if args.benchmark:
        if not config.save_path or not os.path.exists(config.save_path):
            raise FileNotFoundError("--benchmark requires an existing --save-path checkpoint")
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
        agent.load(config.save_path)

    if _optional_bool(args.infer_flag, DEFAULT_INFER_FLAG):
        inference_policy: Agent | Planner = BfsPlanner() if config.planner_fallback else agent
        run_inference_loop(
            inference_policy,
            config,
            args.inference_mazes,
            show_input_channels=_optional_bool(
                args.show_input_channels,
                DEFAULT_SHOW_INPUT_CHANNELS,
            ),
            fps=args.inference_fps,
        )


if __name__ == "__main__":
    main()
