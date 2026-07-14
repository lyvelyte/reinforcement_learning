import json
import random
from collections import deque
from dataclasses import fields, replace
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

import MouseAgent as mouse_agent_module
from gen_maze import generate_random_maze
from MouseAgent import (
    DASHBOARD_MAX_HISTORY_POINTS,
    DASHBOARD_TOOLTIPS,
    DEFAULT_INFER_FLAG,
    DEFAULT_INFERENCE_MAZES,
    DEFAULT_TRAIN_FLAG,
    EVAL_SEED_OFFSET,
    BfsPlanner,
    CurriculumController,
    CurriculumStage,
    Dashboard,
    DeterministicMazePrefetcher,
    EpisodeStats,
    EvalMetrics,
    Maze,
    MazeBatch,
    MazeTaskSampler,
    MaskedPPOAgent,
    MetricsTracker,
    MouseAgent,
    RNDModule,
    RecurrentPPOAgent,
    ReplayBuffer,
    TrainConfig,
    _chart_x_max,
    _confirm_target_candidate,
    _dashboard_tooltip_at,
    _draw_cheese_icon,
    _draw_mouse_icon,
    _draw_start_icon,
    _episode_tick_values,
    _eval_greedy,
    _format_chart_tick,
    _format_number,
    _format_pct,
    _generate_prefetched_grid,
    _optional_bool,
    _non_overlapping_episode_ticks,
    _rects_overlap,
    _recurrent_ppo_precision_schedule,
    _rnd_coefficient,
    _restore_rng_state,
    _should_run_eval,
    _apply_legacy_cli_aliases,
    _save_latest_checkpoint,
    _train_config_from_args,
    _training_speed_payload,
    _training_environment_payload,
    default_artifact_paths,
    invocation_timestamp,
    latest_model_path,
    latest_checkpoint_path,
    paired_log_path,
    resolve_cli_artifacts,
    resolve_num_envs,
    build_n_step_transition,
    automatic_curriculum_sizes,
    curriculum_distance_range,
    dashboard_layout,
    inference_layout,
    linear_epsilon,
    local_observation_bounds,
    make_maze,
    parse_args,
    pretrain_with_expert,
    run_benchmark,
    run_inference_loop,
    train,
)


def fixed_grid():
    return np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 0, 3, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )


def test_wilson_generator_is_seeded_connected_and_acyclic():
    first = generate_random_maze(11, 11, rng=random.Random(1234))
    second = generate_random_maze(11, 11, rng=random.Random(1234))

    assert np.array_equal(first, second)
    assert np.all(first[[0, -1], :] == 1)
    assert np.all(first[:, [0, -1]] == 1)
    traversable = first != 1
    open_cells = int(traversable.sum())
    edges = int(
        (traversable[:, :-1] & traversable[:, 1:]).sum()
        + (traversable[:-1] & traversable[1:]).sum()
    )
    env = Maze(first, observation_mode="full")

    assert np.count_nonzero(env.bfs_distances >= 0) == open_cells
    assert edges == open_cells - 1


def test_automatic_curriculum_size_ladder_and_budget_scaling():
    assert automatic_curriculum_sizes((11, 11)) == [(11, 11)]
    assert automatic_curriculum_sizes((21, 15)) == [
        (11, 11),
        (15, 15),
        (19, 15),
        (21, 15),
    ]
    medium = TrainConfig(maze_size=(15, 15))
    large = TrainConfig(maze_size=(21, 21))
    assert medium.max_env_steps == 136_000_000
    assert large.max_env_steps == 191_000_000
    assert medium.episodes >= medium.max_env_steps
    assert large.episodes >= large.max_env_steps


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
    ],
)
def test_restore_rng_state_normalizes_torch_state_to_cpu(device, monkeypatch):
    expected = torch.get_rng_state()
    saved_state = expected.to(device)
    restored = []
    cuda_restored = []

    monkeypatch.setattr(
        mouse_agent_module.torch,
        "set_rng_state",
        lambda state: restored.append(state),
    )
    if torch.cuda.is_available():
        cuda_expected = torch.cuda.get_rng_state()
        monkeypatch.setattr(
            mouse_agent_module.torch.cuda,
            "set_rng_state_all",
            lambda states: cuda_restored.extend(states),
        )
        training_state = {
            "torch_rng_state": saved_state,
            "cuda_rng_states": [cuda_expected.to(device)],
        }
    else:
        training_state = {"torch_rng_state": saved_state}

    _restore_rng_state(training_state)

    assert len(restored) == 1
    assert restored[0].device == torch.device("cpu")
    assert restored[0].dtype == torch.uint8
    assert torch.equal(restored[0], expected)
    if torch.cuda.is_available():
        assert len(cuda_restored) == 1
        assert cuda_restored[0].device == torch.device("cpu")
        assert cuda_restored[0].dtype == torch.uint8
        assert torch.equal(cuda_restored[0], cuda_expected)


@pytest.mark.parametrize(
    ("window_size", "maze_shape"),
    [
        ((1200, 500), (11, 21)),
        ((500, 900), (21, 11)),
        ((240, 180), (11, 11)),
        ((1600, 1200), (5, 9)),
    ],
)
def test_inference_layout_centers_square_cells_within_viewport(
    window_size,
    maze_shape,
):
    layout = inference_layout(window_size, maze_shape)
    maze_x, maze_y, maze_width, maze_height = layout.maze_rect
    hud_x, hud_y, hud_width, hud_height = layout.hud_rect
    rows, cols = maze_shape

    assert layout.cell_size >= 1
    assert layout.cell_size <= 80
    assert maze_width == cols * layout.cell_size
    assert maze_height == rows * layout.cell_size
    assert 0 <= maze_x
    assert 0 <= maze_y
    assert maze_x + maze_width <= window_size[0]
    assert maze_y + maze_height <= hud_y
    assert abs((maze_x + maze_width / 2) - window_size[0] / 2) <= 0.5
    assert hud_x >= 0
    assert hud_y + hud_height <= window_size[1]
    assert hud_width > 0


def test_local_observation_bounds_cover_center_and_clip_at_edges():
    assert local_observation_bounds((5, 5), 7, (11, 11)) == (2, 2, 9, 9)
    assert local_observation_bounds((1, 1), 7, (11, 11)) == (0, 0, 5, 5)
    assert local_observation_bounds((9, 9), 7, (11, 11)) == (6, 6, 11, 11)

    with pytest.raises(ValueError, match="positive odd"):
        local_observation_bounds((1, 1), 6, (11, 11))


@pytest.mark.parametrize("cell_size", [4, 48])
@pytest.mark.parametrize(
    "draw_icon",
    [_draw_start_icon, _draw_cheese_icon, _draw_mouse_icon],
)
def test_inference_vector_icons_render_on_headless_surface(draw_icon, cell_size):
    pygame = pytest.importorskip("pygame")
    surface = pygame.Surface((cell_size, cell_size))
    background = (11, 13, 17)
    surface.fill(background)

    draw_icon(pygame, surface, 0, 0, cell_size)

    pixels = pygame.surfarray.array3d(surface)
    assert np.any(pixels != np.asarray(background, dtype=np.uint8))


def test_full_observation_channels():
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)

    obs = env.reset()

    assert obs.shape == (4, 5, 5)
    assert obs[0].sum() == np.count_nonzero(env.grid == 1)
    assert obs[1, env.start[0], env.start[1]] == 1.0
    assert obs[1].sum() == 1.0
    assert obs[2, env.goal[0], env.goal[1]] == 1.0
    assert obs[2].sum() == 1.0
    assert np.all(obs[3] == 1.0)


def test_local_observation_keeps_agent_centered_and_goal_visible():
    env = Maze(
        fixed_grid(),
        observation_mode="local",
        view_size=5,
        max_episode_steps=10,
    )

    obs = env.reset()

    assert obs.shape == (5, 5, 5)
    assert obs[1, 2, 2] == 1.0
    assert obs[1].sum() == 1.0
    assert obs[2, 2, 4] == 1.0
    assert np.all(obs[3] == 1.0)
    assert obs[4, 2, 2] == 1.0
    assert obs[4].sum() == 1.0

    next_obs, _reward, _done, _info = env.step(0)
    assert next_obs[4, 2, 1] == 1.0
    assert next_obs[4, 2, 2] == 1.0


def test_reward_invalid_move_solve_and_timeout_flags():
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    state = env.reset()

    next_state, invalid_reward, done, info = env.step(1)
    assert next_state.shape == state.shape
    assert invalid_reward < -0.1
    assert done is False
    assert info["moved"] is False
    assert info["invalid"] is True
    assert env.current_position == env.start

    _, first_reward, done, info = env.step(0)
    assert first_reward > 0.0
    assert done is False
    assert info["moved"] is True

    _, goal_reward, done, info = env.step(0)
    assert goal_reward > 10.0
    assert done is True
    assert info["solved"] is True
    assert info["timeout"] is False

    timeout_env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=1)
    timeout_env.reset()
    _, timeout_reward, done, info = timeout_env.step(1)
    assert timeout_reward < -2.0
    assert done is True
    assert info["timeout"] is True
    assert info["solved"] is False


def test_potential_reward_discourages_two_step_oscillation():
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    env.reset()

    _, toward_reward, done, _info = env.step(0)
    _, reverse_reward, reverse_done, _info = env.step(1)

    assert toward_reward > 0.0
    assert reverse_reward < 0.0
    assert toward_reward + reverse_reward < 0.0
    assert done is False
    assert reverse_done is False

    goal_env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    goal_env.reset()
    goal_env.step(0)
    _, goal_reward, done, info = goal_env.step(0)

    assert goal_reward > 10.0
    assert done is True
    assert info["solved"] is True


def test_valid_action_mask_marks_open_neighbor_actions():
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    env.reset()

    mask = env.valid_action_mask()

    assert mask.tolist() == [True, False, False, False]


def test_dynamic_timeout_uses_optimal_steps_with_cap():
    env = Maze(
        fixed_grid(),
        observation_mode="full",
        max_episode_steps=10,
        timeout_step_factor=4.0,
        min_episode_steps=5,
    )
    capped_env = Maze(
        fixed_grid(),
        observation_mode="full",
        max_episode_steps=6,
        timeout_step_factor=4.0,
        min_episode_steps=5,
    )

    assert env.optimal_start_steps == 2
    assert env.max_episode_steps == 8
    assert capped_env.max_episode_steps == 6


@pytest.mark.parametrize("maze_size", [(11, 11), (21, 21)])
def test_local_exploration_horizon_scales_with_traversable_cells(maze_size):
    grid = generate_random_maze(*maze_size, rng=random.Random(sum(maze_size)))
    local = Maze(
        grid,
        observation_mode="local",
        max_episode_steps=None,
        timeout_step_factor=4.0,
        min_episode_steps=1,
        exploration_step_factor=2.0,
    )
    expected = max(
        1,
        4 * local.optimal_start_steps,
        2 * int(np.count_nonzero(grid != 1)),
    )
    capped = Maze(
        grid,
        observation_mode="local",
        max_episode_steps=expected - 1,
        timeout_step_factor=4.0,
        min_episode_steps=1,
        exploration_step_factor=2.0,
    )
    full = Maze(
        grid,
        observation_mode="full",
        max_episode_steps=None,
        timeout_step_factor=4.0,
        min_episode_steps=1,
    )

    assert local.max_episode_steps == expected
    assert capped.max_episode_steps == expected - 1
    assert full.max_episode_steps == max(1, 4 * full.optimal_start_steps)


def test_default_timeout_provides_local_recovery_budget(monkeypatch):
    config = TrainConfig(
        maze_size=(5, 5),
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    monkeypatch.setattr(
        mouse_agent_module,
        "generate_random_maze",
        lambda rows, cols, rng=None: fixed_grid(),
    )

    env = make_maze(config)

    assert config.min_episode_steps == 20
    assert config.timeout_step_factor == 4.0
    assert config.max_episode_steps is None
    assert config.recurrent_sequence_length == 64
    assert config.recurrent_sequence_minibatch_size == 32
    assert config.recurrent_hidden_size == 128
    assert config.hard_maze_fraction == 0.05
    assert config.precision_recovery_enabled is False
    assert config.target_solve_rate == 1.0
    assert config.target_solve_evals == 3
    assert env.optimal_start_steps == 2
    assert env.max_episode_steps == 20


def test_recurrent_precision_schedule_holds_lr_and_entropy_floors_after_budget():
    config = TrainConfig(
        max_env_steps=100,
        learning_rate=1e-3,
        ppo_entropy_coef=2e-2,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )

    assert _recurrent_ppo_precision_schedule(0, config) == (1e-3, 2e-2)
    learning_rate, entropy = _recurrent_ppo_precision_schedule(100, config)
    assert np.isclose(learning_rate, 1e-5)
    assert np.isclose(entropy, 2e-3)
    learning_rate, entropy = _recurrent_ppo_precision_schedule(200, config)
    assert np.isclose(learning_rate, 1e-5)
    assert np.isclose(entropy, 2e-3)
    learning_rate, entropy = _recurrent_ppo_precision_schedule(1_000, config)
    assert np.isclose(learning_rate, 1e-5)
    assert np.isclose(entropy, 2e-3)
    assert np.isclose(_rnd_coefficient(0, config), config.rnd_reward_coef)
    assert np.isclose(_rnd_coefficient(100, config), 0.0)


def test_default_recurrent_minibatch_preserves_2048_transition_density():
    config = TrainConfig(
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )

    assert config.recurrent_sequence_length == 64
    assert config.recurrent_sequence_minibatch_size == 32
    assert (
        config.recurrent_sequence_length
        * config.recurrent_sequence_minibatch_size
        == 2_048
    )


def test_target_only_evaluation_ignores_episode_cap_until_step_interval():
    target_only = TrainConfig(
        episodes=10,
        eval_every_steps=50,
        target_only_stop=True,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    bounded = replace(target_only, target_only_stop=False)

    assert not _should_run_eval(target_only, completed=11, total_steps=120, last_eval_step=100)
    assert _should_run_eval(target_only, completed=11, total_steps=150, last_eval_step=100)
    assert _should_run_eval(bounded, completed=10, total_steps=120, last_eval_step=100)
    assert not _should_run_eval(
        target_only,
        completed=11,
        total_steps=150,
        last_eval_step=100,
        eval_every_steps=500,
    )


def test_profile_selected_environment_parallelism_preserves_overrides():
    fast = TrainConfig(
        algorithm="recurrent_ppo",
        performance_profile="rtx3090-fast",
        num_envs=None,
    )
    portable = TrainConfig(
        algorithm="recurrent_ppo",
        performance_profile="portable",
        num_envs=None,
    )
    explicit = TrainConfig(
        algorithm="recurrent_ppo",
        performance_profile="rtx3090-fast",
        num_envs=256,
    )

    assert resolve_num_envs(fast) == 768
    assert resolve_num_envs(portable) == 256
    assert resolve_num_envs(explicit) == 256


@pytest.mark.parametrize(
    ("additional_rates", "confirmed"),
    [([1.0, 1.0], True), ([1.0, 0.99], False)],
)
def test_target_confirmation_uses_disjoint_suites_without_updates(
    monkeypatch,
    additional_rates,
    confirmed,
):
    config = TrainConfig(
        eval_seed=123,
        target_solve_rate=1.0,
        target_solve_evals=3,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    policy = type("FrozenPolicy", (), {"update_count": 7})()
    rates = iter(additional_rates)
    observed_seeds = []

    def fake_eval(agent, suite_config, **kwargs):
        assert agent is policy
        observed_seeds.append(suite_config.eval_seed)
        return EvalMetrics(solve_rate=next(rates))

    monkeypatch.setattr(mouse_agent_module, "_eval_greedy", fake_eval)

    result = _confirm_target_candidate(policy, config, EvalMetrics(solve_rate=1.0))

    assert result.confirmed is confirmed
    assert [seed for seed, _metrics in result.suites] == [
        123,
        123 + EVAL_SEED_OFFSET,
        123 + 2 * EVAL_SEED_OFFSET,
    ]
    assert observed_seeds == [
        123 + EVAL_SEED_OFFSET,
        123 + 2 * EVAL_SEED_OFFSET,
    ]
    assert policy.update_count == 7


def test_curriculum_distance_ranges_and_sampling_are_deterministic():
    config = TrainConfig(
        maze_size=(11, 11),
        curriculum_max_retries=500,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    rng = random.Random(123)

    assert curriculum_distance_range(config, 1) == (6, 10)
    assert curriculum_distance_range(config, 6_000) == (8, 16)
    assert curriculum_distance_range(config, 16_000) is None
    assert curriculum_distance_range(TrainConfig(maze_size=(5, 5)), 1) == (3, 5)

    easy = [make_maze(config, rng=rng, episode=1).optimal_start_steps for _ in range(8)]
    medium = [
        make_maze(config, rng=rng, episode=6_000).optimal_start_steps
        for _ in range(8)
    ]

    assert all(6 <= steps <= 10 for steps in easy)
    assert all(8 <= steps <= 16 for steps in medium)


def test_hard_maze_replay_uses_training_failure_and_transformed_variants():
    config = TrainConfig(
        maze_size=(5, 5),
        hard_maze_fraction=1.0,
        hard_maze_pool_size=8,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    sampler = MazeTaskSampler(config, random.Random(123))

    added = sampler.add_failed_grids([fixed_grid()])
    sampler.record_validation_solve_rate(1.0)
    sampled = sampler.sample()

    assert added >= 3
    assert len(sampler.hard_grids) == added
    assert any(np.array_equal(grid, fixed_grid()) for grid in sampler.hard_grids)
    assert any(np.array_equal(sampled.grid, grid) for grid in sampler.hard_grids)
    assert sampled.optimal_start_steps <= sampled.max_episode_steps
    assert sampler.sampling_mix() == {
        "hard": 1.0,
        "current": 0.0,
        "previous": 0.0,
        "unrestricted": 0.0,
    }


def test_hard_maze_pool_restores_exact_variants():
    config = TrainConfig(
        maze_size=(5, 5),
        curriculum_mode="manual",
        hard_maze_fraction=1.0,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    original = MazeTaskSampler(config, random.Random(1))
    original.add_failed_grids([fixed_grid()])
    restored = MazeTaskSampler(config, random.Random(2))

    restored.restore_hard_grids(original.hard_grids)

    assert len(restored.hard_grids) == len(original.hard_grids)
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(restored.hard_grids, original.hard_grids)
    )


def test_hard_maze_reservoirs_are_separated_by_grid_shape():
    config = TrainConfig(
        maze_size=(5, 5),
        hard_maze_fraction=1.0,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    sampler = MazeTaskSampler(config, random.Random(11))
    rectangular = generate_random_maze(5, 7, rng=random.Random(12))

    sampler.add_failed_grids([fixed_grid(), rectangular])

    assert set(sampler.hard_grids_by_shape) == {(5, 5), (5, 7)}
    assert all(grid.shape == (5, 5) for grid in sampler.hard_grids_by_shape[(5, 5)])
    assert all(grid.shape == (5, 7) for grid in sampler.hard_grids_by_shape[(5, 7)])
    assert sampler.hard_grids is sampler.hard_grids_by_shape[(5, 5)]


def test_hard_maze_fraction_ramps_with_validation_and_reservoir_is_deterministic():
    config = TrainConfig(
        maze_size=(5, 5),
        hard_maze_fraction=0.15,
        hard_maze_pool_size=3,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    first = MazeTaskSampler(config, random.Random(7))
    second = MazeTaskSampler(config, random.Random(7))
    for sampler in (first, second):
        sampler.add_failed_grids([fixed_grid()])

    first.record_validation_solve_rate(0.90)
    assert first.effective_hard_maze_fraction() == 0.0
    first.record_validation_solve_rate(0.945)
    assert np.isclose(first.effective_hard_maze_fraction(), 0.075)
    first.record_validation_solve_rate(0.99)
    assert np.isclose(first.effective_hard_maze_fraction(), 0.15)

    assert first.hard_candidates_seen == second.hard_candidates_seen
    assert all(
        np.array_equal(actual, expected)
        for actual, expected in zip(first.hard_grids, second.hard_grids)
    )
    seen_before = first.hard_candidates_seen
    first.add_failed_grids([fixed_grid()])
    assert first.hard_candidates_seen == seen_before


def test_hard_maze_reservoir_state_round_trips_and_legacy_pool_loads():
    config = TrainConfig(
        maze_size=(5, 5),
        hard_maze_pool_size=3,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    original = MazeTaskSampler(config, random.Random(4))
    original.add_failed_grids([fixed_grid()])
    original.record_validation_solve_rate(0.97)
    restored = MazeTaskSampler(config, random.Random(9))

    restored.restore_hard_grids(
        original.hard_grids,
        original._seen_hard_grid_keys,
        original.hard_candidates_seen,
        original.validation_solve_rate,
    )
    legacy = MazeTaskSampler(config, random.Random(10))
    legacy.restore_hard_grids(original.hard_grids)

    assert restored.hard_candidates_seen == original.hard_candidates_seen
    assert restored.validation_solve_rate == 0.97
    assert restored._seen_hard_grid_keys == original._seen_hard_grid_keys
    assert len(legacy.hard_grids) == len(original.hard_grids)
    assert legacy.hard_candidates_seen == len(original.hard_grids)


def test_make_maze_retries_candidates_that_exceed_episode_budget(monkeypatch):
    config = TrainConfig(
        maze_size=(5, 5),
        max_episode_steps=1,
        timeout_step_factor=None,
        curriculum_enabled=False,
        curriculum_max_retries=3,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    one_step_grid = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 3, 0, 1],
            [1, 1, 1, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    generated = [fixed_grid(), one_step_grid]
    calls = 0

    def fake_generate(rows, cols, rng=None):
        nonlocal calls
        assert (rows, cols) == config.maze_size
        calls += 1
        return generated.pop(0)

    monkeypatch.setattr(mouse_agent_module, "generate_random_maze", fake_generate)

    env = make_maze(config)

    assert calls == 2
    assert env.optimal_start_steps == 1
    assert env.optimal_start_steps <= env.max_episode_steps


def test_make_maze_raises_instead_of_returning_unsolvable_budget_candidate(monkeypatch):
    config = TrainConfig(
        maze_size=(5, 5),
        max_episode_steps=1,
        timeout_step_factor=None,
        curriculum_enabled=False,
        curriculum_max_retries=2,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    monkeypatch.setattr(
        mouse_agent_module,
        "generate_random_maze",
        lambda rows, cols, rng=None: fixed_grid(),
    )

    with pytest.raises(RuntimeError, match="solvable within the effective episode limit"):
        make_maze(config)


def test_curriculum_promotes_only_after_repeated_validation_success():
    config = TrainConfig(
        maze_size=(5, 5),
        curriculum_mode="manual",
        curriculum_promotion_rate=0.75,
        curriculum_promotion_evals=2,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    controller = CurriculumController(config)

    assert controller.target_range() == (3, 5)
    assert controller.record_validation(EvalMetrics(solve_rate=0.80)) is False
    assert controller.level == 0
    assert controller.record_validation(EvalMetrics(solve_rate=0.80)) is True
    assert controller.level == 1
    assert controller.target_range() == (4, 7)
    assert controller.previous_range() == (3, 5)


def test_automatic_curriculum_stages_are_deterministic_and_resumable():
    config = TrainConfig(
        maze_size=(7, 7),
        curriculum_probe_mazes=24,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
    )
    first = CurriculumController(config)
    second = CurriculumController(config)

    assert first.stages == second.stages
    assert [stage.maze_size for stage in first.stages] == [(7, 7)] * 3
    assert first.stages[0].complexity_high <= first.stages[1].complexity_high
    assert first.stages[2].complexity_high is None


def test_maze_batch_matches_sequential_maze_transitions():
    sequential = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    batched_source = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    sequential_state = sequential.reset()
    batched_source.reset()
    batch = MazeBatch([batched_source])

    assert np.array_equal(batch.observations()[0], sequential_state)
    for action in (0, 0):
        sequential_state, sequential_reward, sequential_done, sequential_info = sequential.step(
            action
        )
        batch_step = batch.step(np.array([action], dtype=np.int64))

        assert np.array_equal(batch_step.states[0], sequential_state)
        assert np.isclose(batch_step.rewards[0], sequential_reward)
        assert bool(batch_step.dones[0]) is sequential_done
        assert bool(batch_step.invalid[0]) is bool(sequential_info["invalid"])
        assert np.array_equal(batch_step.action_masks[0], sequential.valid_action_mask())


def test_maze_batch_subset_step_leaves_inactive_slots_unchanged():
    environments = [
        Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
        for _ in range(3)
    ]
    batch = MazeBatch(environments)
    inactive_positions = batch.positions[[0, 2]].copy()
    inactive_steps = batch.steps[[0, 2]].copy()

    result = batch.step(
        np.array([0], dtype=np.int64),
        indices=np.array([1], dtype=np.int64),
    )

    assert result.states.shape[0] == 1
    assert batch.steps[1] == 1
    assert np.array_equal(batch.positions[[0, 2]], inactive_positions)
    assert np.array_equal(batch.steps[[0, 2]], inactive_steps)


def test_bfs_planner_and_expert_labels_select_shortest_legal_action():
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    state = env.reset()
    mask = env.valid_action_mask()
    targets = env.expert_action_distribution()
    planner = BfsPlanner()

    assert targets.tolist() == [1.0, 0.0, 0.0, 0.0]
    assert planner.get_action(state, action_mask=mask) == 0
    metrics = _eval_greedy(
        planner,
        TrainConfig(
            maze_size=(5, 5),
            eval_episodes=3,
            dashboard_flag=False,
            save_path=None,
            training_log_path=None,
            device="cpu",
        ),
        maze_factory=lambda: Maze(fixed_grid(), observation_mode="full", max_episode_steps=10),
    )
    assert metrics.solve_rate == 1.0


def test_greedy_eval_retains_failed_grids_for_hard_replay():
    class StuckPolicy:
        def get_actions(self, states, epsilon=0.0, action_masks=None):
            del epsilon, action_masks
            return np.ones(len(states), dtype=np.int64)

    metrics = _eval_greedy(
        StuckPolicy(),
        TrainConfig(
            maze_size=(5, 5),
            eval_episodes=1,
            max_episode_steps=1,
            timeout_step_factor=None,
            min_episode_steps=1,
            dashboard_flag=False,
            save_path=None,
            training_log_path=None,
            device="cpu",
        ),
        maze_factory=lambda: Maze(
            fixed_grid(),
            observation_mode="full",
            max_episode_steps=1,
        ),
    )

    assert metrics.solve_rate == 0.0
    assert len(metrics.failed_grids) == 1
    assert np.array_equal(metrics.failed_grids[0], fixed_grid())


def test_prioritized_replay_samples_high_priority_transition_more_often():
    buffer = ReplayBuffer((3, 5, 5), capacity=8, seed=123)
    state = np.zeros((3, 5, 5), dtype=np.float32)
    mask = np.ones(4, dtype=np.bool_)
    for action in range(8):
        buffer.push(state, action % 4, 0.0, state, False, mask)
    buffer.update_priorities(np.arange(8), np.array([1.0] * 7 + [100.0]))

    sampled = [
        int(buffer.sample(1, prioritized=True, alpha=1.0, beta=0.4)[6][0])
        for _ in range(200)
    ]

    assert sampled.count(7) > 150


def test_transition_indexed_epsilon_schedule_and_benchmark_suites():
    config = TrainConfig(
        maze_size=(5, 5),
        epsilon_decay_steps=10,
        epsilon_final_steps=20,
        epsilon_start=1.0,
        epsilon_mid=0.2,
        epsilon_end=0.1,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )

    assert linear_epsilon(0, config) == 1.0
    assert np.isclose(linear_epsilon(10, config), 0.2)
    assert np.isclose(linear_epsilon(20, config), 0.1)
    result = run_benchmark(BfsPlanner(), config, episodes=3)

    assert result.validation.solve_rate == 1.0
    assert result.final_test.solve_rate == 1.0
    assert result.stress_test.solve_rate == 1.0
    assert result.validation.difficulty_solve_rates
    assert 0.0 < result.validation.solve_rate_lower_bound < 1.0


def test_expert_pretraining_updates_a_full_map_dqn():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        observation_mode="full",
        network_type="flat",
        batch_size=4,
        expert_pretrain_mazes=1,
        expert_pretrain_epochs=1,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))

    loss = pretrain_with_expert(agent, config)

    assert loss is not None
    assert loss >= 0.0
    assert agent.update_count > 0


def test_agent_masks_invalid_greedy_and_random_actions():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        observation_mode="full",
        network_type="flat",
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    state = env.reset()
    mask = env.valid_action_mask()
    agent = MouseAgent(config=config, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in agent.online_net.parameters():
            parameter.zero_()
        agent.online_net.advantage.bias.copy_(torch.tensor([1.0, 10.0, 0.0, 0.0]))

    greedy_action = agent.get_action(state, action_mask=mask)
    random_actions = agent.get_actions(
        np.stack([state] * 20),
        epsilon=1.0,
        action_masks=np.stack([mask] * 20),
    )

    assert greedy_action == 0
    assert set(random_actions.tolist()) == {0}


def test_spatial_q_network_masks_actions_and_saves_metadata(tmp_path):
    save_path = tmp_path / "spatial_checkpoint.pth"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        observation_mode="full",
        save_path=str(save_path),
        dashboard_flag=False,
        training_log_path=None,
        device="cpu",
    )
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    state = env.reset()
    mask = env.valid_action_mask()
    agent = MouseAgent(config=config, device=torch.device("cpu"))

    with torch.no_grad():
        for parameter in agent.online_net.parameters():
            parameter.zero_()
        agent.online_net.advantage_head.bias.copy_(
            torch.tensor([1.0, 10.0, 0.0, 0.0])
        )

    assert agent.q_values(state).shape == (4,)
    assert agent.get_action(state, action_mask=mask) == 0

    agent.save(str(save_path))
    payload = torch.load(save_path, map_location=torch.device("cpu"))

    assert payload["algorithm"] == "dqn"
    assert payload["network_type"] == "spatial"


def test_n_step_transition_uses_discounted_rewards_and_done_flush():
    state0 = np.zeros((3, 5, 5), dtype=np.float32)
    state1 = np.ones((3, 5, 5), dtype=np.float32)
    state2 = np.full((3, 5, 5), 2.0, dtype=np.float32)
    state3 = np.full((3, 5, 5), 3.0, dtype=np.float32)
    mask = np.ones(4, dtype=np.bool_)
    transitions = deque(
        [
            (state0, 2, 1.0, state1, False, mask),
            (state1, 1, 2.0, state2, False, mask),
            (state2, 0, 3.0, state3, False, mask),
        ]
    )

    result = build_n_step_transition(transitions, gamma=0.5, n_steps=3)

    assert result[0] is state0
    assert result[1] == 2
    assert result[2] == 2.75
    assert result[3] is state3
    assert result[4] is False

    done_transitions = deque(
        [
            (state0, 2, 1.0, state1, False, mask),
            (state1, 1, 2.0, state2, True, mask),
            (state2, 0, 99.0, state3, False, mask),
        ]
    )
    done_result = build_n_step_transition(done_transitions, gamma=0.5, n_steps=3)

    assert done_result[2] == 2.0
    assert done_result[3] is state2
    assert done_result[4] is True


class BfsAgent:
    def get_action(self, state, epsilon=0.0):
        walls = state[0] > 0.5
        agent_pos = tuple(np.argwhere(state[1] > 0.5)[0])
        goal_pos = tuple(np.argwhere(state[2] > 0.5)[0])
        distances = np.full(walls.shape, -1, dtype=np.int32)
        distances[goal_pos] = 0
        queue = deque([goal_pos])
        while queue:
            r, c = queue.popleft()
            for action, (dr, dc) in enumerate(Maze.ACTIONS):
                nr, nc = r + dr, c + dc
                if (
                    0 <= nr < walls.shape[0]
                    and 0 <= nc < walls.shape[1]
                    and not walls[nr, nc]
                    and distances[nr, nc] < 0
                ):
                    distances[nr, nc] = distances[r, c] + 1
                    queue.append((nr, nc))

        best_action = 0
        best_distance = distances[agent_pos]
        for action, (dr, dc) in enumerate(Maze.ACTIONS):
            nr, nc = agent_pos[0] + dr, agent_pos[1] + dc
            if (
                0 <= nr < walls.shape[0]
                and 0 <= nc < walls.shape[1]
                and distances[nr, nc] >= 0
                and distances[nr, nc] < best_distance
            ):
                best_action = action
                best_distance = distances[nr, nc]
        return best_action


class LoopAgent:
    def get_actions(self, states, epsilon=0.0, action_masks=None):
        actions = []
        for state in states:
            agent_pos = tuple(np.argwhere(state[1] > 0.5)[0])
            actions.append(1 if agent_pos == (1, 2) else 0)
        return np.array(actions, dtype=np.int64)


def test_greedy_eval_is_separate_from_training_solve_rate():
    tracker = MetricsTracker()
    tracker.record_episode(
        EpisodeStats(
            total_reward=-3.0,
            steps=5,
            solved=False,
            timeout=True,
            invalid_moves=1,
            optimal_steps=2,
        )
    )
    tracker.record_eval(EvalMetrics(solve_rate=1.0, avg_steps=2.0, optimality_ratio=1.0))

    assert tracker.train_solve_rate == 0.0
    assert tracker.latest_eval.solve_rate == 1.0

    config = TrainConfig(
        maze_size=(5, 5),
        episodes=1,
        eval_episodes=3,
        max_episode_steps=10,
        dashboard_flag=False,
        save_path=None,
        device="cpu",
    )
    metrics = _eval_greedy(
        BfsAgent(),
        config,
        maze_factory=lambda: Maze(
            fixed_grid(),
            observation_mode="full",
            max_episode_steps=10,
        ),
    )
    assert metrics.solve_rate == 1.0
    assert metrics.avg_steps == 2.0
    assert metrics.optimality_ratio == 1.0


def test_greedy_eval_reports_loop_diagnostics():
    config = TrainConfig(
        maze_size=(5, 5),
        episodes=1,
        eval_episodes=3,
        max_episode_steps=6,
        timeout_step_factor=None,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )

    metrics = _eval_greedy(
        LoopAgent(),
        config,
        maze_factory=lambda: Maze(
            fixed_grid(),
            observation_mode="full",
            max_episode_steps=6,
            timeout_step_factor=None,
        ),
    )

    assert metrics.solve_rate == 0.0
    assert metrics.timeout_rate == 1.0
    assert metrics.loop_rate == 1.0
    assert metrics.repeated_state_action_rate == 1.0
    assert metrics.failed_final_distance > 0.0


def test_greedy_eval_uses_fixed_seed_without_consuming_global_random(monkeypatch):
    rng_draws = []

    def fake_make_maze(config, rng=None):
        rng_draws.append(rng.random())
        return Maze(
            fixed_grid(),
            observation_mode="full",
            max_episode_steps=10,
        )

    monkeypatch.setattr(mouse_agent_module, "make_maze", fake_make_maze)
    config = TrainConfig(
        maze_size=(5, 5),
        episodes=1,
        eval_episodes=3,
        eval_seed=77,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    random.seed(999)
    expected_next_random = random.Random(999).random()

    first = _eval_greedy(BfsAgent(), config)
    first_draws = rng_draws.copy()
    rng_draws.clear()
    second = _eval_greedy(BfsAgent(), config)

    assert first.solve_rate == 1.0
    assert second.solve_rate == 1.0
    assert rng_draws == first_draws
    assert random.random() == expected_next_random


def test_dashboard_layout_rectangles_do_not_overlap():
    rects = dashboard_layout(1100, 720)
    values = list(rects.values())

    for x, y, width, height in values:
        assert width > 0
        assert height > 0
        assert x >= 0
        assert y >= 0
        assert x + width <= 1100
        assert y + height <= 720

    for index, rect in enumerate(values):
        for other in values[index + 1 :]:
            assert not _rects_overlap(rect, other)


def test_chart_tick_formatter_keeps_small_loss_values_visible():
    assert _format_chart_tick(0.0) == "0"
    assert _format_chart_tick(0.0043) == "0.0043"
    assert _format_chart_tick(0.043) == "0.043"
    assert _format_chart_tick(1.2) == "1.2"
    assert _format_pct(0.9985) == "99.85%"
    assert _format_number(17.123) == "17.12"


def test_chart_histories_keep_full_episode_range():
    tracker = MetricsTracker(window=3)

    for episode in range(1, 8):
        tracker.record_episode(
            EpisodeStats(
                total_reward=float(episode),
                steps=episode,
                solved=episode % 2 == 0,
                timeout=False,
                invalid_moves=0,
                optimal_steps=1,
            ),
            episode=episode,
        )

    assert len(tracker.rewards) == 3
    assert len(tracker.reward_history) == 7
    assert tracker.reward_history[0][0] == 1
    assert tracker.reward_history[-1][0] == 7


def test_chart_histories_are_bounded_and_preserve_run_endpoints():
    tracker = MetricsTracker(window=3)
    final_episode = DASHBOARD_MAX_HISTORY_POINTS * 3

    for episode in range(1, final_episode + 1):
        tracker.record_episode(
            EpisodeStats(
                total_reward=float(episode),
                steps=episode,
                solved=True,
                timeout=False,
                invalid_moves=0,
                optimal_steps=1,
            ),
            episode=episode,
        )

    assert len(tracker.reward_history) <= DASHBOARD_MAX_HISTORY_POINTS
    assert len(tracker.train_solve_history) <= DASHBOARD_MAX_HISTORY_POINTS
    assert tracker.reward_history[0][0] == 1
    assert tracker.reward_history[-1][0] == final_episode


def test_episode_tick_values_scale_without_clutter():
    small_ticks = _episode_tick_values(3, 320)
    large_ticks = _episode_tick_values(10_000, 320)

    assert small_ticks == [0, 1, 2, 3]
    assert large_ticks[0] == 0
    assert large_ticks[-1] == 10_000
    assert len(large_ticks) <= 6
    assert _chart_x_max(5, [(8, 0.5)]) == 8


def test_episode_tick_labels_remove_overlapping_interior_endpoint_neighbor():
    ticks = [0, 10_000_000, 10_100_000]
    visible = _non_overlapping_episode_ticks(
        ticks,
        max_episode=10_100_000,
        plot_width=320,
        label_widths=[7, 70, 70],
    )

    assert visible == [0, 10_100_000]


def test_dashboard_tooltips_cover_every_panel():
    rects = dashboard_layout(1100, 720)

    assert set(rects) == set(DASHBOARD_TOOLTIPS)
    for tooltip in DASHBOARD_TOOLTIPS.values():
        assert tooltip
        assert "better" in tooltip


def test_dashboard_tooltip_lookup_uses_hovered_panel():
    rects = dashboard_layout(1100, 720)
    header_x, header_y, _header_w, _header_h = rects["header"]
    loss_x, loss_y, _loss_w, _loss_h = rects["metric_7"]

    assert _dashboard_tooltip_at(rects, (header_x + 1, header_y + 1)) == (
        DASHBOARD_TOOLTIPS["header"]
    )
    assert _dashboard_tooltip_at(rects, (loss_x + 1, loss_y + 1)) == (
        DASHBOARD_TOOLTIPS["metric_7"]
    )
    assert _dashboard_tooltip_at(rects, (0, 0)) is None


def test_dashboard_poll_repaints_cached_state_on_mouse_motion():
    class FakeEvent:
        def __init__(self, event_type):
            self.type = event_type

    class FakeEventQueue:
        def __init__(self, events):
            self.events = events

        def get(self):
            events = self.events
            self.events = []
            return events

    class FakePygame:
        QUIT = 1
        KEYDOWN = 2
        MOUSEMOTION = 3
        K_ESCAPE = 27

        def __init__(self):
            self.event = FakeEventQueue([FakeEvent(self.MOUSEMOTION)])

    dashboard = Dashboard.__new__(Dashboard)
    dashboard.running = True
    dashboard.disabled = False
    dashboard.screen = object()
    dashboard.pygame = FakePygame()
    state = object()
    tracker = object()
    dashboard._last_state = state
    dashboard._last_tracker = tracker
    render_calls = []

    def fake_render(rendered_state, rendered_tracker):
        render_calls.append((rendered_state, rendered_tracker))

    dashboard._render = fake_render

    dashboard.poll()

    assert render_calls == [(state, tracker)]


def test_cli_omitted_args_use_top_level_train_config_defaults():
    args = parse_args([])

    assert _train_config_from_args(args) == TrainConfig()
    assert _optional_bool(args.train_flag, DEFAULT_TRAIN_FLAG) is DEFAULT_TRAIN_FLAG
    assert _optional_bool(args.infer_flag, DEFAULT_INFER_FLAG) is DEFAULT_INFER_FLAG
    assert args.inference_mazes == DEFAULT_INFERENCE_MAZES


def test_cli_help_renders_all_percentage_descriptions(capsys):
    with pytest.raises(SystemExit) as exit_info:
        parse_args(["--help"])

    assert exit_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "90%" in help_text


def test_target_only_stop_is_finite_by_default_and_can_be_enabled():
    assert TrainConfig(algorithm="recurrent_ppo").target_only_stop is False
    assert TrainConfig(algorithm="dqn").target_only_stop is False
    assert TrainConfig(algorithm="ppo").target_only_stop is False

    args = parse_args(["--algorithm", "recurrent_ppo", "--no-target-only-stop"])

    assert _train_config_from_args(args).target_only_stop is False

    with pytest.raises(ValueError, match="recurrent_ppo"):
        TrainConfig(algorithm="dqn", target_only_stop=True)


def test_default_artifacts_share_timestamp_and_use_results_folders():
    timestamp = invocation_timestamp(
        datetime(2026, 7, 10, 12, 34, 56, 123456, tzinfo=timezone.utc)
    )
    model_path, log_path = default_artifact_paths(timestamp)

    assert timestamp == "20260710T123456123456Z"
    assert model_path.endswith(f"results/models/{timestamp}_mousemaze.pth")
    assert log_path.endswith(f"results/logs/{timestamp}_mousemaze.jsonl")


def test_cli_artifact_resolution_honors_explicit_paths():
    args = parse_args(
        [
            "--train",
            "--save-path",
            "chosen/model.pth",
            "--training-log-path",
            "chosen/log.jsonl",
        ]
    )
    config = resolve_cli_artifacts(
        _train_config_from_args(args), args, "20260710T123456123456Z"
    )

    assert config.save_path == "chosen/model.pth"
    assert config.training_log_path == "chosen/log.jsonl"


def test_latest_model_path_is_deterministic_and_ignores_unrelated_files(tmp_path):
    older = tmp_path / "20260710T123456000001Z_mousemaze.pth"
    newer = tmp_path / "20260710T123456000002Z_mousemaze.pth"
    latest_sidecar = tmp_path / "20260710T123456000003Z_mousemaze.latest.pth"
    unrelated = (
        tmp_path / "latest_mousemaze.pth",
        latest_sidecar,
        tmp_path / "notes.txt",
    )
    for path in (newer, older, *unrelated):
        path.write_text("test")

    assert latest_model_path(str(tmp_path)) == str(newer)
    assert latest_checkpoint_path(str(newer)) == str(
        tmp_path / "20260710T123456000002Z_mousemaze.latest.pth"
    )


def test_latest_model_path_reports_actionable_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="pass --save-path explicitly"):
        latest_model_path(str(tmp_path))


def test_paired_log_path_uses_model_timestamp(tmp_path):
    model_path = tmp_path / "20260710T123456000002Z_mousemaze.pth"

    assert paired_log_path(str(model_path), str(tmp_path / "logs")) == str(
        tmp_path / "logs" / "20260710T123456000002Z_mousemaze.jsonl"
    )


def test_resume_artifact_resolution_reuses_latest_model_and_log(tmp_path, monkeypatch):
    model_path = tmp_path / "models" / "20260710T123456000002Z_mousemaze.pth"
    log_directory = tmp_path / "logs"
    log_path = log_directory / "20260710T123456000002Z_mousemaze.jsonl"
    model_path.parent.mkdir()
    log_directory.mkdir()
    model_path.write_bytes(b"checkpoint")
    log_path.write_text('{"event": "train_end"}\n')
    monkeypatch.setattr(mouse_agent_module, "LOG_RESULTS_DIR", str(log_directory))
    monkeypatch.setattr(mouse_agent_module, "latest_model_path", lambda: str(model_path))
    args = parse_args(["--train", "--resume"])

    config = resolve_cli_artifacts(
        _train_config_from_args(args), args, "20260710T123456123456Z"
    )

    assert config.save_path == str(model_path)
    assert config.training_log_path == str(log_path)


def test_resume_artifact_resolution_falls_back_when_no_model_exists(monkeypatch):
    def no_model():
        raise FileNotFoundError("no model")

    monkeypatch.setattr(mouse_agent_module, "latest_model_path", no_model)
    args = parse_args(["--train", "--resume"])

    config = resolve_cli_artifacts(
        _train_config_from_args(args), args, "20260710T123456123456Z"
    )

    assert config.save_path.endswith("20260710T123456123456Z_mousemaze.pth")
    assert config.training_log_path.endswith("20260710T123456123456Z_mousemaze.jsonl")


def test_resume_artifact_resolution_requires_matching_log(tmp_path, monkeypatch):
    model_path = tmp_path / "20260710T123456000002Z_mousemaze.pth"
    model_path.write_bytes(b"checkpoint")
    monkeypatch.setattr(mouse_agent_module, "LOG_RESULTS_DIR", str(tmp_path / "logs"))
    monkeypatch.setattr(mouse_agent_module, "latest_model_path", lambda: str(model_path))
    args = parse_args(["--train", "--resume"])

    with pytest.raises(FileNotFoundError, match="matching training log"):
        resolve_cli_artifacts(
            _train_config_from_args(args), args, "20260710T123456123456Z"
        )


def test_environment_provenance_tolerates_unavailable_git(monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(mouse_agent_module.subprocess, "run", unavailable)
    payload = _training_environment_payload(torch.device("cpu"))

    assert payload["git"] == {"commit": None, "branch": None, "dirty": None}


def test_cli_inference_maze_count_accepts_finite_and_infinite_values():
    assert parse_args(["--inference-mazes", "4"]).inference_mazes == 4
    assert parse_args(["--inference-mazes", "infinite"]).inference_mazes == 0
    assert parse_args(["--inference-mazes", "0"]).inference_mazes == 0

    with pytest.raises(SystemExit):
        parse_args(["--inference-mazes", "-1"])


def test_inference_loop_generates_requested_number_of_fresh_mazes(monkeypatch):
    config = TrainConfig(maze_size=(5, 5), save_path=None, training_log_path=None)
    generated_sizes = []
    rendered_mazes = []

    def fake_generate(rows, cols, rng=None):
        generated_sizes.append((rows, cols))
        return fixed_grid()

    def fake_visualize(agent, maze_grid, *, observation_mode, config):
        rendered_mazes.append(maze_grid)
        return True

    monkeypatch.setattr(mouse_agent_module, "generate_random_maze", fake_generate)
    monkeypatch.setattr(mouse_agent_module, "visualize_inference", fake_visualize)

    assert run_inference_loop(object(), config, maze_count=3) == 3
    assert generated_sizes == [(5, 5), (5, 5), (5, 5)]
    assert len(rendered_mazes) == 3


def test_inference_loop_infinite_mode_stops_when_window_closes(monkeypatch):
    config = TrainConfig(maze_size=(5, 5), save_path=None, training_log_path=None)
    render_count = 0

    monkeypatch.setattr(
        mouse_agent_module,
        "generate_random_maze",
        lambda rows, cols, rng=None: fixed_grid(),
    )

    def fake_visualize(*args, **kwargs):
        nonlocal render_count
        render_count += 1
        return render_count < 2

    monkeypatch.setattr(mouse_agent_module, "visualize_inference", fake_visualize)

    assert run_inference_loop(object(), config, maze_count=0) == 2


def test_cli_args_override_top_level_train_config_defaults():
    args = parse_args(
        [
            "--episodes",
            "9",
            "--max-env-steps",
            "1234",
            "--maze-size",
            "7",
            "9",
            "--no-dashboard",
            "--eval-every",
            "3",
            "--eval-seed",
            "99",
            "--algorithm",
            "ppo",
            "--network-type",
            "spatial",
            "--resume",
            "--timeout-step-factor",
            "3.5",
            "--min-episode-steps",
            "7",
            "--max-episode-steps",
            "42",
            "--exploration-step-factor",
            "2.5",
            "--distance-shaping-mode",
            "none",
            "--no-curriculum",
            "--curriculum-mode",
            "manual",
            "--curriculum-probe-mazes",
            "16",
            "--curriculum-easy-range",
            "4",
            "8",
            "--n-step-returns",
            "4",
            "--ppo-rollout-steps",
            "8",
            "--recurrent-hidden-size",
            "64",
            "--post-curriculum-eval-every-steps",
            "777",
            "--precision-fraction",
            "0.25",
            "--precision-learning-rate-fraction",
            "0.02",
            "--precision-entropy-fraction",
            "0.2",
            "--target-solve-rate",
            "0.8",
            "--performance-profile",
            "portable",
            "--maze-workers",
            "0",
            "--maze-generation-batch-size",
            "8",
            "--maze-prefetch-batches-per-worker",
            "2",
            "--learning-rate",
            "0.001",
            "--device",
            "cpu",
            "--no-infer",
        ]
    )
    config = _train_config_from_args(args)

    assert config.episodes == 9
    assert config.max_env_steps == 1234
    assert config.maze_size == (7, 9)
    assert config.dashboard_flag is False
    assert config.dashboard_every == 3
    assert config.eval_seed == 99
    assert config.algorithm == "ppo"
    assert config.network_type == "spatial"
    assert config.resume is True
    assert config.timeout_step_factor == 3.5
    assert config.min_episode_steps == 7
    assert config.max_episode_steps == 42
    assert config.exploration_step_factor == 2.5
    assert config.distance_shaping_mode == "none"
    assert config.curriculum_enabled is False
    assert config.curriculum_mode == "manual"
    assert config.curriculum_probe_mazes == 16
    assert config.curriculum_easy_range == (4, 8)
    assert config.n_step_returns == 4
    assert config.ppo_rollout_steps == 8
    assert config.recurrent_hidden_size == 64
    assert config.post_curriculum_eval_every_steps == 777
    assert config.precision_fraction == 0.25
    assert config.precision_learning_rate_fraction == 0.02
    assert config.precision_entropy_fraction == 0.2
    assert config.target_solve_rate == 0.8
    assert config.performance_profile == "portable"
    assert config.maze_workers == 0
    assert config.maze_generation_batch_size == 8
    assert config.maze_prefetch_batches_per_worker == 2
    assert config.learning_rate == 0.001
    assert config.device == "cpu"
    assert _optional_bool(args.infer_flag, DEFAULT_INFER_FLAG) is False


def test_cli_dashboard_every_can_override_eval_every():
    args = parse_args(["--eval-every", "3", "--dashboard-every", "5"])

    config = _train_config_from_args(args)

    assert config.dashboard_every == 5


def test_legacy_train_updates_alias_uses_resolved_environment_count():
    args = parse_args(
        [
            "--algorithm",
            "dqn",
            "--num-envs",
            "8",
            "--train-updates-per-step",
            "2",
        ]
    )
    config = _apply_legacy_cli_aliases(_train_config_from_args(args), args)

    assert config.num_envs == 8
    assert config.updates_per_transition == 0.25


def test_cli_rejects_both_update_rate_interfaces():
    args = parse_args(
        [
            "--train-updates-per-step",
            "2",
            "--updates-per-transition",
            "0.25",
        ]
    )

    with pytest.raises(ValueError, match="cannot be used together"):
        _train_config_from_args(args)


def test_small_cpu_training_smoke_runs_without_dashboard():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        observation_mode="full",
        episodes=3,
        seed=123,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=4,
        target_update_freq=2,
        num_envs=1,
        updates_per_transition=1.0,
        eval_every_steps=2,
        eval_episodes=1,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))

    tracker = train(agent=agent, config=config)

    assert len(agent.buffer) > 0
    assert agent.buffer.next_action_masks.shape == (128, 4)
    assert len(tracker.solved) == 3
    assert tracker.latest_eval.solve_rate >= 0.0


def test_vectorized_dqn_trains_with_prioritized_replay_after_warmup():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        episodes=3,
        seed=321,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=4,
        warmup_steps=4,
        updates_per_transition=1.0,
        prioritized_replay=True,
        target_tau=0.1,
        num_envs=2,
        eval_every_steps=1_000,
        eval_episodes=1,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        vectorized_envs=True,
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))

    train(agent=agent, config=config)

    assert agent.update_count > 0
    assert len(agent.buffer) >= config.min_replay_size
    assert np.all(agent.buffer.priorities[: len(agent.buffer)] > 0)


def test_ppo_masked_sampling_never_chooses_invalid_actions():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="ppo",
        observation_mode="full",
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)
    state = env.reset()
    mask = env.valid_action_mask()
    agent = MaskedPPOAgent(config=config, device=torch.device("cpu"))

    actions, _log_probs, values = agent.sample_actions(
        np.stack([state] * 20),
        np.stack([mask] * 20),
    )

    assert set(actions.tolist()) == {0}
    assert values.shape == (20,)


def test_small_cpu_ppo_training_smoke_runs_without_dashboard():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="ppo",
        observation_mode="full",
        episodes=3,
        seed=123,
        max_episode_steps=10,
        num_envs=1,
        batch_size=4,
        ppo_rollout_steps=4,
        ppo_epochs=1,
        eval_every_steps=2,
        eval_episodes=1,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
    )
    agent = MaskedPPOAgent(config=config, device=torch.device("cpu"))

    tracker = train(agent=agent, config=config)

    assert agent.update_count > 0
    assert len(tracker.solved) == 3
    assert tracker.latest_eval.solve_rate >= 0.0


def test_agent_checkpoint_restores_optimizer_and_counters(tmp_path):
    save_path = tmp_path / "checkpoint.pth"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        save_path=str(save_path),
        device="cpu",
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))
    agent.update_count = 7
    agent.total_env_steps = 42
    agent.best_greedy_solve_rate = 0.25

    agent.save(str(save_path))
    resumed = MouseAgent(config=config, device=torch.device("cpu"))
    resumed.load(str(save_path))

    assert resumed.update_count == 7
    assert resumed.total_env_steps == 42
    assert resumed.best_greedy_solve_rate == 0.25
    assert resumed.network_type == "spatial"
    assert resumed.optimizer.state_dict()["param_groups"] == (
        agent.optimizer.state_dict()["param_groups"]
    )


def test_checkpoint_can_restore_optional_replay_contents(tmp_path):
    save_path = tmp_path / "replay_checkpoint.pth"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        observation_mode="full",
        buffer_size=16,
        min_replay_size=4,
        checkpoint_replay=True,
        save_path=str(save_path),
        dashboard_flag=False,
        training_log_path=None,
        device="cpu",
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))
    state = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10).reset()
    agent.store_transition(state, 0, 1.0, state, False, np.ones(4, dtype=np.bool_))
    agent.save(str(save_path))

    resumed = MouseAgent(config=config, device=torch.device("cpu"))
    resumed.load(str(save_path))

    assert len(resumed.buffer) == 1
    assert resumed.buffer.actions[0] == 0
    assert resumed.buffer.rewards[0] == 1.0


def test_periodic_latest_checkpoint_uses_sidecar_and_cadence(tmp_path):
    save_path = tmp_path / "best.pth"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        observation_mode="full",
        latest_checkpoint_every_steps=10,
        save_path=str(save_path),
        dashboard_flag=False,
        training_log_path=None,
        device="cpu",
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))
    logger = mouse_agent_module._TrainingLogger(None)

    assert not _save_latest_checkpoint(agent, config, logger, 9, "periodic")
    assert not save_path.exists()
    agent.total_env_steps = 10
    agent.training_state = {"total_steps": 10, "completed_episodes": 2}
    assert _save_latest_checkpoint(agent, config, logger, 10, "periodic")

    sidecar = tmp_path / "best.latest.pth"
    payload = torch.load(sidecar, map_location=torch.device("cpu"), weights_only=False)
    assert sidecar.exists()
    assert not save_path.exists()
    assert payload["total_env_steps"] == 10
    assert payload["training_state"]["last_latest_checkpoint_step"] == 10
    assert not _save_latest_checkpoint(agent, config, logger, 19, "periodic")


def test_interruption_saves_latest_sidecar(monkeypatch, tmp_path):
    save_path = tmp_path / "interrupted.pth"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        observation_mode="full",
        episodes=1,
        num_envs=1,
        resume=False,
        save_path=str(save_path),
        dashboard_flag=False,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))

    def interrupt_training(agent, *_args):
        agent.total_env_steps = 37
        agent.training_state = {"total_steps": 37, "completed_episodes": 3}
        raise KeyboardInterrupt

    monkeypatch.setattr(mouse_agent_module, "_train_dqn", interrupt_training)

    with pytest.raises(KeyboardInterrupt):
        train(agent=agent, config=config)

    payload = torch.load(
        tmp_path / "interrupted.latest.pth",
        map_location=torch.device("cpu"),
        weights_only=False,
    )
    assert payload["total_env_steps"] == 37
    assert payload["training_state"]["total_steps"] == 37


def test_sidecar_resume_keeps_current_and_frozen_best_separate(
    monkeypatch,
    tmp_path,
):
    save_path = tmp_path / "recurrent.pth"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        recurrent_hidden_size=16,
        resume=True,
        dashboard_flag=False,
        save_path=str(save_path),
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    best_agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in best_agent.policy_net.parameters():
            parameter.fill_(1.0)
    best_agent.best_greedy_solve_rate = 0.9
    best_agent.total_env_steps = 80
    best_agent.training_state = {"total_steps": 80, "completed_episodes": 8}
    best_agent.save(str(save_path))

    latest_agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    with torch.no_grad():
        for parameter in latest_agent.policy_net.parameters():
            parameter.fill_(2.0)
    latest_agent.best_greedy_solve_rate = 0.9
    latest_agent.total_env_steps = 100
    latest_agent.training_state = {"total_steps": 100, "completed_episodes": 10}
    latest_agent.save(latest_checkpoint_path(str(save_path)))

    captured = []

    def capture_training(
        agent,
        _config,
        _logger,
        _start_time,
        _process_start_cpu,
        *,
        initial_best_weights,
        initial_best_optimizer_state,
    ):
        first_parameter_name = next(iter(dict(agent.policy_net.named_parameters())))
        current_value = float(
            agent.policy_net.state_dict()[first_parameter_name].mean()
        )
        best_value = (
            None
            if initial_best_weights is None
            else float(initial_best_weights[first_parameter_name].mean())
        )
        captured.append(
            (current_value, best_value, initial_best_optimizer_state is not None)
        )
        return MetricsTracker()

    monkeypatch.setattr(
        mouse_agent_module,
        "_train_recurrent_ppo",
        capture_training,
    )
    resumed = RecurrentPPOAgent(config, device=torch.device("cpu"))
    train(agent=resumed, config=config)

    assert resumed.total_env_steps == 100
    assert captured[-1] == (2.0, 1.0, True)

    (tmp_path / "recurrent.latest.pth").unlink()
    legacy_resumed = RecurrentPPOAgent(config, device=torch.device("cpu"))
    train(agent=legacy_resumed, config=config)

    assert legacy_resumed.total_env_steps == 80
    assert captured[-1] == (1.0, None, False)


def test_train_resumes_existing_save_path_only_when_requested(tmp_path, capsys):
    save_path = tmp_path / "resume_weights.pth"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        episodes=1,
        seed=123,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=128,
        target_update_freq=2,
        num_envs=1,
        updates_per_transition=1.0,
        eval_every_steps=1,
        post_curriculum_eval_every_steps=1,
        eval_episodes=1,
        dashboard_flag=False,
        save_path=str(save_path),
        training_log_path=None,
        device="cpu",
        resume=True,
    )
    first_agent = MouseAgent(config=config, device=torch.device("cpu"))
    first_agent.update_count = 11
    first_agent.total_env_steps = 99
    first_agent.save(str(save_path))
    resumed_agent = MouseAgent(config=config, device=torch.device("cpu"))

    train(agent=resumed_agent, config=config)
    output = capsys.readouterr().out

    assert "[train] resumed training state from" in output
    assert str(save_path) in output
    assert resumed_agent.update_count == 11
    assert resumed_agent.total_env_steps >= 99

    fresh_config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        episodes=1,
        seed=123,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=128,
        target_update_freq=2,
        num_envs=1,
        updates_per_transition=1.0,
        eval_every_steps=1,
        post_curriculum_eval_every_steps=1,
        eval_episodes=1,
        dashboard_flag=False,
        save_path=str(save_path),
        training_log_path=None,
        device="cpu",
        resume=False,
    )
    fresh_agent = MouseAgent(config=fresh_config, device=torch.device("cpu"))

    train(agent=fresh_agent, config=fresh_config)
    output = capsys.readouterr().out

    assert "[train] starting a fresh experiment" in output
    assert fresh_agent.update_count != 11


def test_training_does_not_log_duplicate_eval_episodes(tmp_path, monkeypatch):
    class ControlledEnv:
        def __init__(self, done_after, solved):
            self.done_after = done_after
            self.solved = solved
            self.start = (1, 1)
            self.goal = (1, 2)
            self.current_position = self.start
            self.max_episode_steps = done_after
            self.optimal_start_steps = 1
            self.invalid_moves = 0
            self.total_reward = 0.0
            self.steps = 0

        def reset(self):
            self.current_position = self.start
            self.invalid_moves = 0
            self.total_reward = 0.0
            self.steps = 0
            return np.zeros((4, 5, 5), dtype=np.float32)

        def valid_action_mask(self):
            return np.ones(4, dtype=np.bool_)

        def step(self, action):
            self.steps += 1
            done = self.steps >= self.done_after
            if done and self.solved:
                self.current_position = self.goal
                self.total_reward = 1.0
            timeout = done and not self.solved
            return (
                    np.zeros((4, 5, 5), dtype=np.float32),
                0.0,
                done,
                {
                    "invalid": False,
                    "moved": True,
                    "solved": done and self.solved,
                    "timeout": timeout,
                    "distance": 0.0,
                    "optimal_steps": self.optimal_start_steps,
                    "steps": self.steps,
                    "invalid_moves": self.invalid_moves,
                },
            )

    envs = [
        ControlledEnv(done_after=1, solved=True),
        ControlledEnv(done_after=5, solved=False),
        ControlledEnv(done_after=5, solved=False),
    ]

    def fake_make_maze(config, rng=None):
        if envs:
            return envs.pop(0)
        return ControlledEnv(done_after=5, solved=False)

    monkeypatch.setattr(mouse_agent_module, "make_maze", fake_make_maze)
    monkeypatch.setattr(
        mouse_agent_module,
        "_eval_greedy",
        lambda agent, config: EvalMetrics(solve_rate=0.0),
    )
    log_path = tmp_path / "training_log.jsonl"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        observation_mode="full",
        episodes=2,
        seed=123,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=128,
        target_update_freq=2,
        num_envs=2,
        updates_per_transition=1.0,
        eval_every_steps=100,
        eval_episodes=1,
        dashboard_flag=False,
        save_path=None,
        training_log_path=str(log_path),
        device="cpu",
    )

    train(agent=MouseAgent(config=config, device=torch.device("cpu")), config=config)
    eval_episodes = [
        json.loads(line)["episode"]
        for line in log_path.read_text().splitlines()
        if json.loads(line)["event"] == "eval"
    ]

    assert eval_episodes == sorted(set(eval_episodes))
    assert eval_episodes == [1, 2]


def test_training_writes_jsonl_log_with_metrics_speed_and_utilization(tmp_path):
    log_path = tmp_path / "training_log.jsonl"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        episodes=1,
        seed=123,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=128,
        target_update_freq=2,
        num_envs=1,
        updates_per_transition=1.0,
        eval_every_steps=1,
        post_curriculum_eval_every_steps=1,
        eval_episodes=1,
        dashboard_flag=False,
        save_path=None,
        training_log_path=str(log_path),
        device="cpu",
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))

    train(agent=agent, config=config)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    events = [record["event"] for record in records]
    eval_record = next(record for record in records if record["event"] == "eval")

    assert events[0] == "train_start"
    assert "eval" in events
    assert events[-1] == "train_end"
    assert records[0]["config"]["episodes"] == 1
    assert set(records[0]["config"]) == {field.name for field in fields(config)}
    assert records[0]["artifact_timestamp"]
    assert records[0]["artifact_paths"]["log"] == str(log_path)
    assert records[0]["environment"]["selected_device"] == "cpu"
    assert "command_line" in records[0]["environment"]
    assert "working_directory" in records[0]["environment"]
    assert "hostname" in records[0]["environment"]
    assert "packages" in records[0]["environment"]
    assert "git" in records[0]["environment"]
    assert "metrics" in eval_record
    assert "greedy" in eval_record
    assert "steps_per_second" in eval_record["speed"]
    assert "process_cpu_percent" in eval_record["utilization"]
    assert all(record["timestamp"].endswith("+00:00") for record in records)
    assert all(isinstance(record["time_unix"], float) for record in records)


def test_resumed_speed_uses_process_local_counter_deltas(monkeypatch):
    monkeypatch.setattr(mouse_agent_module.time, "perf_counter", lambda: 110.0)

    speed = _training_speed_payload(
        completed=1_005,
        total_steps=10_100,
        start_time=100.0,
        start_completed=1_000,
        start_total_steps=10_000,
    )

    assert speed["process_steps"] == 100
    assert speed["process_episodes"] == 5
    assert speed["steps_per_second"] == 10.0
    assert speed["episodes_per_second"] == 0.5


def test_training_saves_new_best_weights_to_configured_path(tmp_path, capsys):
    save_path = tmp_path / "best_weights.pth"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        episodes=1,
        seed=123,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=4,
        target_update_freq=2,
        num_envs=1,
        updates_per_transition=1.0,
        eval_every_steps=1,
        eval_episodes=1,
        dashboard_flag=False,
        save_path=str(save_path),
        training_log_path=None,
        device="cpu",
    )
    agent = MouseAgent(config=config, device=torch.device("cpu"))

    tracker = train(agent=agent, config=config)
    output = capsys.readouterr().out
    payload = torch.load(save_path, map_location=torch.device("cpu"))

    assert save_path.exists()
    assert "state_dict" in payload
    assert payload["observation_shape"] == agent.observation_shape
    assert "[train] new best weights saved to" in output
    assert str(save_path) in output
    assert tracker.latest_eval.solve_rate >= 0.0


def test_remaining_time_channel_is_observable_and_decreases():
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)

    initial = env.reset()
    next_state, _reward, _done, _info = env.step(0)

    assert np.all(initial[3] == 1.0)
    assert np.allclose(next_state[3], 0.9)


def test_vectorized_local_observations_match_sequential_environment():
    sequential = Maze(
        fixed_grid(),
        observation_mode="local",
        view_size=5,
        max_episode_steps=10,
    )
    batched = Maze(
        fixed_grid(),
        observation_mode="local",
        view_size=5,
        max_episode_steps=10,
    )
    batch = MazeBatch([batched])

    assert np.array_equal(batch.observations()[0], sequential.reset())
    for action in (0, 0):
        expected, expected_reward, expected_done, _info = sequential.step(action)
        result = batch.step(np.array([action]))
        assert np.array_equal(result.states[0], expected)
        assert np.isclose(result.rewards[0], expected_reward)
        assert bool(result.dones[0]) is expected_done


def test_recurrent_sequence_matches_step_unroll_and_resets_hidden_state():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="full",
        num_envs=2,
        ppo_rollout_steps=4,
        recurrent_sequence_length=2,
        recurrent_hidden_size=16,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    observation = torch.from_numpy(
        np.stack(
            [
                Maze(fixed_grid(), observation_mode="full").reset(),
                Maze(fixed_grid(), observation_mode="full").reset(),
            ]
        )
    )
    observations = observation.unsqueeze(0).repeat(2, 1, 1, 1, 1)
    previous_actions = torch.tensor([[-1, -1], [0, 0]])
    previous_rewards = torch.zeros(2, 2)
    episode_starts = torch.tensor([[True, True], [False, False]])
    initial_hidden = agent.initial_policy_state(2)

    sequence_logits, sequence_values, sequence_hidden = agent.policy_net.forward_sequence(
        observations,
        previous_actions,
        previous_rewards,
        episode_starts,
        initial_hidden,
    )
    hidden = initial_hidden
    step_logits = []
    step_values = []
    for step in range(2):
        logits, values, hidden = agent.policy_net.forward_step(
            observations[step],
            previous_actions[step],
            previous_rewards[step],
            episode_starts[step],
            hidden,
        )
        step_logits.append(logits)
        step_values.append(values)

    assert torch.allclose(sequence_logits, torch.stack(step_logits))
    assert torch.allclose(sequence_values, torch.stack(step_values))
    assert torch.allclose(sequence_hidden, hidden)
    reset_logits, _values, _hidden = agent.policy_net.forward_step(
        observation,
        torch.full((2,), -1),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        torch.randn_like(initial_hidden),
    )
    zero_logits, _values, _hidden = agent.policy_net.forward_step(
        observation,
        torch.full((2,), -1),
        torch.zeros(2),
        torch.ones(2, dtype=torch.bool),
        initial_hidden,
    )
    assert torch.allclose(reset_logits, zero_logits)


def test_local_recurrent_agent_uses_rnd_without_distance_shaping():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        distance_shaping_mode="potential",
        num_envs=1,
        ppo_rollout_steps=4,
        recurrent_sequence_length=2,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    observation = Maze(
        fixed_grid(), observation_mode="local", view_size=5, max_episode_steps=10
    ).reset()
    states = torch.from_numpy(observation).view(1, 1, 5, 5, 5).repeat(2, 1, 1, 1, 1)

    assert config.distance_shaping_mode == "none"
    assert agent.rnd is not None
    bonuses = agent.rnd.bonus_and_update(states, config.rnd_reward_clip)
    assert bonuses.shape == (2, 1)
    assert torch.isfinite(bonuses).all()
    assert (bonuses >= 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_rnd_running_variance_uses_configured_cuda_device():
    device = torch.device("cuda")
    rnd = RNDModule((4, 5, 5), device)
    observations = torch.zeros((2, 1, 4, 5, 5), device=device)

    bonuses = rnd.bonus_and_update(observations, clip=5.0)

    assert rnd.error_variance.device == observations.device
    assert bonuses.device == observations.device
    assert torch.isfinite(bonuses).all()


def test_prefetched_generation_is_seed_deterministic():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        maze_workers=0,
    )

    first = _generate_prefetched_grid(config, 12345, (3, 5))
    second = _generate_prefetched_grid(config, 12345, (3, 5))

    assert np.array_equal(first, second)


def test_process_prefetcher_preserves_seeded_task_order():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        curriculum_enabled=True,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        maze_workers=1,
        maze_generation_batch_size=1,
        maze_prefetch_batches_per_worker=1,
    )
    expected_sampler = MazeTaskSampler(config, random.Random(91))
    expected_stage = expected_sampler.sample_stage()
    expected_seed = expected_sampler.rng.randrange(0, 2**63)
    expected = _generate_prefetched_grid(
        replace(config, maze_size=expected_stage.maze_size),
        expected_seed,
        expected_stage.distance_range,
        expected_stage.complexity_high,
    )
    actual_sampler = MazeTaskSampler(config, random.Random(91))
    prefetcher = DeterministicMazePrefetcher(actual_sampler, workers=1)
    try:
        actual = prefetcher.next().grid
    finally:
        prefetcher.close()

    assert np.array_equal(actual, expected)


def test_process_prefetcher_prioritizes_hard_replay_variants():
    config = TrainConfig(
        maze_size=(5, 5),
        hard_maze_fraction=1.0,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        maze_workers=1,
        maze_generation_batch_size=1,
        maze_prefetch_batches_per_worker=1,
    )
    sampler = MazeTaskSampler(config, random.Random(17))
    sampler.add_failed_grids([fixed_grid()])
    sampler.record_validation_solve_rate(1.0)
    prefetcher = DeterministicMazePrefetcher(sampler, workers=1)
    try:
        environment = prefetcher.next()
    finally:
        prefetcher.close()

    assert any(
        np.array_equal(environment.grid, variant)
        for variant in sampler.hard_grids
    )


def test_recurrent_checkpoint_v3_round_trip_and_rejects_legacy(tmp_path):
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        recurrent_hidden_size=16,
        ppo_rollout_steps=4,
        recurrent_sequence_length=2,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    checkpoint = tmp_path / "recurrent_v3.pth"
    legacy = tmp_path / "legacy.pth"
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    agent.update_count = 7
    agent.total_env_steps = 42
    agent.training_state = {
        "precision_recovery": {
            "plateau_evals": 3,
            "active": True,
            "end_step": 99,
            "count": 2,
        },
        "hard_maze_seen_keys": [fixed_grid().tobytes()],
    }
    agent.save(str(checkpoint))

    restored = RecurrentPPOAgent(config, device=torch.device("cpu"))
    restored.load(str(checkpoint))
    torch.save({"schema_version": 2, "algorithm": "dqn"}, legacy)

    assert restored.update_count == 7
    assert restored.total_env_steps == 42
    assert restored.training_state == agent.training_state
    with pytest.raises(ValueError, match="schema-v3"):
        restored.load(str(legacy))


def test_short_local_recurrent_training_smoke():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        episodes=3,
        max_env_steps=64,
        max_episode_steps=10,
        num_envs=2,
        ppo_rollout_steps=4,
        recurrent_sequence_length=2,
        recurrent_sequence_minibatch_size=2,
        ppo_epochs=1,
        target_only_stop=False,
        eval_every_steps=1_000,
        eval_episodes=1,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))

    tracker = train(agent=agent, config=config)

    assert agent.update_count > 0
    assert len(tracker.solved) == 3


def test_recurrent_resume_uses_saved_curriculum_definitions(monkeypatch):
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        episodes=1,
        max_env_steps=1,
        max_episode_steps=4,
        num_envs=1,
        ppo_rollout_steps=1,
        ppo_epochs=1,
        recurrent_hidden_size=16,
        recurrent_sequence_length=1,
        recurrent_sequence_minibatch_size=1,
        rnd_reward_coef=0.0,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    saved_stage = CurriculumStage((5, 5), "saved-unrestricted")
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    agent.total_env_steps = 1
    agent.training_state = {
        "completed_episodes": 1,
        "total_steps": 1,
        "curriculum": {
            "level": 0,
            "success_streak": 0,
            "stages": [saved_stage.payload()],
        },
    }
    monkeypatch.setattr(
        mouse_agent_module,
        "resolve_curriculum_stages",
        lambda _config: pytest.fail("resume recomputed curriculum stages"),
    )

    train(agent=agent, config=config)

    assert agent.training_state["curriculum"]["stages"] == [saved_stage.payload()]


def test_short_recurrent_training_promotes_automatic_curriculum(monkeypatch, tmp_path):
    monkeypatch.setattr(
        mouse_agent_module,
        "_eval_greedy",
        lambda agent, config, **kwargs: EvalMetrics(solve_rate=1.0),
    )
    log_path = tmp_path / "auto-curriculum.jsonl"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        episodes=10,
        max_env_steps=10,
        max_episode_steps=4,
        num_envs=1,
        ppo_rollout_steps=1,
        ppo_epochs=1,
        recurrent_hidden_size=16,
        recurrent_sequence_length=1,
        recurrent_sequence_minibatch_size=1,
        rnd_reward_coef=0.0,
        eval_every_steps=1,
        post_curriculum_eval_every_steps=1,
        eval_episodes=1,
        curriculum_probe_mazes=8,
        curriculum_promotion_rate=0.9,
        curriculum_promotion_evals=1,
        curriculum_eval_episodes=1,
        dashboard_flag=False,
        save_path=None,
        training_log_path=str(log_path),
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))

    train(agent=agent, config=config)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    promotions = [
        record for record in records
        if record["event"] == "curriculum" and record["promoted"]
    ]
    learners = [record for record in records if record["event"] == "learner"]

    assert len(promotions) == 2
    assert promotions[-1]["stage"]["name"] == "5x5-unrestricted"
    assert all("active_eval_every_steps" in record for record in learners)


def test_target_only_recurrent_ppo_confirms_frozen_candidate_and_ignores_legacy_streak(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        mouse_agent_module,
        "_eval_greedy",
        lambda agent, config, **kwargs: EvalMetrics(solve_rate=1.0),
    )
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        episodes=1,
        max_env_steps=1,
        max_episode_steps=4,
        num_envs=1,
        ppo_rollout_steps=1,
        ppo_epochs=1,
        recurrent_hidden_size=16,
        recurrent_sequence_length=1,
        recurrent_sequence_minibatch_size=1,
        eval_every_steps=1,
        post_curriculum_eval_every_steps=1,
        eval_episodes=1,
        target_solve_rate=1.0,
        target_solve_evals=2,
        target_only_stop=True,
        curriculum_enabled=True,
        curriculum_promotion_rate=0.9,
        curriculum_promotion_evals=3,
        curriculum_eval_episodes=1,
        dashboard_flag=False,
        save_path=str(tmp_path / "confirmed.pth"),
        training_log_path=str(tmp_path / "training.jsonl"),
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    agent.best_greedy_solve_rate = 1.0
    agent.training_state = {"target_streak": 99}

    train(agent=agent, config=config)
    records = [
        json.loads(line)
        for line in (tmp_path / "training.jsonl").read_text().splitlines()
    ]

    assert config.target_only_stop is True
    assert agent.total_env_steps >= 6
    assert agent.training_state["target_confirmed"] is True
    assert "target_streak" not in agent.training_state
    assert any(
        record["event"] == "target_confirmation" and record["confirmed"] is True
        for record in records
    )
    assert any(
        record["event"] == "checkpoint"
        and record["reason"] == "target_confirmed"
        for record in records
    )
    assert all(
        "entropy_coefficient" in record
        for record in records
        if record["event"] == "learner"
    )
    assert all(
        "effective_transition_minibatch_size" in record["ppo"]
        and "optimizer_updates_this_rollout" in record["ppo"]
        for record in records
        if record["event"] == "learner"
    )


def test_failed_target_confirmation_resumes_training(monkeypatch):
    rates = deque([1.0, 0.0, 1.0, 1.0])

    def fake_eval(agent, config, **kwargs):
        del agent, config, kwargs
        solve_rate = rates.popleft() if rates else 1.0
        return EvalMetrics(
            solve_rate=solve_rate,
            failed_grids=[fixed_grid()] if solve_rate < 1.0 else [],
        )

    monkeypatch.setattr(mouse_agent_module, "_eval_greedy", fake_eval)
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        episodes=1,
        max_env_steps=1,
        max_episode_steps=4,
        num_envs=1,
        ppo_rollout_steps=1,
        ppo_epochs=1,
        recurrent_hidden_size=16,
        recurrent_sequence_length=1,
        recurrent_sequence_minibatch_size=1,
        eval_every_steps=1,
        post_curriculum_eval_every_steps=1,
        eval_episodes=1,
        target_solve_rate=1.0,
        target_solve_evals=2,
        target_only_stop=True,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=None,
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))

    train(agent=agent, config=config)

    assert agent.training_state["target_confirmed"] is True
    assert agent.total_env_steps >= 2
    assert "hard_maze_grids" in agent.training_state
    assert not rates


def test_disabled_precision_recovery_clears_legacy_active_state(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        mouse_agent_module,
        "_eval_greedy",
        lambda agent, config, **kwargs: EvalMetrics(solve_rate=1.0),
    )
    log_path = tmp_path / "disabled-recovery.jsonl"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        episodes=1,
        max_env_steps=1,
        max_episode_steps=4,
        num_envs=1,
        ppo_rollout_steps=1,
        ppo_epochs=1,
        recurrent_hidden_size=16,
        recurrent_sequence_length=1,
        recurrent_sequence_minibatch_size=1,
        rnd_reward_coef=0.0,
        eval_every_steps=1,
        eval_episodes=1,
        target_solve_rate=1.0,
        target_solve_evals=2,
        target_only_stop=True,
        precision_recovery_enabled=False,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=str(tmp_path / "legacy-recovery.pth"),
        training_log_path=str(log_path),
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    legacy_agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    legacy_agent.training_state = {
        "total_steps": 0,
        "completed_episodes": 0,
        "precision_recovery": {
            "plateau_evals": 12,
            "active": True,
            "end_step": 999,
            "count": 4,
        },
    }
    legacy_agent.save(config.save_path)
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))

    train(agent=agent, config=config)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]

    assert agent.training_state["precision_recovery"] == {
        "enabled": False,
        "plateau_evals": 0,
        "active": False,
        "end_step": 0,
        "count": 0,
    }
    assert not any(record["event"] == "precision_recovery" for record in records)
    learner = next(record for record in records if record["event"] == "learner")
    assert learner["precision_recovery_active"] is False
    assert learner["timing"]["estimated_seconds_remaining"] is None


@pytest.mark.parametrize(
    ("third_rate", "expected_outcome"),
    [(0.6, "improved"), (0.5, "rollback")],
)
def test_precision_recovery_handles_improvement_and_rollback(
    monkeypatch,
    tmp_path,
    third_rate,
    expected_outcome,
):
    rates = deque([0.5, 0.5, third_rate, 1.0, 1.0])

    def fake_eval(agent, config, **kwargs):
        del agent, config, kwargs
        return EvalMetrics(solve_rate=rates.popleft() if rates else 1.0)

    monkeypatch.setattr(mouse_agent_module, "_eval_greedy", fake_eval)
    log_path = tmp_path / f"recovery-{expected_outcome}.jsonl"
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="recurrent_ppo",
        observation_mode="local",
        view_size=5,
        episodes=1,
        max_env_steps=1,
        max_episode_steps=4,
        num_envs=1,
        ppo_rollout_steps=1,
        ppo_epochs=1,
        recurrent_hidden_size=16,
        recurrent_sequence_length=1,
        recurrent_sequence_minibatch_size=1,
        rnd_reward_coef=0.0,
        eval_every_steps=1,
        post_curriculum_eval_every_steps=1,
        eval_episodes=1,
        target_solve_rate=1.0,
        target_solve_evals=2,
        target_only_stop=True,
        precision_plateau_evals=1,
        precision_recovery_steps=1,
        precision_recovery_lr_fraction=0.05,
        precision_recovery_enabled=True,
        curriculum_enabled=False,
        dashboard_flag=False,
        save_path=None,
        training_log_path=str(log_path),
        device="cpu",
        require_cuda=False,
        performance_profile="portable",
        maze_workers=0,
    )
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))

    train(agent=agent, config=config)
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    recovery_records = [
        record for record in records if record["event"] == "precision_recovery"
    ]
    recovery_learners = [
        record
        for record in records
        if record["event"] == "learner"
        and record["precision_recovery_active"] is True
    ]

    assert any(record["phase"] == "start" for record in recovery_records)
    assert any(
        record["phase"] == "end" and record["outcome"] == expected_outcome
        for record in recovery_records
    )
    assert recovery_learners
    assert all(
        np.isclose(
            record["learning_rate"],
            config.learning_rate * config.precision_recovery_lr_fraction,
        )
        for record in recovery_learners
    )
    assert agent.training_state["precision_recovery"]["active"] is False
    assert agent.training_state["target_confirmed"] is True
    assert not rates
