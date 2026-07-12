import json
import random
from collections import deque
from dataclasses import fields
from datetime import datetime, timezone

import numpy as np
import pytest
import torch

import MouseAgent as mouse_agent_module
from MouseAgent import (
    DASHBOARD_TOOLTIPS,
    DEFAULT_INFER_FLAG,
    DEFAULT_INFERENCE_MAZES,
    DEFAULT_TRAIN_FLAG,
    BfsPlanner,
    CurriculumController,
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
    SumTree,
    ReplayBuffer,
    TrainConfig,
    _chart_x_max,
    _dashboard_tooltip_at,
    _draw_cheese_icon,
    _draw_mouse_icon,
    _draw_start_icon,
    _episode_tick_values,
    _eval_greedy,
    _format_chart_tick,
    _generate_prefetched_grid,
    _optional_bool,
    _rects_overlap,
    _restore_rng_state,
    _train_config_from_args,
    _training_environment_payload,
    default_artifact_paths,
    invocation_timestamp,
    latest_model_path,
    paired_log_path,
    resolve_cli_artifacts,
    build_n_step_transition,
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

    assert obs.shape == (4, 5, 5)
    assert obs[1, 2, 2] == 1.0
    assert obs[1].sum() == 1.0
    assert obs[2, 2, 4] == 1.0
    assert np.all(obs[3] == 1.0)


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


def test_curriculum_promotes_only_after_repeated_validation_success():
    config = TrainConfig(
        maze_size=(5, 5),
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


def test_episode_tick_values_scale_without_clutter():
    small_ticks = _episode_tick_values(3, 320)
    large_ticks = _episode_tick_values(10_000, 320)

    assert small_ticks == [0, 1, 2, 3]
    assert large_ticks[0] == 0
    assert large_ticks[-1] == 10_000
    assert len(large_ticks) <= 6
    assert _chart_x_max(5, [(8, 0.5)]) == 8


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


def test_target_only_stop_defaults_to_recurrent_ppo_and_can_be_disabled():
    assert TrainConfig(algorithm="recurrent_ppo").target_only_stop is True
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
    unrelated = (tmp_path / "latest_mousemaze.pth", tmp_path / "notes.txt")
    for path in (newer, older, *unrelated):
        path.write_text("test")

    assert latest_model_path(str(tmp_path)) == str(newer)


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
    args = parse_args(["--train"])

    config = resolve_cli_artifacts(
        _train_config_from_args(args), args, "20260710T123456123456Z"
    )

    assert config.save_path == str(model_path)
    assert config.training_log_path == str(log_path)


def test_resume_artifact_resolution_falls_back_when_no_model_exists(monkeypatch):
    def no_model():
        raise FileNotFoundError("no model")

    monkeypatch.setattr(mouse_agent_module, "latest_model_path", no_model)
    args = parse_args(["--train"])

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
    args = parse_args(["--train"])

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

    def fake_generate(rows, cols):
        generated_sizes.append((rows, cols))
        return np.zeros((rows, cols), dtype=np.uint8)

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
        lambda rows, cols: np.zeros((rows, cols), dtype=np.uint8),
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
            "--distance-shaping-mode",
            "none",
            "--no-curriculum",
            "--curriculum-easy-range",
            "4",
            "8",
            "--n-step-returns",
            "4",
            "--ppo-rollout-steps",
            "8",
            "--recurrent-hidden-size",
            "64",
            "--target-solve-rate",
            "0.8",
            "--performance-profile",
            "portable",
            "--maze-workers",
            "0",
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
    assert config.eval_every == 3
    assert config.dashboard_every == 3
    assert config.eval_seed == 99
    assert config.algorithm == "ppo"
    assert config.network_type == "spatial"
    assert config.resume is True
    assert config.timeout_step_factor == 3.5
    assert config.min_episode_steps == 7
    assert config.distance_shaping_mode == "none"
    assert config.curriculum_enabled is False
    assert config.curriculum_easy_range == (4, 8)
    assert config.n_step_returns == 4
    assert config.ppo_rollout_steps == 8
    assert config.recurrent_hidden_size == 64
    assert config.target_solve_rate == 0.8
    assert config.performance_profile == "portable"
    assert config.maze_workers == 0
    assert config.learning_rate == 0.001
    assert config.device == "cpu"
    assert _optional_bool(args.infer_flag, DEFAULT_INFER_FLAG) is False


def test_cli_dashboard_every_can_override_eval_every():
    args = parse_args(["--eval-every", "3", "--dashboard-every", "5"])

    config = _train_config_from_args(args)

    assert config.eval_every == 3
    assert config.dashboard_every == 5


def test_small_cpu_training_smoke_runs_without_dashboard():
    config = TrainConfig(
        maze_size=(5, 5),
        algorithm="dqn",
        episodes=3,
        seed=123,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=4,
        target_update_freq=2,
        num_envs=1,
        train_updates_per_step=1,
        eval_every=2,
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
        episodes=3,
        seed=123,
        max_episode_steps=10,
        num_envs=1,
        batch_size=4,
        ppo_rollout_steps=4,
        ppo_epochs=1,
        eval_every=2,
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
        train_updates_per_step=1,
        eval_every=1,
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

    assert "[train] resumed weights from" in output
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
        train_updates_per_step=1,
        eval_every=1,
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

    assert "[train] starting fresh" in output
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
        episodes=2,
        seed=123,
        max_episode_steps=10,
        buffer_size=128,
        batch_size=4,
        min_replay_size=128,
        target_update_freq=2,
        num_envs=2,
        train_updates_per_step=1,
        eval_every=100,
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
        train_updates_per_step=1,
        eval_every=1,
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
        train_updates_per_step=1,
        eval_every=1,
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
        np.stack([Maze(fixed_grid()).reset(), Maze(fixed_grid()).reset()])
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
    states = torch.from_numpy(observation).view(1, 1, 4, 5, 5).repeat(2, 1, 1, 1, 1)

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
    )
    expected_sampler = MazeTaskSampler(config, random.Random(91))
    expected_range = expected_sampler.sample_target_range()
    expected_seed = expected_sampler.rng.randrange(0, 2**63)
    expected = _generate_prefetched_grid(config, expected_seed, expected_range)
    actual_sampler = MazeTaskSampler(config, random.Random(91))
    prefetcher = DeterministicMazePrefetcher(actual_sampler, workers=1)
    try:
        actual = prefetcher.next().grid
    finally:
        prefetcher.close()

    assert np.array_equal(actual, expected)


def test_recurrent_checkpoint_v2_round_trip_and_rejects_legacy(tmp_path):
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
    checkpoint = tmp_path / "recurrent_v2.pth"
    legacy = tmp_path / "legacy.pth"
    agent = RecurrentPPOAgent(config, device=torch.device("cpu"))
    agent.update_count = 7
    agent.total_env_steps = 42
    agent.save(str(checkpoint))

    restored = RecurrentPPOAgent(config, device=torch.device("cpu"))
    restored.load(str(checkpoint))
    torch.save({"schema_version": 1, "algorithm": "dqn"}, legacy)

    assert restored.update_count == 7
    assert restored.total_env_steps == 42
    with pytest.raises(ValueError, match="schema-v2"):
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


def test_target_only_recurrent_ppo_ignores_caps_until_curriculum_completes(monkeypatch):
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
        eval_episodes=1,
        target_solve_rate=0.9,
        target_solve_evals=2,
        curriculum_enabled=True,
        curriculum_promotion_rate=0.9,
        curriculum_promotion_evals=3,
        curriculum_eval_episodes=1,
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

    assert config.target_only_stop is True
    assert agent.total_env_steps >= 6
