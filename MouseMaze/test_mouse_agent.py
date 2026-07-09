import json
import random
from collections import deque

import numpy as np
import torch

import MouseAgent as mouse_agent_module
from MouseAgent import (
    DASHBOARD_TOOLTIPS,
    DEFAULT_INFER_FLAG,
    DEFAULT_TRAIN_FLAG,
    Dashboard,
    EpisodeStats,
    EvalMetrics,
    Maze,
    MetricsTracker,
    MouseAgent,
    TrainConfig,
    _chart_x_max,
    _dashboard_tooltip_at,
    _episode_tick_values,
    _eval_greedy,
    _format_chart_tick,
    _optional_bool,
    _rects_overlap,
    _train_config_from_args,
    dashboard_layout,
    parse_args,
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


def test_full_observation_channels():
    env = Maze(fixed_grid(), observation_mode="full", max_episode_steps=10)

    obs = env.reset()

    assert obs.shape == (3, 5, 5)
    assert obs[0].sum() == np.count_nonzero(env.grid == 1)
    assert obs[1, env.start[0], env.start[1]] == 1.0
    assert obs[1].sum() == 1.0
    assert obs[2, env.goal[0], env.goal[1]] == 1.0
    assert obs[2].sum() == 1.0


def test_local_observation_keeps_agent_centered_and_goal_visible():
    env = Maze(
        fixed_grid(),
        observation_mode="local",
        view_size=5,
        max_episode_steps=10,
    )

    obs = env.reset()

    assert obs.shape == (3, 5, 5)
    assert obs[1, 2, 2] == 1.0
    assert obs[1].sum() == 1.0
    assert obs[2, 2, 4] == 1.0


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


def test_agent_masks_invalid_greedy_and_random_actions():
    config = TrainConfig(
        maze_size=(5, 5),
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


def test_cli_args_override_top_level_train_config_defaults():
    args = parse_args(
        [
            "--episodes",
            "9",
            "--maze-size",
            "7",
            "9",
            "--no-dashboard",
            "--eval-every",
            "3",
            "--eval-seed",
            "99",
            "--timeout-step-factor",
            "3.5",
            "--min-episode-steps",
            "7",
            "--learning-rate",
            "0.001",
            "--device",
            "cpu",
            "--no-infer",
        ]
    )
    config = _train_config_from_args(args)

    assert config.episodes == 9
    assert config.maze_size == (7, 9)
    assert config.dashboard_flag is False
    assert config.eval_every == 3
    assert config.dashboard_every == 3
    assert config.eval_seed == 99
    assert config.timeout_step_factor == 3.5
    assert config.min_episode_steps == 7
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


def test_agent_checkpoint_restores_optimizer_and_counters(tmp_path):
    save_path = tmp_path / "checkpoint.pth"
    config = TrainConfig(maze_size=(5, 5), save_path=str(save_path), device="cpu")
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
    assert resumed.optimizer.state_dict()["param_groups"] == (
        agent.optimizer.state_dict()["param_groups"]
    )


def test_train_automatically_resumes_existing_save_path(tmp_path, capsys):
    save_path = tmp_path / "resume_weights.pth"
    config = TrainConfig(
        maze_size=(5, 5),
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
            return np.zeros((3, 5, 5), dtype=np.float32)

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
                np.zeros((3, 5, 5), dtype=np.float32),
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
    assert records[0]["environment"]["selected_device"] == "cpu"
    assert "metrics" in eval_record
    assert "greedy" in eval_record
    assert "steps_per_second" in eval_record["speed"]
    assert "process_cpu_percent" in eval_record["utilization"]


def test_training_saves_new_best_weights_to_configured_path(tmp_path, capsys):
    save_path = tmp_path / "best_weights.pth"
    config = TrainConfig(
        maze_size=(5, 5),
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
