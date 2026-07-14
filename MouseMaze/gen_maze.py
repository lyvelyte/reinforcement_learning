import math
import random

import numpy as np


def create_grid(m, n):
    return np.ones((m, n), dtype=np.uint8)


def valid_neighbors(m, n, r, c):
    neighbors = []
    for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
        nr, nc = r + dr, c + dc
        if 0 < nr < m and 0 < nc < n:
            neighbors.append((nr, nc))
    return neighbors


def wilsons_algorithm(grid, rng=None):
    """Carve a uniform spanning tree with Wilson's loop-erased random walks."""

    random_source = rng if rng is not None else random
    m, n = grid.shape
    cells = [(r, c) for r in range(1, m, 2) for c in range(1, n, 2)]
    unvisited = set(cells)
    root = random_source.choice(cells)
    unvisited.remove(root)

    while unvisited:
        current = random_source.choice(sorted(unvisited))
        walk = [current]
        positions = {current: 0}

        while current in unvisited:
            next_cell = random_source.choice(valid_neighbors(m, n, *current))
            loop_start = positions.get(next_cell)
            if loop_start is not None:
                for removed in walk[loop_start + 1 :]:
                    positions.pop(removed, None)
                del walk[loop_start + 1 :]
            else:
                positions[next_cell] = len(walk)
                walk.append(next_cell)
            current = next_cell

        for source, destination in zip(walk, walk[1:]):
            wall = (
                source[0] + (destination[0] - source[0]) // 2,
                source[1] + (destination[1] - source[1]) // 2,
            )
            grid[wall] = 0
            unvisited.discard(source)
    return grid


def draw_maze(grid, start, end):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap="gray_r", origin="upper")
    ax.scatter(start[1], start[0], c="green", s=100, marker="o", label="Start")
    ax.scatter(end[1], end[0], c="red", s=100, marker="o", label="End")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.savefig("maze.png")
    plt.show()


def generate_random_maze(n_rows, n_cols, visualize_maze_flag=False, rng=None):
    random_source = rng if rng is not None else random
    m, n = n_rows, n_cols
    grid = create_grid(m, n)
    grid[1::2, 1::2] = 0
    maze = wilsons_algorithm(grid, rng=random_source)

    # Pick all open (non-wall) corridor cells.
    open_cells = [(r, c) for r in range(m) for c in range(n) if grid[r, c] == 0]
    start = random_source.choice(open_cells)

    min_manhattan = math.ceil(max(m, n) / 2)
    far_enough = [
        cell
        for cell in open_cells
        if abs(cell[0] - start[0]) + abs(cell[1] - start[1]) >= min_manhattan
    ]
    if far_enough:
        end = random_source.choice(far_enough)
    else:
        # Fallback: pick the single farthest cell.
        end_cell = max(
            open_cells,
            key=lambda cell: abs(cell[0] - start[0]) + abs(cell[1] - start[1]),
        )
        end = end_cell

    if visualize_maze_flag:
        draw_maze(maze, start, end)

    maze[start] = 2
    maze[end] = 3

    if visualize_maze_flag:
        print(maze)

    return maze
