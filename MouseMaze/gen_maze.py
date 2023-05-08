import random
import matplotlib.pyplot as plt
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

def wilsons_algorithm(grid):
    m, n = grid.shape
    unvisited = set((r, c) for r in range(1, m, 2) for c in range(1, n, 2))
    current = random.choice(list(unvisited))
    unvisited.remove(current)

    while unvisited:
        r, c = current
        neighbors = valid_neighbors(m, n, r, c)
        next_r, next_c = random.choice(neighbors)
        if (next_r, next_c) in unvisited:
            grid[r + (next_r - r) // 2, c + (next_c - c) // 2] = 0
            unvisited.remove((next_r, next_c))
        current = (next_r, next_c)
    return grid

def draw_maze(grid, start, end):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(grid, cmap='gray_r', origin='upper')
    ax.scatter(start[1], start[0], c='green', s=100, marker='o', label='Start')
    ax.scatter(end[1], end[0], c='red', s=100, marker='o', label='End')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend()
    plt.savefig('maze.png')
    plt.show()

def generate_random_maze(n_rows, n_cols, visualize_maze_flag = False):
    m, n = n_rows, n_cols
    grid = create_grid(m, n)
    grid[1::2, 1::2] = 0
    maze = wilsons_algorithm(grid)

    start, end = (1, 1), (m-2, n-2)

    if visualize_maze_flag:
        draw_maze(maze, start, end)

    maze[start] = 2
    maze[end] = 3

    if visualize_maze_flag:
        print(maze)

    return maze