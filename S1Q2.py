import numpy as np
import matplotlib.pyplot as plt
import heapq
from math import sqrt

def read_grid_from_csv(filename):
    """Read grid from CSV file and extract start/goal positions"""
    grid = np.genfromtxt(filename, delimiter=',', dtype=int)
    start_pos = np.where(grid == 2)
    goal_pos = np.where(grid == 3)
    start = (start_pos[0][0], start_pos[1][0])
    goal = (goal_pos[0][0], goal_pos[1][0])
    return grid, start, goal

def get_cost(cell_value):
    """Map cell values to traversal costs for Section 2"""
    cost_map = {
        0: 1,              # White - Open space
        1: float('inf'),   # Black - Obstacle
        2: 1,              # Green - Start
        3: 1,              # Red - Goal
        4: 2               # Grey - High-cost terrain
    }
    return cost_map.get(cell_value, float('inf'))

def heuristic(pos1, pos2):
    """Use Euclidean distance as heuristic"""
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_neighbors(pos, grid_shape):
    """Get valid 8-connected neighbors"""
    row, col = pos
    rows, cols = grid_shape
    neighbors = []
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            neighbors.append((new_row, new_col))
    return neighbors

def movement_cost(pos1, pos2):
    """Diagonal movement costs sqrt(2), straight movement costs 1"""
    row_diff = abs(pos1[0] - pos2[0])
    col_diff = abs(pos1[1] - pos2[1])
    if row_diff == 1 and col_diff == 1:
        return sqrt(2)
    else:
        return 1

def a_star(grid, start, goal):
    rows, cols = grid.shape
    open_set = [(0, 0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    in_open_set = {start}
    while open_set:
        current_f, current_g, current = heapq.heappop(open_set)
        in_open_set.discard(current)
        if current == goal:
            path = []
            total_cost = g_score[current]
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, total_cost
        for neighbor in get_neighbors(current, (rows, cols)):
            neighbor_cost = get_cost(grid[neighbor])
            if neighbor_cost == float('inf'):
                continue
            move_cost = movement_cost(current, neighbor)
            tentative_g = g_score[current] + move_cost * neighbor_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                if neighbor not in in_open_set:
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
                    in_open_set.add(neighbor)
    return None, float('inf')

def visualize_grid_and_path(grid, path, start, goal):
    vis_grid = grid.copy().astype(float)
    # Custom colormap: 0-white, 1-black, 2-green, 3-red, 4-grey
    from matplotlib.colors import ListedColormap
    colors = ['white', 'black', 'green', 'red', 'grey']
    cmap = ListedColormap(colors)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(vis_grid, cmap=cmap, vmin=0, vmax=4)
    if path:
        path_rows = [pos[0] for pos in path]
        path_cols = [pos[1] for pos in path]
        ax.plot(path_cols, path_rows, 'b-', linewidth=2, alpha=0.7, label='Path')
        ax.plot(path_cols, path_rows, 'bo', markersize=3, alpha=0.7)
    ax.plot(start[1], start[0], 'go', markersize=12, label='Start')
    ax.plot(goal[1], goal[0], 'ro', markersize=12, label='Goal')
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    ax.set_title('Section 2 Problem 2: A* Path Planning', fontsize=16)
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    filename = 'path planning/csv_files/S1Q2.csv'  # Make sure this file is in your working directory
    grid, start, goal = read_grid_from_csv(filename)
    print(f"Grid shape: {grid.shape}")
    print(f"Start position: {start}")
    print(f"Goal position: {goal}")
    print("\nRunning A* algorithm...")
    path, total_cost = a_star(grid, start, goal)
    if path:
        print(f"\nPath found!")
        print(f"Path length: {len(path)} nodes")
        print(f"Total cost: {total_cost:.4f}")
        print(f"\nPath coordinates:")
        for i, pos in enumerate(path):
            print(f"Step {i}: ({pos[0]}, {pos[1]})")
        visualize_grid_and_path(grid, path, start, goal)
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
