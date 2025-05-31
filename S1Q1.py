import numpy as np
import matplotlib.pyplot as plt
import heapq
from math import sqrt

def read_grid_from_csv(filename):
    """Read grid from CSV file and extract start/goal positions"""
    grid = np.genfromtxt(filename, delimiter=',', dtype=int)
    
    # Find start (value 2) and goal (value 3) positions
    start_pos = np.where(grid == 2)
    goal_pos = np.where(grid == 3)
    
    start = (start_pos[0][0], start_pos[1][0])
    goal = (goal_pos[0][0], goal_pos[1][0])
    
    return grid, start, goal

def get_cost(cell_value):
    """Map cell values to traversal costs"""
    cost_map = {
        0: 1,              # White - Open space
        1: float('inf'),   # Black - Obstacle
        2: 1,              # Green - Start point
        3: 1               # Red - Goal point
    }
    return cost_map.get(cell_value, float('inf'))

def heuristic(pos1, pos2):
    """Calculate Euclidean distance heuristic"""
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_neighbors(pos, grid_shape):
    """Get valid neighboring positions (8-directional movement)"""
    row, col = pos
    rows, cols = grid_shape
    neighbors = []
    
    # 8-directional movement (including diagonals)
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            neighbors.append((new_row, new_col))
    
    return neighbors

def movement_cost(pos1, pos2):
    """Calculate movement cost between adjacent cells"""
    # Diagonal movement costs sqrt(2), orthogonal movement costs 1
    row_diff = abs(pos1[0] - pos2[0])
    col_diff = abs(pos1[1] - pos2[1])
    
    if row_diff == 1 and col_diff == 1:
        return sqrt(2)  # Diagonal movement
    else:
        return 1        # Orthogonal movement

def a_star(grid, start, goal):
    """A* pathfinding algorithm"""
    rows, cols = grid.shape
    
    # Priority queue: (f_score, g_score, position)
    open_set = [(0, 0, start)]
    
    # Track the path
    came_from = {}
    
    # Cost from start to each node
    g_score = {start: 0}
    
    # Estimated total cost from start to goal through each node
    f_score = {start: heuristic(start, goal)}
    
    # Set of discovered nodes
    in_open_set = {start}
    
    while open_set:
        # Get node with lowest f_score
        current_f, current_g, current = heapq.heappop(open_set)
        in_open_set.discard(current)
        
        # Goal reached
        if current == goal:
            path = []
            total_cost = g_score[current]
            
            # Reconstruct path
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            
            return path, total_cost
        
        # Explore neighbors
        for neighbor in get_neighbors(current, (rows, cols)):
            # Skip if neighbor is an obstacle
            neighbor_cost = get_cost(grid[neighbor])
            if neighbor_cost == float('inf'):
                continue
            
            # Calculate tentative g_score
            move_cost = movement_cost(current, neighbor)
            tentative_g = g_score[current] + move_cost * neighbor_cost
            
            # If this path to neighbor is better than previous one
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                
                if neighbor not in in_open_set:
                    heapq.heappush(open_set, (f_score[neighbor], tentative_g, neighbor))
                    in_open_set.add(neighbor)
    
    return None, float('inf')  # No path found

def visualize_grid_and_path(grid, path, start, goal):
    """Visualize the grid with the path"""
    # Create a copy for visualization
    vis_grid = grid.copy().astype(float)
    
    # Color mapping for visualization
    # 0: white (open), 1: black (obstacle), 2: green (start), 3: red (goal)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    
    # Create custom colormap
    colors = ['white', 'black', 'green', 'red']
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(colors)
    
    # Display grid
    im = ax.imshow(vis_grid, cmap=cmap, vmin=0, vmax=3)
    
    # Draw path
    if path:
        path_rows = [pos[0] for pos in path]
        path_cols = [pos[1] for pos in path]
        ax.plot(path_cols, path_rows, 'b-', linewidth=3, alpha=0.7, label='Path')
        ax.plot(path_cols, path_rows, 'bo', markersize=4, alpha=0.7)
    
    # Mark start and goal
    ax.plot(start[1], start[0], 'go', markersize=15, label='Start')
    ax.plot(goal[1], goal[0], 'ro', markersize=15, label='Goal')
    
    # Grid lines
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
    
    ax.set_title('Grid with A* Path Planning Result', fontsize=16)
    ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    """Main function to solve the pathfinding problem"""
    # Read the grid
    filename = 'path planning/csv_files/S1Q1.csv'  # Make sure this file is in your working directory
    grid, start, goal = read_grid_from_csv(filename)
    
    print(f"Grid shape: {grid.shape}")
    print(f"Start position: {start}")
    print(f"Goal position: {goal}")
    
    # Find path using A*
    print("/nRunning A* algorithm...")
    path, total_cost = a_star(grid, start, goal)
    
    if path:
        print(f"/nPath found!")
        print(f"Path length: {len(path)} nodes")
        print(f"Total cost: {total_cost:.4f}")
        
        print(f"/nPath coordinates:")
        for i, pos in enumerate(path):
            print(f"Step {i}: ({pos[0]}, {pos[1]})")
        
        # Visualize results
        visualize_grid_and_path(grid, path, start, goal)
        
    else:
        print("No path found!")

if __name__ == "__main__":
    main()
