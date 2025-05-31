import numpy as np
import matplotlib.pyplot as plt
import heapq
from math import sqrt

def read_grid_from_csv(filename):
    """Read grid from CSV file and extract start/goal positions"""
    grid = np.genfromtxt(filename, delimiter=',', dtype=int)
    
    # Find start (value 6) and goal (value 5) positions
    start_pos = np.where(grid == 6)
    goal_pos = np.where(grid == 5)
    
    if len(start_pos[0]) == 0:
        raise ValueError("Start position (value 6) not found in grid")
    if len(goal_pos[0]) == 0:
        raise ValueError("Goal position (value 5) not found in grid")
    
    start = (start_pos[0][0], start_pos[1][0])
    goal = (goal_pos[0][0], goal_pos[1][0])
    
    return grid, start, goal

def get_cost(cell_value):
    """Map cell values to traversal costs for Section 2 Q2"""
    cost_map = {
        0: 1,              # Light green - Open space
        1: float('inf'),   # Dark blue - Obstacle (impassable)
        2: 1,              # Light blue/cyan - Normal terrain
        3: 3,              # Brown/orange - High-cost terrain
        5: 1,              # Red - Goal
        6: 1               # Black - Start
    }
    return cost_map.get(cell_value, float('inf'))

def heuristic(pos1, pos2):
    """Euclidean distance heuristic"""
    return sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

def get_neighbors(pos, grid_shape):
    """Get valid 8-connected neighbors"""
    row, col = pos
    rows, cols = grid_shape
    neighbors = []
    
    # 8-directional movement
    directions = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for dr, dc in directions:
        new_row, new_col = row + dr, col + dc
        if 0 <= new_row < rows and 0 <= new_col < cols:
            neighbors.append((new_row, new_col))
    
    return neighbors

def movement_cost(pos1, pos2):
    """Calculate movement cost between adjacent cells"""
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
    
    # Track nodes explored for statistics
    nodes_explored = 0
    
    while open_set:
        # Get node with lowest f_score
        current_f, current_g, current = heapq.heappop(open_set)
        in_open_set.discard(current)
        nodes_explored += 1
        
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
            
            return path, total_cost, nodes_explored
        
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
    
    return None, float('inf'), nodes_explored  # No path found

def visualize_grid_and_path(grid, path, start, goal):
    """Visualize the grid with the path"""
    # Create a copy for visualization
    vis_grid = grid.copy().astype(float)
    
    # Create custom colormap for visualization
    # 0: light green, 1: dark blue, 2: light blue, 3: brown/orange, 5: red, 6: black
    from matplotlib.colors import ListedColormap, BoundaryNorm
    
    # Define colors for each value
    colors = ['lightgreen', 'darkblue', 'lightblue', 'brown', 'white', 'red', 'black']
    cmap = ListedColormap(colors)
    bounds = [0, 1, 2, 3, 4, 5, 6, 7]
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    
    # Display grid
    im = ax.imshow(vis_grid, cmap=cmap, norm=norm)
    
    # Draw path
    if path:
        path_rows = [pos[0] for pos in path]
        path_cols = [pos[1] for pos in path]
        ax.plot(path_cols, path_rows, 'yellow', linewidth=4, alpha=0.9, label='Optimal Path')
        ax.plot(path_cols, path_rows, 'orange', marker='o', markersize=3, alpha=0.8)
    
    # Mark start and goal with larger markers
    ax.plot(start[1], start[0], 'ko', markersize=15, markeredgecolor='white', 
            markeredgewidth=3, label=f'Start {start}')
    ax.plot(goal[1], goal[0], 'ro', markersize=15, markeredgecolor='white', 
            markeredgewidth=3, label=f'Goal {goal}')
    
    # Add grid lines for better visibility
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=0.8, alpha=0.4)
    
    # Add major grid lines every 10 cells
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 5), minor=False)
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 5), minor=False)
    ax.grid(which='major', color='white', linestyle='-', linewidth=1.2, alpha=0.6)
    
    # Add legend with cost information
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightgreen', label='Open Space (Cost: 1)'),
        plt.Rectangle((0,0),1,1, facecolor='darkblue', label='Obstacles (Impassable)'),
        plt.Rectangle((0,0),1,1, facecolor='lightblue', label='Normal Terrain (Cost: 1)'),
        plt.Rectangle((0,0),1,1, facecolor='brown', label='High-Cost Terrain (Cost: 3)'),
        plt.Rectangle((0,0),1,1, facecolor='red', label='Goal'),
        plt.Rectangle((0,0),1,1, facecolor='black', label='Start'),
    ]
    
    ax.set_title('Section 2 Question 2: A* Path Planning Result', fontsize=18, fontweight='bold')
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1))
    
    # Add path legend
    if path:
        ax.plot([], [], 'yellow', linewidth=4, label=f'Optimal Path (Length: {len(path)})')
        ax.legend(loc='upper left', bbox_to_anchor=(1.02, 0.7))
    
    plt.tight_layout()
    plt.show()

def print_detailed_results(grid, path, total_cost, nodes_explored, start, goal):
    """Print detailed analysis of the pathfinding results"""
    print("="*60)
    print("         SECTION 2 QUESTION 2 - PATHFINDING RESULTS")
    print("="*60)
    
    print(f"\nGrid Information:")
    print(f"  Grid Size: {grid.shape[0]} x {grid.shape[1]}")
    print(f"  Start Position: {start}")
    print(f"  Goal Position: {goal}")
    
    # Calculate Euclidean distance
    euclidean_distance = sqrt((goal[0] - start[0])**2 + (goal[1] - start[1])**2)
    print(f"  Straight-line Distance: {euclidean_distance:.2f}")
    
    if path:
        print(f"\nPath Found Successfully!")
        print(f"  Path Length: {len(path)} nodes")
        print(f"  Total Cost: {total_cost:.4f}")
        print(f"  Nodes Explored: {nodes_explored}")
        print(f"  Search Efficiency: {len(path)/nodes_explored:.2%}")
        print(f"  Path Efficiency: {euclidean_distance/total_cost:.2%}")
        
        # Analyze path composition
        terrain_counts = {0: 0, 1: 0, 2: 0, 3: 0, 5: 0, 6: 0}
        for pos in path:
            cell_value = grid[pos]
            terrain_counts[cell_value] += 1
        
        print(f"\nPath Composition:")
        terrain_names = {0: "Open Space", 1: "Obstacles", 2: "Normal Terrain", 
                        3: "High-Cost Terrain", 5: "Goal", 6: "Start"}
        for value, count in terrain_counts.items():
            if count > 0:
                print(f"  {terrain_names[value]}: {count} cells")
        
        print(f"\nFirst 15 Path Coordinates:")
        for i, pos in enumerate(path[:15]):
            cell_value = grid[pos]
            cell_cost = get_cost(cell_value)
            print(f"  Step {i:2d}: {pos} (Cell Value: {cell_value}, Cost: {cell_cost})")
        
        if len(path) > 15:
            print(f"  ... ({len(path)-15} more steps)")
            print(f"  Final: {path[-1]} (Goal)")
        
    else:
        print(f"\nNo Path Found!")
        print(f"  Nodes Explored: {nodes_explored}")
        print(f"  The goal may be unreachable from the start position.")

def main():
    """Main function to solve Section 2 Question 2"""
    try:
        # Read the grid - make sure S2Q2.csv is in your working directory
        filename = 'path planning/csv_files/S2Q2.csv'
        grid, start, goal = read_grid_from_csv(filename)
        
        print("Loading grid and running A* pathfinding algorithm...")
        print(f"Grid loaded successfully: {grid.shape[0]}x{grid.shape[1]}")
        
        # Find path using A*
        path, total_cost, nodes_explored = a_star(grid, start, goal)
        
        # Print detailed results
        print_detailed_results(grid, path, total_cost, nodes_explored, start, goal)
        
        # Visualize results
        print(f"\nGenerating visualization...")
        visualize_grid_and_path(grid, path, start, goal)
        
        # Additional analysis
        if path:
            print(f"\n" + "="*60)
            print("ADDITIONAL ANALYSIS")
            print("="*60)
            
            # Count terrain types in entire grid
            unique, counts = np.unique(grid, return_counts=True)
            terrain_stats = dict(zip(unique, counts))
            
            print(f"\nGrid Composition:")
            terrain_names = {0: "Open Space", 1: "Obstacles", 2: "Normal Terrain", 
                            3: "High-Cost Terrain", 5: "Goal", 6: "Start"}
            for value, count in terrain_stats.items():
                percentage = (count / grid.size) * 100
                print(f"  {terrain_names.get(value, f'Unknown ({value})')}: {count} cells ({percentage:.1f}%)")
        
    except FileNotFoundError:
        print("Error: S2Q2.csv file not found!")
        print("Make sure the CSV file is in the same directory as this script.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
