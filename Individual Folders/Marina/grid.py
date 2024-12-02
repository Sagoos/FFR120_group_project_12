import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

# Initialize grid size
GRID_SIZE = 20

# Initialize sugarscape grid
sugarscape = np.random.randint(1, 10, size=(GRID_SIZE, GRID_SIZE))  # Sugar levels (1-9)

# Number and radius of death zones
num_death_zones = 5
death_zone_radius = 3

# Number of agents
num_agents = 20

# Randomly initialize death zone centers and directions
death_zone_centers = [(random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1))
                      for _ in range(num_death_zones)]
death_zone_directions = [(random.choice([-1, 1]), random.choice([-1, 1]))
                         for _ in range(num_death_zones)]  # Movement directions (x, y)

# Initialize agents with random positions and vision
agents = [{'position': (random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)),
           'sugar': 0,
           'vision': random.randint(2, 5)}  # Vision radius (2-5 cells)
          for _ in range(num_agents)]

# Function to move death zones
def move_death_zones(centers, directions, grid_size):
    new_centers = []
    for i, (cx, cy) in enumerate(centers):
        dx, dy = directions[i]
        new_cx, new_cy = cx + dx, cy + dy

        # Bounce off walls
        if new_cx < 0 or new_cx >= grid_size:
            dx = -dx
            new_cx = cx + dx
        if new_cy < 0 or new_cy >= grid_size:
            dy = -dy
            new_cy = cy + dy

        directions[i] = (dx, dy)  # Update direction after bouncing
        new_centers.append((new_cx, new_cy))
    return new_centers, directions

# Function to add death zones to the grid
def add_death_zones(grid, centers, radius):
    grid_copy = grid.copy()  # Work on a copy to avoid overwriting the original grid
    for cx, cy in centers:
        for x in range(max(0, cx-radius), min(GRID_SIZE, cx+radius+1)):
            for y in range(max(0, cy-radius), min(GRID_SIZE, cy+radius+1)):
                if (x - cx)**2 + (y - cy)**2 <= radius**2:  # Circle condition
                    grid_copy[x, y] = -1  # Mark as a death zone
    return grid_copy

# Agent movement logic with vision
def move_agents(agents, grid, death_zones):
    for agent in agents[:]:  # Use a copy of the list for safe iteration
        x, y = agent['position']
        vision = agent['vision']

        # Look at all cells within the agent's vision range
        neighborhood = [
            (nx, ny)
            for nx in range(max(0, x-vision), min(GRID_SIZE, x+vision+1))
            for ny in range(max(0, y-vision), min(GRID_SIZE, y+vision+1))
        ]

        # Find the cell with the highest sugar level that isn't a death zone
        best_cell = max(
            neighborhood,
            key=lambda pos: grid[pos] if grid[pos] > 0 else -np.inf
        )

        # Move the agent to the best cell
        agent['position'] = best_cell

        # Check for death zones
        if grid[best_cell] == -1:
            agents.remove(agent)  # Agent dies
        else:
            # Collect sugar and reduce sugar level in the cell
            agent['sugar'] += grid[best_cell]
            grid[best_cell] = 0

# Simulation metrics
survival_count = []
total_sugar_collected = []

# Main simulation loop
steps = 50  # Number of steps to simulate
for step in range(steps):
    # Move death zones
    death_zone_centers, death_zone_directions = move_death_zones(death_zone_centers, death_zone_directions, GRID_SIZE)
    
    # Update grid with new death zones
    updated_grid = add_death_zones(sugarscape, death_zone_centers, death_zone_radius)
    
    # Move agents
    move_agents(agents, updated_grid, death_zone_centers)
    
    # Track metrics
    survival_count.append(len(agents))
    total_sugar_collected.append(sum(agent['sugar'] for agent in agents))

# Metrics analysis
final_survival_rate = (len(agents) / num_agents) * 100
final_total_sugar = sum(agent['sugar'] for agent in agents)
average_sugar_per_agent = final_total_sugar / len(agents) if agents else 0

# Plot survival over time
plt.figure(figsize=(8, 5))
plt.plot(range(steps), survival_count, label="Surviving Agents")
plt.title("Agent Survival Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Agents")
plt.legend()
plt.grid()
plt.show()

# Plot total sugar collected over time
plt.figure(figsize=(8, 5))
plt.plot(range(steps), total_sugar_collected, label="Total Sugar Collected")
plt.title("Total Sugar Collected Over Time")
plt.xlabel("Step")
plt.ylabel("Sugar Collected")
plt.legend()
plt.grid()
plt.show()

# Print final metrics
print("=== Final Metrics ===")
print(f"Initial Agents: {num_agents}")
print(f"Surviving Agents: {len(agents)} ({final_survival_rate:.2f}% survival rate)")
print(f"Total Sugar Collected: {final_total_sugar}")
print(f"Average Sugar per Surviving Agent: {average_sugar_per_agent:.2f}")
