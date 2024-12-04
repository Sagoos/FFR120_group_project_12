import numpy as np
import matplotlib.pyplot as plt
import random

# Define grid size and parameters
grid_size = 50
sugar_grid = np.zeros((grid_size, grid_size))
original_sugar_grid = np.zeros((grid_size, grid_size))  # To track original sugar levels
death_zone = np.zeros((grid_size, grid_size), dtype=bool)
regrowth_rate = 1  # Sugar regrowth rate per step

# Define agents as a list of dictionaries
agents = []

# Function to add variable-sized sugar hills
def add_sugar_hills(grid, num_hills):
    original_grid = np.zeros_like(grid)  # To store original sugar levels
    for _ in range(num_hills):
        hill_x, hill_y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        hill_height = random.randint(5, 15)  # Max sugar at the center
        hill_radius = random.randint(3, 10)  # Random radius
        for x in range(grid_size):
            for y in range(grid_size):
                distance = np.sqrt((x - hill_x)**2 + (y - hill_y)**2)
                if distance <= hill_radius:
                    sugar_value = max(hill_height - distance, 0)
                    grid[x, y] += sugar_value
                    original_grid[x, y] = min(grid[x, y], 10)  # Cap sugar at max 10
    return grid, original_grid

# Function to add variable-sized circular death zones
def add_circular_death_zones(grid, num_zones):
    for _ in range(num_zones):
        center_x, center_y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
        zone_radius = random.randint(3, 10)  # Random radius
        for x in range(grid_size):
            for y in range(grid_size):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                if distance <= zone_radius:
                    grid[x, y] = True  # Mark as part of the death zone
    return grid

# Function to add agents
def initialize_agents(num_agents):
    for _ in range(num_agents):
        while True:
            x, y = random.randint(0, grid_size - 1), random.randint(0, grid_size - 1)
            # Ensure agents don't start in a death zone
            if not death_zone[x, y]:
                sugar = random.randint(5, 15)  # Initial sugar level
                agents.append({"x": x, "y": y, "sugar": sugar})
                break

# Function to simulate agent movement and interactions
def simulate_step():
    global sugar_grid, original_sugar_grid, agents, death_zone

    new_agents = []  # To store surviving agents
    total_collected_sugar = 0

    for agent in agents:
        # Move agent randomly
        dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])  # Random direction
        new_x = (agent["x"] + dx) % grid_size  # Wrap-around movement
        new_y = (agent["y"] + dy) % grid_size

        # Check for danger zone
        if death_zone[new_x, new_y]:
            # Agent is "killed"
            continue

        # Collect sugar
        collected_sugar = sugar_grid[new_x, new_y]
        agent["sugar"] += collected_sugar
        sugar_grid[new_x, new_y] = 0  # Sugar consumed
        total_collected_sugar += collected_sugar

        # Update agent's position
        agent["x"], agent["y"] = new_x, new_y

        # Keep surviving agent
        new_agents.append(agent)

    # Update agent list
    agents[:] = new_agents

    # Regenerate sugar only if below the original level
    for x in range(grid_size):
        for y in range(grid_size):
            if sugar_grid[x, y] < original_sugar_grid[x, y]:  # Check if regeneration needed
                sugar_grid[x, y] += regrowth_rate
                sugar_grid[x, y] = min(sugar_grid[x, y], original_sugar_grid[x, y])  # Cap at original level

    return len(new_agents), total_collected_sugar

# Initialize the grid and agents
num_agents = 20
sugar_grid, original_sugar_grid = add_sugar_hills(sugar_grid, num_hills=10)
death_zone = add_circular_death_zones(death_zone, num_zones=5)
initialize_agents(num_agents)

# Simulation parameters
time_steps = 50

# Set up the figure for animation
plt.figure(figsize=(10, 10))
sugar_plot = plt.imshow(sugar_grid, cmap='YlGn', alpha=0.8, origin='upper')
plt.colorbar(label='Sugar Levels')

# Overlay death zones
for x in range(grid_size):
    for y in range(grid_size):
        if death_zone[x, y]:
            plt.scatter(y, x, color='red', s=5)

agent_scatter = plt.scatter([], [], color='blue', s=30)  # Placeholder for agents
plt.title('Sugarscape Simulation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Simulation loop
for step in range(time_steps):
    # Simulate one step
    num_survivors, sugar_collected = simulate_step()

    # Update sugar grid
    sugar_plot.set_data(sugar_grid)

    # Update agent positions
    agent_positions = np.array([[agent["y"], agent["x"]] for agent in agents])
    if len(agent_positions) > 0:  # Only update if agents exist
        agent_scatter.set_offsets(agent_positions)

    plt.title(f'Sugarscape - Step {step + 1} | Survivors: {num_survivors}')
    plt.pause(0.2)

# Final metrics, maybe we'll add some stuff idk
total_agents_killed = num_agents - len(agents)
print("Simulation Metrics:")
print(f"Initial Agents: {num_agents}")
print(f"Final Surviving Agents: {len(agents)}")
print(f"Total Agents Killed: {total_agents_killed}")
print(f"Sugar Remaining on Grid: {sugar_grid.sum()}")

plt.show()
