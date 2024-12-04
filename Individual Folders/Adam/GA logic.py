from random import sample
import numpy as np
import pandas as pd

# Number of agents
num_agents = 400

# Create a DataFrame with random values for each attribute
agents = pd.DataFrame({
    "sugar": np.random.uniform(0, 1, num_agents),       # Random sugar values
    "metabolism": np.random.uniform(0, 1, num_agents), # Random metabolism values
    "vision": np.random.uniform(0, 1, num_agents),     # Random vision values
    "fitness": np.zeros(num_agents)                    # Initialize fitness to 0
})

# Display the first 5 agents
print(agents.head())


def calculate_fitness(agent):
    # A simple fitness function
    return agent['sugar'] * agent['metabolism'] * agent['vision']

def select_parents(agents, num_parents):
    parents = []
    for _ in range(num_parents):
        selected = agents.sample(3)  # Select 3 random agents for the tournament
        parent = selected.loc[selected['fitness'].idxmax()]  # Select the best one
        parents.append(parent)
    return pd.DataFrame(parents)

def crossover(parent1, parent2):
    crossover_point = np.random.randint(0, 3)  # Randomly choose a point (since there are 3 traits)
    
    child1 = parent1.copy()
    child2 = parent2.copy()

    # Swap the genes after the crossover point
    if crossover_point == 0:
        child1['sugar'], child2['sugar'] = parent2['sugar'], parent1['sugar']
    elif crossover_point == 1:
        child1['metabolism'], child2['metabolism'] = parent2['metabolism'], parent1['metabolism']
    elif crossover_point == 2:
        child1['vision'], child2['vision'] = parent2['vision'], parent1['vision']
    
    return child1, child2


def mutate(agent, mutation_rate=0.01):
    if np.random.rand() < mutation_rate:
        # Randomly choose one attribute to mutate
        attribute_to_mutate = np.random.choice(['sugar', 'metabolism', 'vision'])
        agent[attribute_to_mutate] = np.random.uniform(0, 1)
    return agent


def replace_population(agents, new_agents):
    return pd.concat([agents, new_agents]).reset_index(drop=True)



SuSca = np.array([
    [1, 1, 2, 2, 1],
    [1, 2, 3, 3, 2],
    [2, 3, 4, 4, 3],
    [2, 3, 4, 4, 3],
    [1, 2, 3, 3, 2]
])
visible_positions = [(0, 0), (1, 2), (3, 4)]

best_cell = max(
           visible_positions,
           key=lambda pos: SuSca[pos[0], pos[1]]  #indexing wrong here? , I want to compare positions but this indexes intp an array, and then the elements of the array. Resulting in a singel element. 
       )
print(best_cell)