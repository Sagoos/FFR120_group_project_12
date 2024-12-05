from random import sample
import numpy as np
import pandas as pd


def calculate_fitness(agent):
    # A simple fitness function
    # What do we want to have as fitness? 
    fitness =agent['sugar'] * agent['metabolism'] * agent['vision']
    return fitness

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
        agent[attribute_to_mutate] = np.random.randint(0, 6)
    return agent


def replace_population(agents, new_agents):
    return pd.concat([agents, new_agents]).reset_index(drop=True)


def remove_population(agents, num_agents):
    