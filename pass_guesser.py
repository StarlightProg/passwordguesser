import random
import numpy as np
from deap import base, creator, tools, algorithms

TARGET_PASSWORD = "tamamonomae"
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attribute", random.choice, ALPHABET)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=len(TARGET_PASSWORD))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual):
    matches = sum(1 for a, b in zip(individual, TARGET_PASSWORD) if a == b)
    return len(TARGET_PASSWORD) - matches,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)

def custom_mutate(individual, indpb):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = random.choice(ALPHABET)
    return individual,

toolbox.register("mutate", custom_mutate, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

POPULATION_SIZE = 100
N_GENERATIONS = 50
CXPB, MUTPB = 0.5, 0.2

def main():
    population = toolbox.population(n=POPULATION_SIZE)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", np.min)
    stats.register("avg", np.mean)

    population, log = algorithms.eaSimple(
        population, toolbox, cxpb=CXPB, mutpb=MUTPB, ngen=N_GENERATIONS,
        stats=stats, verbose=True
    )

    best_ind = tools.selBest(population, k=1)[0]
    print(f"Лучший индивид: {''.join(map(str, best_ind))} с приспособленностью: {best_ind.fitness.values}")

if __name__ == "__main__":
    main()
