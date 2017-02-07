import random

from genetic import MovingPopulation
from genetic import WeightedSelection
from genetic import RandomSelection

def hybridation(population):
    pairs = [(population[i], population[i + 1]) for i in range(0, len(population) - 1, 2)]
    new_population = []
    for p in pairs:
        n = random.randint(0, len(p[0]))
        new_population.append(p[0][:n] + p[1][n:])
        new_population.append(p[1][:n] + p[0][n:])
    return new_population


def mutation(population):
    for e in population:
        a = random.randint(0, len(e) - 1)
        b = random.randint(0, len(e) - 1)
        small = min(a, b)
        big = max(a, b)
        tmp = e[small]
        e[small] = e[big]
        e[big] = tmp
    return population

if __name__ == "__main__":
    print("=============== Genetic test - Population with max sum selection ===============")
    set_size = 1000
    population_size = 50
    element_size = 10
    dataset = list(range(set_size))
    random.shuffle(dataset)
    initial_population = [RandomSelection.reservoirSampling(dataset, element_size) for _ in range(population_size)]
    fitness = lambda x : sum(x)
    population = MovingPopulation(
        initial_population,
        fitness,
        first_selection_function=WeightedSelection.tournament,
        hybridation_function=hybridation,
        mutation_function=mutation
    )
    print("First generation:")
    print("\n".join([str(p) for p in population.population]))
    population.grow(100, 0)
    print("\n\n\n100th generation:")
    print("\n".join([str(p) for p in population.population]))
