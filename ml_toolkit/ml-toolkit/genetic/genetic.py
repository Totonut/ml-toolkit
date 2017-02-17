import random

class MovingPopulation:
    def __init__(self, population, fitness_function=lambda x : x, first_selection_function=lambda x : x, second_selection_function=lambda old, new, fit, nb : new, hybridation_function=lambda x : x, mutation_function=lambda x : x):
        self.population = population
        self.first_selection = first_selection_function
        self.second_selection = second_selection_function
        self.fitness = fitness_function
        self.hybridation = hybridation_function
        self.mutation = mutation_function

    def grow(self, nb_generation, nb_surviving=0):
        for _ in range(nb_generation):
            self.population = self.second_selection(
                self.population,
                self.mutation(
                    self.hybridation(
                        self.first_selection(self.population, self.fitness, len(self.population))
                    )
                ),
                self.fitness,
                nb_surviving
            )

