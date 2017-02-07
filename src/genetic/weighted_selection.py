import random

class WeightedSelection:
    @staticmethod
    def elitism(old_population, new_population, fitness_function, nb_element, nb_surviving=0):
        """
            Assume nb_surviving <= nb_element
        """
        return sorted(old_population, key=fitness_function, reverse=True)[:nb_surviving] + sorted(new_population, key=fitness_function, reverse=True)[:nb_element - nb_surviving]


    @staticmethod
    def eugenism(population, fitness_function, nb_element):
        return sorted(population, key=fitness_function, reverse=True)[:nb_element]

    @staticmethod
    def wheel(population, fitness_function, nb_element):
        new_population = []
        s = sum([fitness_function(e) for e in population])
        for _ in range(nb_element):
            n = random.randint(0, s)
            cumul = 0
            i = 0
            while cumul < n:
                cumul += fitness_function(population[i])
                i += 1
            new_population.append(population[i - 1])
        return new_population

    @staticmethod
    def rank(population, fitness_function, nb_element):
        new_population = []
        tab = sorted(population, key=fitness_function)
        s = len(tab) * (len(tab) + 1) / 2
        for _ in range(nb_element):
            n = random.randint(1, s)
            cumul = 1
            i = 1
            while cumul < n:
                i += 1
                cumul += i
            new_population.append(tab[i - 1])
        return new_population

    @staticmethod
    def tournament(population, fitness_function, nb_element):
        assert nb_element == len(population), "tournament selection is best used to generate n children generation from a n-sized population"
        pairs = [(population[i], population[i + 1]) for i in range(-1, nb_element - 1)]
        return [p[0] if fitness_function(p[0]) > fitness_function(p[1]) else p[1] for p in pairs]
