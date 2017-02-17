import copy
import random


def shuffle(array):
    """
        Redefines shuffling function so that it returns a copy of the generated value
    """
    arr = copy.copy(array)
    random.shuffle(arr)
    return arr

class RandomSelection:
    @staticmethod
    def reservoirSampling(population, nb_element):
        """
            Assume element_size < len(population)
        """
        result = [population[j] for j in range(nb_element)]
        for i in range(nb_element, len(population)):
            n = random.randint(0, i)
            if n < nb_element:
                result[n] = population[i]
        return result

    @staticmethod
    def shuffle(population, nb_element):
        """
            Assume element_size == len(population)
        """
        return [shuffle(population) for _ in range(nb_element)]
