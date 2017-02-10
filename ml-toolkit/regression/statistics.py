import math
import statistics

class Statistics:
    @staticmethod
    def combinaison(x, n):
        return math.factorial(n) / (math.factorial(x) * math.factorial(n - x)) 

    @staticmethod
    def permutation(x, n):
        return math.factorial(n) / math.factorial(n - x)

    @staticmethod
    def covariance(a, b, mean_a = None, mean_b = None):
        assert len(a) == len(b), "Size of each dataset must be equal"
        mean_a = statistics.mean(a) if not mean_a else mean_a
        mean_b = statistics.mean(b) if not mean_b else mean_b
        return sum([(ai - mean_a) * (bi - mean_b) for (ai, bi) in zip(a, b)]) / len(a)

    @staticmethod
    def pearson(a, b, mean_a = None, mean_b = None):
        mean_a = statistics.mean(a) if not mean_a else mean_a
        mean_b = statistics.mean(b) if not mean_b else mean_b
        return Statistics.covariance(a, b, mean_a, mean_b) / (statistics.pstdev(a, mu=mean_a) * statistics.pstdev(b, mu=mean_b))


    @staticmethod
    def spearman(a, b):
        n = len(a)
        assert n == len(b), "Size of each dataset must be equal"
        rank_a = [sorted(list(set(a))).index(elt) + 1 for elt in a]
        rank_b = [sorted(list(set(b))).index(elt) + 1 for elt in b]
        if max(rank_a) == len(rank_a) and max(rank_b) == len(rank_b):
            return 1 - (6 * sum([(ai - bi) ** 2 for (ai, bi) in zip(rank_a, rank_b)]) / (n ** 3 - n))
        return Statistics.pearson(rank_a, rank_b)

    @staticmethod
    def determination(y, computed_y):
        mean_y = statistics.mean(y)
        return sum([(y - mean_y) ** 2 for y in computed_y]) / sum([(y - mean_y) ** 2 for y in y]) * 100
