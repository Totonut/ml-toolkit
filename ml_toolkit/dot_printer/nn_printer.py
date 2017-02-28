from graphviz import Digraph

class NNPrinter(DotPrinter):
    def print_dot(self, filename, filetitle="NN"):
        dot = Digraph(comment=filetitle)
        previous_size = len(self.model.layers[0].weights[0]) - 1
        for i in range(len(self.model.layers[0].weights[0]) - 1):
            dot.node("I_{}".format(i), "I_{}".format(i))
        for i in range(len(self.model.layers)):
            for j in range(len(self.model.layers[i].weights)):
                dot.node("N_{}{}".format(i, j), "N_{}{}".format(i, j))
                for k in range(previous_size):
                    if (i == 0):
                        dot.edge("I_{}".format(k), "N_{}{}".format(i, j), label="{}".format(round(self.model.layers[i].weights[j][k], 2)))
                    else:
                        dot.edge("N_{}{}".format(i - 1, k), "N_{}{}".format(i, j), label="{}".format(round(self.model.layers[i].weights[j][k], 2)))
            previous_size = len(self.model.layers[i].weights[0]) - 1
        dot.render(filename, view=False)
