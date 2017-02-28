from graphviz import Digraph

class HMMPrinter(DotPrinter):
    def print_dot(self, filename, filetitle="HMM"):
        dot = Digraph(comment=filetitle)
        for i in range(self.model.m):
            dot.node("C{}".format(i), "C{}".format(i))
        for i in range(self.model.n):
            dot.node("H{}".format(i), "H{}".format(i))
            for j in range(self.model.n):
                dot.edge("H{}".format(i), "H{}".format(j), label="{}".format(round(self.model.A[i][j], 2)))
            for j in range(self.model.m):
                dot.edge("H{}".format(i), "C{}".format(j), label="{}".format(round(self.model.B[i][j], 2)), color="blue")
        dot.render(filename, view=False)
