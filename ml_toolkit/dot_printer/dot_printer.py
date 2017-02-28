import abc

class DotPrinter:
    """
    Dot printer using graphviz module
    """

    def __init__(self, model):
        self.model = model

    @abc.abstractmethod
    def print_dot(self, filename, filetitle="NN"):
        pass
