from graphviz import Digraph
from .layer import Layer
from .activation import TanH
from .distance import distest

class NeuralNetwork(object):
    def __init__(self, layers, activation=TanH, loss=distest(), seed=None):
        self.loss = loss
        if (type(layers[0]) is int):
            self.layers = [Layer(layers[i], layers[i + 1], activation, seed) for i in range(len(layers) - 1)]
        elif (issubclass(type(layers[0]), Layer)):
            self.layers = layers
        else:
            raise TypeError("Argument 'layers' must be a list(int) or list(Layer)")

    def __call__(self, inputs):
        return self.predict(inputs)

    def print_dot(self, filename, filetitle="NN"):
        dot = Digraph(comment=filetitle)
        previous_size = len(self.layers[0].weights[0]) - 1
        for i in range(len(self.layers[0].weights[0]) - 1):
            dot.node("I_{}".format(i), "I_{}".format(i))
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i].weights)):
                dot.node("N_{}{}".format(i, j), "N_{}{}".format(i, j))
                for k in range(previous_size):
                    if (i == 0):
                        dot.edge("I_{}".format(k), "N_{}{}".format(i, j), label="{}".format(round(self.layers[i].weights[j][k], 2)))
                    else:
                        dot.edge("N_{}{}".format(i - 1, k), "N_{}{}".format(i, j), label="{}".format(round(self.layers[i].weights[j][k], 2)))
            previous_size = len(self.layers[i].weights[0]) - 1
        dot.render(filename, view=False)

    def predict(self, inputs):
        results = inputs
        for layer in self.layers:
            results = layer.forward(results)
        return results
        
    def train(self, inputs, outputs, epochs=2500, learning_rate=0.1, momentum=0.1):
        for _ in range(epochs):
            for i in range(len(inputs)):
                inp = inputs[i]
                exp = outputs[i]
                errors = self.loss(self.predict(inp), exp)
                for layer in self.layers[::-1]:
                    errors = layer.backward(errors, learning_rate, momentum)
