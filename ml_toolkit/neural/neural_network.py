from .layer import Layer
from .activation import TanH
from ml_toolkit.cost import Quadratic

class NeuralNetwork(object):
    def __init__(self, layers, activation=TanH, loss=Quadratic, seed=None):
        self.loss = loss
        if (type(layers[0]) is int):
            self.layers = [Layer(layers[i], layers[i + 1], activation, seed) for i in range(len(layers) - 1)]
        elif (isinstance(type(layers[0]), Layer)):
            self.layers = layers
        else:
            raise TypeError("Argument 'layers' must be a list(int) or list(Layer)")

    def __call__(self, inputs):
        return self.predict(inputs)
    
    def sizes(self):
        return [self.layers[0].input_size] + [layer.output_size for layer in self.layers]

    def predict(self, inputs):
        results = inputs
        for layer in self.layers:
            results = layer.forward(results)
        return results
        
    def train(self, inputs, outputs, epochs=2500, learning_rate=0.1, momentum=0.1):
        for _ in range(epochs):
            global_error = 0
            for i in range(len(inputs)):
                inp = inputs[i]
                exp = outputs[i]
                pre = self.predict(inp)
                
                global_error += self.loss.loss(exp, pre)
                errors = self.loss.derivative(exp, pre)
                
                for layer in self.layers[::-1]:
                    errors = layer.backward(errors, learning_rate, momentum)
        return global_error