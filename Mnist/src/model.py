import numpy as np

class DenseLayer: 
    def __init__(self, input_size, number_of_neurons):
        self.weights = 0.1 * np.random.randn(input_size, number_of_neurons)
        self.biases = np.zeros((1, number_of_neurons))
        self.velocity_weights = np.zeros_like(self.weights)
        self.velocity_biases = np.zeros_like(self.biases)

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues, inputs):
        self.dweights = np.dot(inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class ReLU: 
    def forward(self, x):
        self.inputs = x
        self.output = np.maximum(0, x)

    def backward(self, dvalues): 
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class Accuracy:
    def calculate(self, y_pred, y_true): 
        y_pred_class = np.argmax(y_pred, axis=1)
        if len(y_true.shape) == 2:
            y_true_class = np.argmax(y_true, axis=1)
        else:
            y_true_class = y_true
        return np.mean(y_pred_class == y_true_class)


class Loss: 
    def forward(self, y_pred, y_true):
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(len(y_pred)), y_true]
        else:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        return np.mean(-np.log(correct_confidences))


class Softmax:
    def forward(self, x):
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


class SoftmaxCrossEntropy:
    def __init__(self):
        self.activation = Softmax()
        self.loss = Loss()

    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.output = self.activation.output
        return self.loss.forward(self.output, y_true)

    def backward(self, y_true):
        self.dinputs = self.output.copy()
        if len(y_true.shape) == 1:
            self.dinputs[range(len(self.dinputs)), y_true] -= 1
        else:
            self.dinputs -= y_true
        self.dinputs /= len(self.dinputs)


class Optimizer:
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def update(self, layer):
        if self.decay:
            self.current_learning_rate = self.learning_rate / (1 + self.decay * self.iterations)

        layer.velocity_weights = self.momentum * layer.velocity_weights - self.current_learning_rate * layer.dweights
        layer.velocity_biases = self.momentum * layer.velocity_biases - self.current_learning_rate * layer.dbiases

        layer.weights += layer.velocity_weights
        layer.biases += layer.velocity_biases

        self.iterations += 1
