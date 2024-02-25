import numpy as np
import math

# overall structure for any neural network layer
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    # takes in input, changes it based on type of layer (ex. Dense layer will apply weights and biases)
    def forward(self, input):
        pass
    # takes in previous gradient to continue back propagation
    def backward(self, output_gradient, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size):
        # randomize parameters
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)
    def forward(self, input):
        # Y = WX + B
        self.input = input
        return np.dot(self.weights, self.input) + self.biases
    def backward(self, output_gradient, learning_rate):
         # all gradients come from dE/dY, or error with respect to output
         # breakig down dE/dY into dE/dY_1 + dE/dY_2 + dE/dY_3 + ... and finding individual derivatives (ex. dE/dW_21) can lead back to these generalized gradients
         next_output_gradient = np.dot(self.weights.T, output_gradient)
         weights_gradient = np.dot(output_gradient, self.input.T)
         self.weights -= learning_rate * weights_gradient
         biases_gradient = output_gradient
         self.biases -= learning_rate * biases_gradient
         # backprop method in each layer takes the previous output gradient as input, so we return it
         return next_output_gradient
    def to_string(self):
        return "dense"
    def get_gradients(self):
        return np.concatenate([self.weights.flatten(), self.biases.flatten()])
    
class Activation(Layer):
    def __init__(self, nonlinear, nonlinear_prime):
        # nonlinear function is defined in sub-class; this class is essentially abstract - a structure
        self.nonlinear = nonlinear
        self.nonlinear_prime = nonlinear_prime
    def forward(self, input):
        self.input = input
        # running non-linear function on input
        return self.nonlinear(self.input)
    def backward(self, output_gradient, learning_rate):
        # derivative of error with respect to non-linear function can be found similar to gradients in Dense class
        return np.multiply(output_gradient, self.nonlinear_prime(self.input))
    
class ReLU(Activation):
    def __init__(self):
        relu = lambda x: np.maximum(0, x)
        def relu_prime(x):
            slopes = []
            for value in x:
                if value <= 0:
                    slopes.append(0)
                else:
                    slopes.append(1)
            return np.array(slopes).reshape(len(slopes), 1)
        super().__init__(relu, relu_prime)
    def to_string(self):
        return "relu"

class Tanh(Layer):
    def __init__(self):
        pass
    def forward(self, input):
        self.input = input
        return np.tanh(input)
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, 1 - np.tanh(self.input) ** 2)
    def to_string(self):
        return "tanh"

"""class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        # known derivative of tanh
        tanh_prime = lambda x: 1 - np.tanh(x) ** 2
        # sending parameters to parent class
        super().__init__(tanh, tanh_prime)
    def to_string(self):
        return "tanh"
"""

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output
    def backward(self, output_gradient, learning_rate):
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
    def to_string(self):
        return "softmax"
    
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)
    def to_string(self):
        return "sigmoid"
    def get_gradients(self):
        return np.array([])

class Recurrent(Layer):
    def __init__(self, input_size, hidden_size, activation):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.activation = activation

        # initialize parameters
        self.hidden_weights = np.random.randn(hidden_size, hidden_size) * 0.01
        self.input_weights = np.random.randn(hidden_size, input_size) * 0.01
        self.biases = np.random.randn(hidden_size, 1)

        # hidden state that is passed to the next layer
        self.hidden_state = np.random.randn(hidden_size, 1)
    def forward(self, input):
        self.input = input
        # storing previous hidden state (temp variable, maybe not necessary)
        prev_hidden_state = self.hidden_state
        # resulting vector from previous hidden state
        hidden_vector = np.dot(self.hidden_weights, prev_hidden_state)
        # resulting vector from input
        input_vector = np.dot(self.input_weights, input)
        # summing two vectors and biases (operation to give next hidden state)
        self.hidden_state = hidden_vector + input_vector + self.biases
        return self.activation.forward(self.hidden_state)
    def backward(self, output_gradient, learning_rate):
        # finding all gradients
        output_gradient = self.activation.backward(output_gradient, learning_rate)
        next_output_gradient = np.dot(self.hidden_weights.T, output_gradient)
        hidden_weights_gradient = np.dot(output_gradient, self.hidden_state.T)
        input_weights_gradient = np.dot(output_gradient, self.input.T)
        biases_gradient = output_gradient
        # updating parameters according to gradients
        self.hidden_weights -= learning_rate * hidden_weights_gradient
        self.input_weights -= learning_rate * input_weights_gradient
        self.biases -= learning_rate * biases_gradient
        # for activation layer and previous hidden layer to continue backprop
        return next_output_gradient
    def get_gradients(self):
        return np.concatenate([self.hidden_weights.flatten(), self.input_weights.flatten(), self.biases.flatten()])
    def to_string(self):
        return "recurrent"
    
class LSTM(Layer):
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # initialize parameters
        self.forget_weights = np.random.randn(hidden_size, input_size)
        # potential long term memory
        self.gate_weights = np.random.randn(hidden_size, input_size)
        # percent of potential long term memory to remember
        self.input_weights = np.random.randn(hidden_size, input_size)
        # for new short term memory
        self.output_weights = np.random.randn(hidden_size, input_size)
        # short term memory state
        self.short_term_memory = np.random.randn(hidden_size, 1)
        # long term memory state
    
    def lstm_cell(self, input):
        # activated forget gate result
        forget_activated = Sigmoid.forward(np.dot(input, self.forget_weights))
        # activated input gate result
        input_activated = Sigmoid.forward(np.dot(input, self.input_weights))
        # activated output gate result
        output_activated = Sigmoid.forward(np.dot(input, self.output_weights))
        # activated gate gate results
        gate_activated = Tanh.forward(np.dot(input, self.gate_weights))
        long_term_memory
        