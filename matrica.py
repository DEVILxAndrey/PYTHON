import numpy as np
import scipy

# I = np.array([0.9, 0.1, 0.8], ndmin=2).T
# t = np.array([0.2, 0.09, 0.5], ndmin=2).T
# W_i_h = np.array(([0.9, 0.3, 0.4], [0.2, 0.8, 0.2], [0.1, 0.5, 0.6]))
# W_h_o = np.array(([0.3, 0.7, 0.5], [0.6, 0.5, 0.2], [0.8, 0.1, 0.9]))
#
# for i in range(10000):
#     O_h = W_i_h.dot(I)
#     O_h = scipy.special.expit(O_h)
#     O = W_h_o.dot(O_h)
#     O = scipy.special.expit(O)
#
#     E_o = t - O
#     E_h = np.dot(W_h_o.T, E_o)
#     W_h_o += 0.1 * E_o * O * (1 - 0) * O_h.T
#     W_i_h += 0.1 * E_h * O_h * (1 - O_h) * I.T
#
# print(O)

class NeuralNetwork:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        self.learning_rate = learning_rate

        np.random.seed(42)
        self.W_i_h = np.random.normal(0.01, 0.99, (hidden_nodes, input_nodes))
        self.W_h_o = np.random.normal(0.01, 0.99, (hidden_nodes, output_nodes))
    def forward(self, inputs):
        inputs = np.array(inputs, ndmin=2).T
        O_h = self.W_i_h.dot(inputs)
        O_h = scipy.special.expit(O_h)
        O = self.W_h_o.dot(O_h)
        O = scipy.special.expit(O)
        return O

net = NeuralNetwork(3, 3, 3, 0.1)
result = net.forward([0.9, 0.1, 0.8])
print(result)