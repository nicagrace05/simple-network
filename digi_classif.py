import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.datasets import mnist
np.random.seed(0)

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
xtrain = (xtrain/255).reshape(xtrain.shape[0], -1)  # normalizing pixel values

#structure 
list_neurons = [784,784,784,10] 
batch_len = 50
learning_rate = .015

xbatch = xtrain[0:batch_len]
ybatch = ytrain[0:batch_len]

class Neurons:
    def __init__(self, xtrain, ytrain, list_neurons, n_output_neurons):
        self.n_inputs = xtrain.shape[1]
        self.input = np.array(xtrain)
        self.onehot = np.array(ytrain)
        self.list_neurons = list_neurons
        self.n_output_neurons = n_output_neurons  
        self.batchlen = batch_len #batch_len is varible for inside the class 
              
    def initialize_wnb(self):
        self.weights = []  # storing all the weights for the network
        self.biases = []
        prev_layer_size = self.n_inputs
        for n_neurons in self.list_neurons:
            weights_layer = np.random.randn(n_neurons, prev_layer_size) * 0.01
            bias_layer = np.zeros((n_neurons, 1)) + 0.01
            self.weights.append(weights_layer)
            self.biases.append(bias_layer) 
            prev_layer_size = n_neurons
        print(f'Initialized {len(self.weights)} weight matrices.')

    def foreward_hidden(self, xbatch):
        self.activations = [xbatch.T]  # storing for each layer, input vectors turn into rows of matrix sucsessivley itterated through layers 
        current_activation = xbatch.T
        for i in range(len(self.list_neurons) - 1):  
            z = self.weights[i] @ current_activation + self.biases[i]
            current_activation = np.maximum(0, z)  # ReLU activation
            self.activations.append(current_activation)
        return current_activation
        
    def foreward_output(self, hidden_activation):
        output_weights = self.weights[-1]
        output_bias = self.biases[-1]
        z_output = output_weights @ hidden_activation + output_bias
        output_activation = 1 / (1 + np.exp(-z_output))  # sigmoid activation 
        self.activations.append(output_activation)
        return output_activation

    def backpropagation(self, xbatch, ybatch, learning_rate):
        hidden_final = self.foreward_hidden(xbatch)
        output_activation = self.foreward_output(hidden_final)
        y_onehot = np.eye(10)[ybatch].T  # shape (10, batch_len)
        output_error = output_activation - y_onehot  # output gradient
        grad_weights_out = (output_error @ self.activations[-2].T) / self.batchlen  # finding the average gradient over the whole batch 
        grad_bias_out = np.sum(output_error, axis=1, keepdims=True) / self.batchlen
        grad_weights = [None] * len(self.weights)
        grad_biases = [None] * len(self.biases)
        grad_weights[-1] = grad_weights_out
        grad_biases[-1] = grad_bias_out
        current_error = output_error
        for l in range(len(self.weights) - 2, -1, -1):  # iterating backwards through the layers 
            next_weights = self.weights[l + 1]
            error_before_activ = next_weights.T @ current_error
            z_current = self.weights[l] @ self.activations[l] + self.biases[l]
            relu_deriv = (z_current > 0).astype(float)
            current_error = error_before_activ * relu_deriv
            grad_weights[l] = (current_error @ self.activations[l].T) / self.batchlen
            grad_biases[l] = np.sum(current_error, axis=1, keepdims=True) / self.batchlen
        for l in range(len(self.weights)):
            self.weights[l] -= learning_rate * grad_weights[l]
            self.biases[l] -= learning_rate * grad_biases[l]
        loss = np.sum(output_error ** 2) / self.batchlen  # means squared error loss 
        return loss 

    def predict(self, x):
        # x: (n_samples, n_inputs)
        a = x.T
        for i in range(len(self.list_neurons) - 1): #through each layer 
            z = self.weights[i] @ a + self.biases[i]
            a = np.maximum(0, z)
        z = self.weights[-1] @ a + self.biases[-1]
        out = 1 / (1 + np.exp(-z))
        # return label
        preds = np.argmax(out, axis=0)
        return preds

batches_train = 50 
network =  Neurons(xtrain, ytrain, list_neurons, 10) #creates item 
network.initialize_wnb()
loss_iterating = []

for b in range(batches_train) : 
    xbatch = xtrain[(b*batch_len):(b*batch_len + batch_len)] 
    ybatch = ytrain[(b*batch_len):(b*batch_len + batch_len)] 
    hid_out = network.foreward_hidden(xbatch)
    output = network.foreward_output(hid_out)
    loss_iterating.append(backpropagation(xbatch, ybatch, learning_rate)) 
print(f'loss iderating through {batches_train} batches of mnist data: {loss_iterating}')

#Outside data
for filename in os.listdir(digits_outside_raw) :
    Image.open() #do i need to resize to 28 x 28?
    #np.array()

