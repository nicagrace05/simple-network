import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os #for interacting with computers file system 
from PIL import Image #python imagaging library 
import pickle 
import matplotlib.pyplot as plt 
np.random.seed(0)

(xtrain, ytrain), (xtest, ytest) = mnist.load_data() #training data already arranged (vector, laybel)
xtrain = (xtrain/255).reshape(xtrain.shape[0], -1)  #normalizing pixel values
xtest = (xtest/255).reshape(xtest.shape[0], -1)  
all_images = np.concatenate([xtrain, xtest], axis = 0 ) #training on all the data cuz im making my own test stuff 
all_labels = np.concatenate([ytrain, ytest], axis = 0 )

#structure 
epochs = 6 
list_neurons = [784,324,100,10] #maybe we can plot through as it iterates through the layers?
batch_len = 50
learning_rate = .005
batches_train = len(all_images) // batch_len

#THE MACHINE!!! coded line by line by me :)
class Neurons:
    def __init__(self, xtrain, ytrain, list_neurons, n_output_neurons):
        self.n_inputs = xtrain.shape[1]
        self.input = np.array(xtrain)
        self.onehot = np.array(ytrain)
        self.list_neurons = list_neurons
        self.n_output_neurons = n_output_neurons  
        self.batchlen = batch_len #batch_len is varible for inside the class 
              
    def initialize_wnb(self):
        self.weights = []  #storing all the weights for the network
        self.biases = []
        prev_layer_size = self.n_inputs
        for n_neurons in self.list_neurons:
            weights_layer = np.random.randn(n_neurons, prev_layer_size) * np.sqrt(2.0 / prev_layer_size) #He intialization for ReLU activation
            bias_layer = np.zeros((n_neurons, 1)) + 0.01
            self.weights.append(weights_layer)
            self.biases.append(bias_layer) 
            prev_layer_size = n_neurons
        print(f'Initialized {len(self.weights)} weights matrices')

    def foreward_hidden(self, xbatch):
        self.activations = [xbatch.T]  #storing for each layer, input vectors turn into rows of matrix sucsessivley itterated through layers 
        current_activation = xbatch.T
        for i in range(len(self.list_neurons) - 1):  
            z = self.weights[i] @ current_activation + self.biases[i]
            current_activation = np.maximum(0, z)  #ReLU activation
            self.activations.append(current_activation)
        return current_activation
        
    def foreward_output(self, hidden_activation):
        output_weights = self.weights[-1]
        output_bias = self.biases[-1]
        z_output = output_weights @ hidden_activation + output_bias
        exp_scores = np.exp(z_output - np.max(z_output, axis = 0, keepdims = True)) #using softmax activation so the predtictions must sum to one, better for classification, still elegent gradient funtion
        output_activation = exp_scores / np.sum(exp_scores, axis = 0 , keepdims = True) #subtracting the max to keep numbers manageble: preventing inf values 
        self.activations.append(output_activation)
        return output_activation

    def backpropagation(self, xbatch, ybatch, learning_rate):
        output_activation = self.activations[-1]
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
        loss = -np.sum(y_onehot * np.log(output_activation + 1e-8)) / self.batchlen #adding a tiny safe value to prevent log from approaching - infinty
        return loss 

    def predict(self, X): #X in array  
        a = X.T
        for i in range(len(self.list_neurons) - 1): #through each layer 
            z = self.weights[i] @ a + self.biases[i]
            a = np.maximum(0, z)
        z = self.weights[-1] @ a + self.biases[-1]
        exp_scores = np.exp(z - np.max(z, axis = 0, keepdims = True))
        out = exp_scores / np.sum(exp_scores, axis = 0, keepdims = True)
        preds = np.argmax(out, axis=0) # returns the label
        return preds

#training 
network =  Neurons(all_images, all_labels, list_neurons, 10) #creates item 
network.initialize_wnb()
loss_iterating = []

for epoch in range(epochs) : #using epochs lets the machine see the training data more than once!
    for b in range(batches_train) :
        xbatch = all_images[(b*batch_len):(b*batch_len + batch_len)] 
        ybatch = all_labels[(b*batch_len):(b*batch_len + batch_len)] 
        hid_out = network.foreward_hidden(xbatch)
        output = network.foreward_output(hid_out)
        loss_iterating.append(round(network.backpropagation(xbatch, ybatch, learning_rate),2))
    print(f'loss after epoch {epoch}: {loss_iterating[-1]}')

with open('weights_halloween.pkl', 'wb') as f: #numpy being stubborn  :(
    pickle.dump(network.weights, f)
with open('biases_halloween.pkl', 'wb') as f:
    pickle.dump(network.biases, f)
print('wnb saved as .pkl files!!')

#outside data
outside_x = []
outside_y = []
for filename in os.listdir('digits_outside_raw/') :
    if filename.endswith('.png') :
        y_file = int(filename.split('_')[0])
        full_path = os.path.join('digits_outside_raw/',filename)
        img = Image.open(full_path).convert('L') #must convert to greyscale so shape is 28,28, no third dimension
        img = img.resize((28,28)) #in case too big 
        img_array = 1 - np.array(img) #inverting the pixels cause the mnnist data is white on black 
        x_file = (img_array / 255.0).reshape(784)
        outside_x.append(x_file)
        outside_y.append(y_file)

tot = 0
outside_x = np.array(outside_x)
print(np.array(outside_x).shape)
guesses = network.predict(outside_x)
for i, value in enumerate(outside_y) : 
    if guesses[i] == outside_y[i]: 
        tot += 1 
    else:
        pass
        print(f' ruh roh! number: {value}, guess: {guesses[i]}')
        out_x_img = outside_x[i].reshape(28, 28)
        plt.imshow(out_x_img, cmap = 'gray')
        plt.show()
right = (tot/15)*100
print(f'{right:.2f} percent correct on your hard handwritten test!')
