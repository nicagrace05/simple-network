import numpy as np 
import math
import os 
from PIL import Image 
import pickle
import matplotlib.pyplot as plt 
import tensorflow as tf
from tensorflow.keras.datasets import mnist

(xmnist, ymnist),(_,__) = mnist.load_data()


with open('weights.pkl', 'rb') as f:
     weights = pickle.load(f)
with open('biases.pkl', 'rb') as f:
      biases = pickle.load(f)

list_neurons = [784,625,324,10] 

class Lets_open_er_up :
    def __init__(self, X, weights, biases, file_name) :
        self.name = file_name 
        self.weights = weights
        self.biases = biases
        self.input = np.array(X).reshape(-1, 1)

    def layer1(self) :
        img_imp = self.input.reshape(28,28)
        plt.imshow(img_imp, cmap = 'gray')
        plt.title(f'input image: {self.name}')
        plt.show()
        lay1 = self.weights[0] @ self.input + self.biases[0]
        lay1_out = np.maximum(0, lay1)
        lay1_img = lay1_out.reshape(28,28)
        plt.imshow(lay1_img, cmap = 'gray')
        plt.title(f'first layer activations: {self.name}')
        plt.show()
        return lay1_out 

    def layer2(self, lay2_imp) :
        lay2 = self.weights[1] @ lay2_imp + self.biases[1]
        lay2_out = np.maximum(0, lay2)
        pixels2 = int(math.sqrt(list_neurons[1]))
        lay2_img = lay2_out.reshape(pixels2, pixels2)
        plt.imshow(lay2_img, cmap = 'gray')
        plt.title(f'second layer activations: {self.name}')
        plt.show()
        return lay2_out

    def layer3(self, lay3_imp) :
        lay3 = self.weights[2] @ lay3_imp + self.biases[2]
        lay3_out = np.maximum(0, lay3)
        pixels3 = int(math.sqrt(list_neurons[2]))
        lay3_img = lay3_out.reshape(pixels3, pixels3)
        plt.imshow(lay3_img, cmap = 'gray')
        plt.title(f'third layer activations: {self.name}')
        plt.show()
        return lay3_out

    def output_lay(self, out_lay_imp) :
        output = self.weights[-1] @ out_lay_imp + self.biases[-1]
        plt.imshow(output, cmap = 'gray')
        plt.show()
        exp_scores = np.exp(output - np.max(output, axis = 0, keepdims = True)) #softmax activation
        output_activation = exp_scores / np.sum(exp_scores, axis = 0 , keepdims = True)
        print(f'networks guess {np.argmax(output_activation)}')
        return output_activation
      
outside_x = []
outside_y = []
file_names = []
for filename in os.listdir('indipendent_digits/') :
    if filename.endswith('.png') :
        y_file = int(filename.split('_')[0])
        name_indipendent = filename.replace('.png','')
        full_path = os.path.join('indipendent_digits/',filename)
        img = Image.open(full_path).convert('L') #must convert to greyscale so shape is 28,28, no third dimension
        img = img.resize((28,28)) #in case too big 
        img_array = 1 - np.array(img) #inverting the pixels cause the mnnist data is white on black 
        x_file = (img_array / 255.0).reshape(784)
        file_names.append(name_indipendent)
        outside_x.append(x_file)
        outside_y.append(y_file)

# for numb, imp in enumerate(outside_x) :
#     testin = Lets_open_er_up(outside_x[numb], weights, biases, file_names[numb])
#     lay2imp = testin.layer1()
#     lay3imp = testin.layer2(lay2imp)
#     output_layimp = testin.layer3(lay3imp)
#     testin.output_lay(output_layimp)

# for i, weight in enumerate(weights[:-1]) :
#     print(weight.shape)
#     pixel = int(math.sqrt(len(weight)))
#     weight_row = weight[0].reshape(pixel, pixel)
#     plt.imshow(weight_row, cmap = 'gray')
#     plt.title(f'weight visualization layer {i}')
#     plt.show()

total = 0 
for i, imp in enumerate(xmnist[0:15]) :
     testin = Lets_open_er_up(imp, weights, biases, ymnist[i])
     lay2imp = testin.layer1()
     lay3imp = testin.layer2(lay2imp)
     output_layimp = testin.layer3(lay3imp)
     guess = testin.output_lay(output_layimp)
     if np.argmax(guess, axis=0) == ymnist[i] :
         total += 1 
print(f'percent on mnist data: {(total/15)*100}')

#seperating some data  
x_sep_ints = []
for intiger in range (0,9) :
    x_sep = []
    y_sep = []
    for times in range (0,9) :
        y_sep = ymnist.index(intiger)
        x_sep = [xmnist[ind] for ind in y_sep]
        x_sep_ints.append(x_sep)
