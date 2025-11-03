import numpy as np 
import os 
from PIL import Image 
import pickle
import matplotlib.pyplot as plt 

with open('weights_halloween.pkl', 'rb') as f:
     weights = pickle.load(f)
with open('biases_halloween.pkl', 'rb') as f:
      biases = pickle.load(f)
print(len(weights))

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
        lay2_img = lay2_out.reshape(28,28)
        plt.imshow(lay2_img, cmap = 'gray')
        plt.title(f'second layer activations: {self.name}')
        plt.show()
        return lay2_out

    def layer3(self, lay3_imp) :
        lay3 = self.weights[2] @ lay3_imp + self.biases[2]
        lay3_out = np.maximum(0, lay3)
        lay3_img = lay3_out.reshape(28,28)
        plt.imshow(lay3_img, cmap = 'gray')
        plt.title(f'third layer activations: {self.name}')
        plt.show()
        return lay3_out

    def output_lay(self, out_lay_imp) :
       output = self.weights[-1] @ out_lay_imp + self.biases[-1]
       output_activation = 1 / (1 + np.exp(-output))
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

for numb, imp in enumerate(outside_x) :
    testin = Lets_open_er_up(outside_x[numb], weights, biases, file_names[numb])
    lay2imp = testin.layer1()
    lay3imp = testin.layer2(lay2imp)
    output_layimp = testin.layer3(lay3imp)
    testin.output_lay(output_layimp)
