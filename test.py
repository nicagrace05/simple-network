import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from PIL import Image #python imagaging library 
import matplotlib.pyplot as plt 
import os

(images_see, _ ),(test, test_) = mnist.load_data()

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

for i in range(0,15) :
    img = outside_x[i].reshape(28, 28)
    print(i)
    plt.imshow(img)
    plt.show()

for i in range(0,15) :
    img = images_see[i]
    print(i)
    plt.imshow(img)
    plt.show()