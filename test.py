import numpy as np
import math
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import os #for interacting with computers file system 
from PIL import Image #python imagaging library 
import pickle 
import matplotlib.pyplot as plt 

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
        # for i in x_file: 
        #     if i > .3 :
        #         i = 1
        #     else:
        #         pass 
        # img_array = x_file.reshape(28,28)
        rows = np.any(img_array > 0, axis=1)
        cols = np.any(img_array > 0, axis=0)
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        cropped = img_array[ymin:ymax+1, xmin:xmax+1]
        cropped_img = Image.fromarray(cropped)
        resized = cropped_img.resize((20,20), Image.LANCZOS)
        canvas = np.zeros((28,28))
        canvas[4:24, 4:24] = np.array(resized)
        outside_x.append(canvas.reshape(784))
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