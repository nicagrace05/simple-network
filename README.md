# simple-network
welcome:
I want to make a simple handwritten digit model and run her on images with various tweaks like number size, background noise, pen darkness ect. Trying to create funtions to map the neural activations and see how these tweaked images affect the models confidance to gain insight on the way it idenifys digits.

done:
class neurons intitizales the weights and biases around zero before batching the mnist data in sets of 50 through 3 layers using ReLU activation. The ouput layer uses sigmoid, and then the backprop funtion batches the gradient back through the weights using chain rule and retuns the loss. [around 2 after single batch].
outside handrawn data in the repos in png format, drawn on sketchbook.io 28X28 with a black pensil of width 2 pixels.

the visulaization file uses the saved weights and iterates through 3 hidden layers, plotting the cross product of the wieghts and input vector as a 28x28 image before using relu acitvation. 

next steps : 
-debug the size errors with the outside images you drew
-debug visulization 

   