# simple-network
welcome:
I want to make a simple handwritten digit model and run her on images with various tweaks like number size, background noise, pen darkness ect. Trying to create funtions to map the neural activations and see how these tweaked images affect the models confidance to gain insight on the way it idenifys digits.

done:
class neurons intitizales the weights and biases around zero before batching the mnist data through the layers using ReLU activation. The ouput layer uses softmax, and then the backprop funtion batches the gradient back through the weights using chain rule and retuns the loss (around .18 after 6 epochs). 
weights and biases saved as .pkl files. 

outside handrawn data in the repository in png format, drawn on sketchbook.io 28X28 with a black pensil of width 2 pixels. the machine does not like my handwritten digits!! i did make them purposefully hard but its accuraccy was 26% as of 11/6 :(. I recon its overtrained 

the visulaization file uses the saved weights and iterates over the file indipendent_digits which are numbers i drew, tweaking various attributes meantioned above. it then plots the activated cross product each layer as an image, but i saw nothing but random looking geometrical patterns as of 11/6. 

comparing my handwritten numbers to the minst ones using the test file, i can see that their pen with might be slightly bigger, but i was suprised at how sloppy the data was. decided to train the model on the mnist train and testing data collections for hopes of smaller loss. the network keeps tripping up on the same digits so i might try redrawing the testing set, or implimenting a bit of the mnist again to pinpoint whats going on

next steps : 
-test weights and biases on handwritten images by me that very closley resemble mnist data to see if theres a formatting issue?? 
-visualize weights on mnist data to see if experimental method is sound

   