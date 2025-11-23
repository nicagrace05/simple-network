# simple-network
    welcome:
i am instersted in gaining further insight into the methods neural networks use to generalize, and what factors are important in an input to maximize accuracy. its proving a lot harder than i thought to manage errors surrounding slight differnences in referncial and created dataset.  

    done:
class neurons intitizales the weights and biases around zero before batching the mnist data through the layers using ReLU activation. The ouput layer uses softmax, and then the backprop funtion batches the gradient back through the weights using chain rule and retuns the loss (around .04 after 6 epochs, training data, cross entropy). 
weights and biases saved as .pkl files in repsitory.

outside handrawn data in the repository in png format, drawn on sketchbook.io 28X28 with a black pensil of width 2 pixels. the machine does not like my handwritten digits!! i did make them purposefully hard but its accuraccy is stagnant at 26%, even with changes in the activation and sturucture. the model doesnt appear to be overtrained as tests well on unseen mnist data, i recon there are still critical differances in my data vs the mnist, even after resizing and recentering my images. 

comparing my handwritten numbers to the minst ones using the test file, i can see that their pen width looks be slightly thicker, but i was suprised at how sloppy the data was. also, my digits look fuzzier and less dark. decided to train the model on the mnist train and testing data collections in hopes of smaller loss. 

the visulaization file uses the saved weights and biases and iterates over some inputs, plotting the activated cross product +bias for each layer as an image. i seperated some data into the classes and flashed 10 layers of each class for each layer, in sperete instances to try and idenify patterns.  

    findings:
using my visualization file over group mnist data, flashing a single plotted layer over 10 instances of the same class, i can notice some vauge patterns, especially in the third hidden layer. as the layer gets closer to the output, the images for a class get more and more similar, which makes sense as they are growing computationally closer to the same output. the first layer however was much harder to idenify any consistent patterns, i conclude this is due to slight changes in the input (placement, slan, ect.) causeing unique patterns along the many neurons in this layer. i noticed that among the first layer plottings in a class, the white to black pixel ratio seemed rather consistent.

the fact that the recenetering and resizing of my handwritten digits did not change the low accuracy of the models predictions was incredibly suprising. this leads me to belive that the pixel width of the numbers and the darkness of the pen is very important to my models idenification methods, due to the generally consistancy of these factors within the mnist dataset. this is interesting, that my model learned the mnist dataset, but not nessesarily the characters of the digits themselves, which was my goal when coding the model. i wonder how this manifests in more complex neural networks, and what problems it could cause if suffecient testing isnt done.

    next steps : 
-i could compress the images of the first and second layer to see if pattern recognition is easier
-i could run a series of the same shape with varying pen width and see how it affects the activated neurons and the confidence in the output
    ~could indipendently tweak many facotrs like pixel density, background noise ect. 