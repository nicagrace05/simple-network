# simple-network
welcome:
i am instersted in gaining further insight into the methods neural networks use to generalize, and what factors are most importnant to maximize accuracy. its proving a lot harder than i thought to manage errors surrounding slight differnences in referncial and created datasets. 

done:
class neurons intitizales the weights and biases around zero before batching the mnist data through the layers using ReLU activation. The ouput layer uses softmax, and then the backprop funtion batches the gradient back through the weights using chain rule and retuns the loss (around .04 after 6 epochs). 
weights and biases saved as .pkl files. 

outside handrawn data in the repository in png format, drawn on sketchbook.io 28X28 with a black pensil of width 2 pixels. the machine does not like my handwritten digits!! i did make them purposefully hard but its accuraccy was 26% as of 11/10. the model isnt overtrained as it gets 100% on unseen mnist data, i recon there are still critical differances in my data vs the mnist, even after resizing and recentering my images. 

the visulaization file uses the saved weights and iterates over some inputs, plotting the activated cross product +bias for each layer as an image. im working with a few data sets in here, mnist is the most sound.

comparing my handwritten numbers to the minst ones using the test file, i can see that their pen width looks be slightly thicker, but i was suprised at how sloppy the data was. decided to train the model on the mnist train and testing data collections in hopes of smaller loss. 

findings:
i finally am seeing what i was looking for!! using my visualization file over unseen mnist data i can identify meaningful patterns in the whitest and blackest neurons of the plotted images. the whitest ones represent the neurons that are idenifying patterns, and i see them in the first layer finding angled lines, it seems to spcifically favor postivley sloped ones. dark spots occor in the first layer where there is no pen. 

in the second layer it displays randoma fully white pixels, but maybe its becuase the image was condensed into less pixels and close ones get lumped togther. it looks like the neurons might be more conserned with vertical patterns, and you can clearly see the dark spaces looking for non existant curves. 

the third layer is more abstract and unpredictable, more observation needed.

the fact that the recnetering and resizing of my handwritten digits did not change the low accuracy of the models predictions, was incredibly suprising. this leads me to belive that the pixel width of the numbers, and possibly factors that im not even noticing is incredibly important to this models generalization of digits. 

next steps : 
-run the spereated mnist data on the visualisation machine and make observations about the layers of each number
-measure pixel width of mnist dataset and redraw your own 
-i could run a series of the same shape with varying pen width and see how it affects the activated neurons and the confidence in the output
    ~could indipendently tweak many facotrs like pixel density, background noise ect. 