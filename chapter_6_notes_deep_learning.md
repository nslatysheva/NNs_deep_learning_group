#Chapter 6 – Deep learning
http://neuralnetworksanddeeplearning.com/chap6.html

##Concepts covered

+ Deep convolutional networks
+ Convolutions
+ Convolutional matrix/kernel/filter
+ Local receptive fields
+ Stride lengths
+ Shared weights and biases
+ Pooling
+ Max pooling
+ L2 pooling
+ The use of GPUs
+ Algorithmic expansion of training data
+ Dropout
+ Using ensembles of networks
+ Recurrent neural networks
+ Long short-term memory units
+ Deep belief nets
+ Generative neural networks
+ Restricted Boltzmann machines 
+ Reinforcement learning
+ The future of neural networks and deep learning
+ Intention-driven user interfaces
+ The role of deep learning in artificial intelligence

##Some quick notes	

We have reason to believe that deeper networks are more powerful than shallow networks, but as we saw in the last chapter, deeper networks are more difficult to train for a variety of reasons (like the unstable gradient problem). 

This chapter covers one of the most widely used types of deep networks – deep convolutional networks. We’ll go through a lot of different techniques that result in our network achieving basically human performance in recognizing numbers.	

##Deep learning for image recognition
Deep convolutional neural networks (CNNs, also ‘convolutional nets’) are one of the most common types of deep network, and are the architectures used in most NNs built for image recognition. CNNs can take advantage of spatial data – before, we were treating pixels the same regardless of where in the image it was placed, but convolutional networks leverage spatial proximity. This is an intuitively sensible thing to do, since if we treat pixels independently, we throw away useful information. As one might expect, this approach is useful for classifying images. 

This type of architecture also permits quick training because it minimizes the number of parameters we have to optimise, which in turn allows us to make deep networks with many layers that tend to be quite good at identifying images. 

Convolutional networks are sometimes called ‘convolutional neural nets’, and the neurons ‘units’, because these networks are quite far removed from biology. Still, they are quite similar to the networks we’ve been looking at so far, and ideas like as backpropagation, gradient descent, regularization, non-linear activation functions, etc. all apply.

##Overview of convolutional neural network structure

CNN architectures look a bit different to the other networks we’ve looked at. But at its core, CNNs are still networks that are composed of simple components, and we are still trying to optimize the behavior of the components by finding the best weights and biases for the task we’re trying to solve. We are still going to use a training set of data together with stochastic gradient descent and backpropagation to train the weights and biases of the network. 

A CNN has the following components:

![alt text](https://github.com/nslatysheva/NNs_deep_learning_group/tree/master/images/CNN_overview.pdf "Convolutional neural network structure")

So, the CNN first takes an image as its input (input layer). We then perform a series of computations on the image pixel values – specifically, we extract features form the image using special filters (or kernels). As we go deeper into the CNN, we move from detecting simple features like edges using the convolutional layers, to simplifying the feature detection using the pooling layers, and then move on to more abstract recognition of objects in the fully connected layer. We then output the predictions as before.

#Local receptive fields

Unlike before, we will not be connecting each pixel to 1 hidden layer neuron. Instead, we connect a group of pixels to a hidden layer neuron. The first hidden layer is composed of many neurons, and each one is connected to a different group of pixels. The group of pixels that each hidden neuron is connected to is called the local receptive field of the hidden neuron. Here is an example of a 2x2 local receptive field on a 4x4 image:

![alt text](https://github.com/nslatysheva/NNs_deep_learning_group/tree/master/images/CNN_pixels.pdf "Local receptive fields of units in the first convolution layer")
 
Instead of there being one weight between each pixel and a hidden neuron, there is just one weight from the entire local receptive field to the hidden neuron. The hidden neuron still has just one bias. 

This explanation covers what just 1 hidden layer neuron is doing. We build up the rest of the first hidden layer by sliding the local receptive field across the input image, like so:

![alt text](https://github.com/nslatysheva/NNs_deep_learning_group/tree/master/images/CNN_conv.pdf "Sliding window of local receptive fields of different units in the first convolution layer")
 
Here, we moved the window of the receptive field over by 1 pixel at each time, but other stride lengths are possible (e.g. 2). 

##What exactly is happening in the first hidden layer?

It turns out that the first hidden layer of neurons will detect simple features of an image, like edges, no matter where they are in the image. How? The weights connecting each of the shifted receptive fields to a different hidden layer neuron are all the same (these are called shared weights). Why would we want all the neurons in the first hidden layer to detect the same features (i.e. respond to input in the same way because the weights are the same)? 
The answer is that we want our CNN to have translational invariance – we want it to detect objects regardless of where they are in the image. For example, say that we have a hidden neuron that can detect a diagonal line. There is a certain type of matrix (called a filter or kernel) that can detect a diagonal line. This is a useful feature to find in other parts of the image - there could be diagonal lines elsewhere, and we want the network to respond to a diagonal line in the same way regardless of where it is. Have shared weights accomplishes this. 

Using shared weights also reduces the number of parameters in the network, speeding up training. Intuitively, having the convolutional layer make use of translational invariance greatly reduces the number of parameters it needs to get the same performance as the fully-connected model.

So, we are basically just applying a filter (the matrix of weights, which are all shared) to the whole image by sliding it along and getting the output for each local receptive field. We can pad the outer rim of the images with zeros to try to detect features on the edges. The weights of this filter are learned during training – the network has to learn what type of features to extract out from images. 

People used to design a lot of these types of filters (e.g. detecting edges) by hand. Now, the network learns them. Here are some examples of filters that detect specific features in images:

![alt text](https://github.com/nslatysheva/NNs_deep_learning_group/tree/master/images/convolutional_matrices.png "Different types of feature filters from Wikipedia")
 
So far, we can detect one type of feature – e.g. diagonal lines. This mapping between the input and first hidden layer is called a feature map. We will need more than one feature map (weights between input layer and first hidden layer) to recognize objects in images. We want multiple feature maps, which together form a convolutional layer. The feature maps are also sometimes called slices. 

Here are some of the feature maps which our network will learn this chapter.

![alt text](http://neuralnetworksanddeeplearning.com/images/net_full_layer_0.png "First layer feature maps learned for MNIST data")

These are clearly non-random, so the first convolution layer really is learning something about the spatial structure of handwritten digits. 

##Pooling layers

The pooling layers are positioned after the convolution layer. They simplify the feature map, condensing (or pooling) the information in them. This also reduces the number of parameters we have to train, which simplifies our life. Max-pooling is a common approach to information condensation – it is surprisingly simple and often functions really well. A pooling unit takes some number of first hidden layer neurons (e.g. say the values are 10,20,30 and 40), a certain size of region (e.g. 2x2 region), and simply outputs the maximum of the 4 outputs (e.g. 40). 
 
The idea is that once we detect some sort of feature in an image (by applying a filter), its exact location isn't as important as its general location relative to other features. So we can get away with summarizing the information a bit.
L2 pooling is a similar way of condensing the information in a convolutional layer, but it takes the square root of the sum of outputs in a region instead of just the maximum value.

We then perform max-pooling on each feature map, getting a pooling layer for each convolutional layer. If we had two convolutional layers (two feature maps), the CNN we are constructing would so far look like this:
 
##Adding sigmoid neurons

We can add a final fully connected sigmoid layer serves to serve as the output layer. Each neuron is connected to every single other neuron in the pooling layer. This fully connected layer of neurons extracts higher level features which can be quite abstract. We can also add more than one filly connected layer towards the end of the network. 

In modern image classification networks, it is common to see a softmax output layer (which outputs probabilities of different classes) with a log-likelihood cost function.

##Stacking more layers

We can add more convolutional layers after the pooling layer in order to win at our task:

![alt text](https://github.com/nslatysheva/NNs_deep_learning_group/tree/master/images/NN_philosophy.png "Different types of feature filters from Wikipedia")

The second convolutional-pooling layer sees the output from the first layer, which detected specific types of features in the image. The pixels represent the presence (or absence) of particular features in the original input image. So the second layer has a version of the original input image, but with specific features highlighted (and condensed a bit, due to pooling). Still, there is a lot of spatial structure left, so it makes sense to use another convolutional layer. The second convolutional layer has access to the features in all feature maps in the first layer. 

We can’t really plot the feature maps that layers after the first one see – they are too abstract to be understandable. But there is another way of looking into the network – we can visualize features in the model by examining the activations that occur when the network is presented with a certain type of image. In particular, we can look at images that result in the highest activations to see what that layer of neuron responds best to. Check out this paper to see what deeper layers of the network ‘see’. 
 
It turns out that hinge activation functions like the rectified linear unit do a better job than sigmoid or tanh functions. 

It’s a good idea to algorithmically expand our training data. We can displace images, rotate them, tint them, etc. This greatly reduces overfitting (see chapter 3). Or we can do more complicated things, like developing a model (elastic distortion) for how muscles oscillate when writing digits and use the model to artificially write out a lot more digits.

It’s also a good idea to apply dropout to the fully connected layers.  The convolutional layers already have a lot of inbuilt resistance to overfitting – the shared weights mean that the convolutional filters have to learn the entire image. This means they are less likely to led astray by idiosyncrasies in the data. 

Another easy way to improve performance is to use an ensemble of neural networks and get them to vote on classification. Ensembling is a common trick in machine learning. 

##How have we avoided issues?

What are the useful things that we’ve done so far in this chapter that have led to such great performance on the digit recognition task?

1. Using convolutional layers to greatly reduce the number of parameters 
2. Using more powerful regularization techniques (notably dropout and convolutional layers) to reduce overfitting
3. Using rectified linear units instead of sigmoid neurons, to speed up training 3-5x fold
4. Using GPUs and being willing to train for a long period of time.
5. Using a large dataset that has been artificially expanded 
6. Using the right cost function to avoid learning slowdown
7. Using good weight initializations

##Guidelines for doing deep learning	

Here are some general good practices for deep learning.

1. Start with good data. Acquiring high quality data that is large, cleanly labelled. Good input data is incredibly important for doing any type of machine learning. As the saying goes, “garbage in, garbage out”.
2. Data augmentation. It is important to have as large of a training set as possible. Depending on the data type, the training set size can be increased by translating or rotating images, introducing noise into speech data, etc.
3. Preprocessing. The input variables should generally be processed before being used. A common approach is to centre and scale the data so that the mean of each variable is 0 and the variance is 1. This makes the network’s learning work much better by keeping the weight updates low and less correlated.
4. Weight initialisation. The initial weight and bias values in the network should be initialised sensibly. This is more important for deep networks than shallow networks, where SGD can usually easily learn the correct weights regardless of initial values. In deep networks, if the initial weights aren’t set correctly, learning can be very difficult. There are several heuristics for choosing appropriate starting weights, e.g. 0.02*randn(parameter_count).
5. Optimise the learning rate. The learning rate alters how quickly the weights in the network will be updated. Incorrect rates can lead to instability in the network. A learning rate schedule, which changes the learning rate through the training phase to avoid overfitting, can also be important. 
6. Generalise your network. The performance of deep networks can be improved by averaging the predictions of multiple networks. This principle is behind the dropout technique, where neurons are randomly deleted. Another technique is ensembling, in which the predictions of several, separately-trained neural networks are averaged to give the final predictions.

##Other approaches to neural nets	

In recurrent neural nets, there is some notion of dynamic change over time. A neuron’s activation could depend on its activation at a previous time. They are particularly useful for analysing data or processes that change over time, like speech or natural language. 
RNNs are especially difficult to train because of the unstable gradient problem (gradients aren’t just propagated backwards, also backwards through time!)

Long short-term memory units were introduced to address the unstable gradient problem in RNNs. LSTMs make it much easier to train RNNs.
FF and RNN networks are fashionable now, but deep belief networks used to be hot. They are an example of a generative model. The network can be run backwards in order to generate input activations. So, a network trained on handwritten digits could potentially also write handwritten digits. Restricted Boltzmann machines are a key component of deep belief nets. 

NNs have been doing well in reinforcement learning, e.g. playing video games without knowing any of the rules and still beating world experts.

##Conclusion and the future of deep learning

The study of neural networks and their applications is a quickly expanding field that is fairly poorly understood – many questions remain unanswered, and major developments occur every year. It is therefore difficult, though interesting, to speculate on what the future of neural networks might hold. The fact that neural networks can theoretically compute any function, have powerful learning algorithms, and are modeled after human neurobiology (which we know to be powerful) has led some people to regard NNs as the future of artificial intelligence. Indeed, some NNs are now capable of performing more and more tasks (like driving cars, playing chess, or classifying images) better than humans. Massive improvements in using neural networks for language, speech, image, and video data occur on the scale of months. It is clear that neural networks will continue to be developed to carry out a wide variety of practical tasks that are useful to humans. 

More ambitiously, some have argued that neural networks will eventually be at the core of a truly conscious machine that may match or surpass the intelligence of human brains. Perhaps they could be behind the technological singularity of sci-fi and futurist fame, which is the moment when artificial intelligence gains the capability to recursively self-improve, quickly becoming much more powerful than the human intellect. Many predictions have been made for when the singularity will occur (the median year: 2040. Keep your calendar clear). For example, Google’s director of engineering believes computers will gain what looks like consciousness in a little over a decade. But of course, this is impossible to predict with any accuracy. 

It is interesting to look at specific applications where deep learning could have a real impact fairly soon. Intention-driven user interfaces are increasingly being developed – these help distill user queries into a given meaning, correcting ambiguities and giving people what they really want even if they can’t express it perfectly. This is very hard to do, but some products like Apple's Siri, Wolfram Alpha and IBM's Watson, do a decent job. It is likely that more and more products like these will be developed, which will change the way everyone interacts with computers. 



