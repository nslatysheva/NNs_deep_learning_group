*Chapter 3 – Improving the way that neural networks learn*

http://neuralnetworksanddeeplearning.com/chap3.html

**Concepts covered**

Neuron saturation
Learning slowdown
Cross entropy cost function
Hyper-parameters

Softmax output layer
Network output as a probability distribution
Log-likelihood cost function

Overfitting and regularization
Validation sets
Early stopping

Weight decay or L2 regularization
Lambda, the regularization parameter

L1 regularization
Dropout
Artificial expansion of training data

Initializing weights of the network

Choosing good values for hyper-parameters
Learning rate schedules
Automated approaches, grid search

Variations on stochastic gradient descent
Hessian optimization
Momentum techniques

Other types of artificial neurons
Tanh neurons
Sigmoid neurons
 
**Some quick notes**

This chapter is about different ways to improve on the basic neural network design. 

**Learning slowdown**

 We can play with a toy example where we just try to train a single neuron to produced a desired output of 0. The initial output is 0.82, so we need it to shift its behavior quite a long ways away (remember the sigmoid neurons produce output between 0 and 1). The neuron learns relatively quickly.

However, if the weight and bias are chosen so as to produce an initial output of 0.98, which is very far from the desired output of 0, then the neuron has a much harder time moving away from this and learning the correct behavior. In fact, the animations show that it takes until about epoch 180 (that’s 180 rounds of gradient descent!) to begin its quick learning phase, where the weight and bias shift fairly quickly towards the values required for outputting a 0.

Ideally, we would actually want our neuron to learn the fastest when it’s badly wrong (very far away from the desired output). Currently, we have the opposite situation. 

Since the neuron learns by changing the parameters proportionally to the partial derivatives of the cost function, this must mean that our partial derivatives are small when the neuron is very wrong. 

Let’s take a look at the cost function, C=(y−a)2. When we differentiate w.r.t. to w and b (recall that a = sigma(z) and that z = wx + b), we find that the partial derivatives we’re interested in scale with the derivative of the activation function, and this is close to 0 when the output is close to 1. This is the origin of the learning slow down. Learning slowdown in whole networks occurs for a similar reason as the slowdown for this 1 toy neuron example.  

**Cross entropy cost function**	

We can get rid of the learning slowdown problem by using the cross entropy cost function instead of the quadratic cost function. Intuitively, cross entropy works as a cost function because it is always non-negative, and is close to zero if the actual observed output is close to the desired output. 

When we get the partial derivative of the cost function w.r.t. the parameters, we find that the rate of change actually scales with sigma(z) – y, which is the error in the output. This is contrast to the rate scaling with sigma prime. So, the larger the error, the faster the neuron learns. 

The cross entropy cost function equation was chosen explicitly so that it would have this property – the sigma primes cancel out during the differentiation, and so the rate of learning is not dependent on it (see derivation in textbook). Cross-entropy provides a lot of steepness in the initial part of our learning curves when the neuron starts out badly wrong. This is a desirable property.

Note: it is not possible to directly compare learning rates when the cost functions are different. I think this might be because, when we decide how far to step in feature space, we multiply the partial derivatives by the learning rate, but the partial derivatives are of the cost function, and this could be of any arbitrary scale (i.e. when training our network, we care about finding the relative low points of the cost function space, rather than some absolute value of it). So we can’t really compare the magnitude of the derivatives of different cost functions.

It turns out that cross entropy is nearly always a better choice of cost function, provided that the output neurons are sigmoid neurons. When we initialize the parameters of the network, one common approach is to e.g. set the mean to 0 and variance to 1, but this could by chance to lead the network to be very badly wrong for some training example (e.g. the network thinks a “5” is a “dog”). Using cross  entropy will avoid some of the learning slowdown we’d experience if we used quadratic cost.

When running the network on the MNIST data with cross-entropy, we drop the error rate from 3.41 percent to 3.18. However, this doesn’t prove that it’s the better choice, because so far we haven’t put much effort into optimizing the hyper-parameters of the network: learning rate, mini-batch size. The improvement would only be very convincing if we’d already put a lot of effort into optimizing the hyper-parameters. Other improvements to the network, especially regularization, will yield much larger improvements to our digit classification. 

**Meaning of cross entropy**	

The standard interpretation of cross entropy comes from information theory.  The textbook interprets it as a measure of surprise - we get low surprise if the output is what we expect, and high surprise if the output is unexpected. I’ve seen another interpretation that says that it’s a measure of the information content of the error - if your error contains a lot of information, then you’re not doing a good job and you can extract more from it.

**Softmax – a new type of output layer**

Instead of having normal sigmoid neurons in the output layer of our neural network, we can instead use softmax neurons. These neurons apply the softmax function to the input instead of the sigmoid function. The softmax function guarantees that the activations of the output neurons will all sum to 1 and will be positive – this allows us to treat our output as a probability distribution. So, the softmax layer outputs activations that all sum to 1. Now, in our MNIST problem, we can regard the output of the output neuron representing “0” as the probability that the digit the network has just seen is 0. This is a nice and intuitive way of looking at the output of the network. 

The softmax function is not voodoo, it is just a way of rescaling the output and making sure the values all sum to 1.  

Note: why is the softmax function a “softened” version of the maximum function?

**Log-likelihood cost function**

Whenever softmax output layers are used, it is common to see the log-likelihood cost function used. It is just the negative (natural) log of the probabilities output by the softmax layer. 

Consider when the network does a good job classifying an image – say we fed in an image of a 5, and the output neuron for “5” output a high probability (say 0.9, the network thinks there is a 90% chance of the image being a 5). Then, -ln(0.9) is ~0.1, which is a low number. If the network wasn’t so sure, then we might get –ln(0.4) = 0.9, which is much higher. 

I’ll write down a random example. Say we are classifying images into 10 digits. One of the worst outcomes would be if we had no clue what an image was, the probabilities of each class are 0.1. The cost function is  –ln(0.1) * 10 = 23. Instead, if we are quite sure (0.9) that the image is for instance a 5, and we’re equally uncertain about the other options, the cost would be –ln(0.9) + -ln(0.1/9)*9 = 40.6. Wait a second... maybe I don’t really understand what’s going on here. ***I’ll flag this up for group discussion***.

The softmax output layer with log-likelihood cost turns out to be quite similar to a sigmoid output layer with cross-entropy cost. It avoids learning slowdown in an analogous manner. 

When do we use sigmoid neurons with cross entropy and when do we use softmax output layer and maximum likelihood? Often, both will due just fine. In certain cases we might want the nice feature of the outputs being a probability distribution (e.g. if we have disjoint classes, like digits), so that can be a factor for deciding one over the other. 

**Overfitting and regularization**

“The true test of a model is its ability to make predictions in situations it hasn't been exposed to before”. This is the motivation behind validation and test sets.

The problem is that, the more parameters we have, the more susceptible we are to overtuning or overfitting to our data set, which leads to a bad solution that doesn’t have the ability to generalize to other data sets.

As training continues, our error on the training set will go down monotonically. But the classification accuracy on the test data won’t be such a smooth line, though it’ll generally get better at the beginning of training. However, after a certain amount of training (after a certain number of epochs), we reach a point where we actually start to do worse on the test data. This is the result of overfitting.  Note – we can judge the performance on the test data using cost functions as well, but it’s easier to just talk about classification accuracy (the proportion of test cases the network gets right). 

What we really care about is doing a good job on the test set. So, we need to look out for the point during the training at which overfitting dominates learning in the network. 

Overfitting is a huge problem for neural networks, especially modern neural networks (deep networks) that have a massive amount of weights and biases. 

We need a way to detect when overfitting is happening. One strategy is to monitor the accuracy on the test set, as we mentioned. A better strategy is to use a validation set, which is basically identical to the test set. It’s just that we want the test set to be completely left alone until we’re totally done training and optimizing a network, to get an impression for how our network would do in the wild. The worst thing would be to overfit to the test data.

We can use a technique called early stopping – once we see the accuracy on the validation set is starting to drop, we stop training the network. This requires some element of personal judgement. 

The validation set will also be used to optimize various hyperparameters.

Using more training data will greatly reduce overfitting, causing the training error and the test error to be more similar. This is because more data exposes the network to more examples it needs to be able to interpret, causing it focus on developing a more robust approach to the problem, in effect glossing over any particular idiosyncrasies of specific training examples. The network generalizes better. Getting more data will always cause a substantial improvement, but this isn’t always possible.

**Regularization**	

There are other ways to reduce the level of overfitting in a network other than getting more training data. We could also reduce the size of the network, for instance. However, larger networks can be more powerful, so we have an incentive to have them be larger. 

Regularization techniques are a set of tools to help reduce overfitting.

***L2 regularization***

One such technique is L2 regularization, or weight decay. Note: in case you’re familiar with L2 regularization in linear regression (also called ridge regression), this is exactly the same principle.

Essentially, we apply a penalty to the cost function. Instead of just minimizing the quantity we normally minimize, we minimize that quantity + a penalty term. That penalty term is the value of the weights^2. 

This has the effect that the network wants to minimize both the first term, the normal cost function expression, as well as the weights squared. Essentially, we now prefer to have smaller weights (hence “weight decay”), unless having large ones results in a really large improvement to the cost function. The standard explanation for why the improvement occurs is that smaller weights are “lower complexity”, and provide a simpler/more powerful explanation for the data. We will unpack this statement later. The lambda regularization parameter controls how much emphasis we give to this new weights^2 portion of the cost function. 

Small weights mean that the network’s behaviour won't change too much if we change a few random inputs, in other words single pieces of evidence don’t matter too much to the network’s output. This means that it is difficult for a regularized network to learn (or overfit to) the local noise in that specific data. Instead, the network learns to respond to recurring patterns of signal across all of the data in the data set. 

“In a nutshell, regularized networks are constrained to build relatively simple models based on patterns seen often in the training data, and are resistant to learning peculiarities of the noise in the training data.”

By contrast, networks without regularization might change its behavior quite a bit in response to small changes in the input. The large weights that this network learns carries a lot of information about the noise of that particular dataset.

It is empirical fact that regularized neural networks tend to generalize better than unregularized networks. The theory explaining why has not been fully developed. We don’t really understand why regularized NNs generalize well. 

The human brain generalizes extremely well, even from just a couple of inputs. This is probably going to be a huge area of improvement for NNs in the near future. 

L2 regularization suppresses overfitting, so it makes the gap between our accuracy on the train and test sets smaller. Interestingly, it also makes our overall error smaller – we do a better job even on the training data. 

So, regularization is a way to reduce overfitting and increase classification accuracies. Furthermore, it reduces the chance we’ll get stuck in local minima of the cost function (e.g. when weights are initialized in a way that is highly suboptimal). This has the effect that different runs can have very different results – different weight initialisations lead to very different networks. However, regularized networks are much more stable. 

Here is an approximate explanation for why we might get stuck if we’re doing unregularised training: the weights are likely to grow during training, all other things being equal. They can get so large that it gets stuck pointing in one direction, since changes due to gradient descent only make tiny changes to the direction. Our learning algorithm is then forced into doing a poor job of exploring the weight space, and so it’s consequently harder to find good minima of the cost function.

***L1 regularization***


L1 regularization (in linear regression, the analogous technique is called lasso regression) is quite similar to L2 regularization – we stick a penalty term to the end of the cost function which reduces the magnitude of our weights.  In L1 regression, the penalty is the sum of the absolute values of our weights.

Both techniques shrink the weights. But the way they do it is different. The book explains it well:

“In L1 regularization, the weights shrink by a constant amount toward 0. In L2 regularization, the weights shrink by w^2. 

And so when a particular weight has a large magnitude, L1 regularization shrinks the weight much less than L2 regularization does. 

By contrast, when |w| is small, L1 regularization shrinks the weight much more than L2 regularization. 

The net result is that L1 regularization tends to concentrate the weight of the network in a relatively small number of high-importance connections, while the other weights are driven toward zero.”

***Dropout***

This is a really different regularization technique. Instead of modifying the cost function, we modify the network itself. 

We start by temporarily deleting half the hidden layer neurons while leaving the input and output neurons intact. 

We then do the normal things with do in training for each mini-batch – forward propagating the input values through the network, then back propagating to find the partial derivatives of the cost function w.r.t. all the parameters and updating the parameter values. 

Then, we redo the process by randomly deleting another half of the neurons. 

When we drop out different neurons, it’s like we’re training different networks. The different networks overfit in different ways, and if we average them somehow (e.g. majority vote on classifications), it’s like we average away much of the overfitting.  The net effect of dropout is to reduce overfitting. 

Dropout has been a very successful in improving the performance of neural networks, especially large and deep ones where overfitting is an even bigger concern. 

***Artificially expanding the training data***

Performance improves as we use more training data. An easy way to get more training data is by artificially expanding. For example, if we’re training on images, we can expand by taking the images and rotating, shifting, distorting, skewing or tinting them. These can be very simple distortions (e.g. rotate each image 15 degrees) or very complex (e.g. model the random oscillations in hand muscles to “write” slightly different numbers from what you have). Either way, we try to expand the experience of the network by exposing it to the types of variations found in real data.

If it’s speech data we’re interested in, we could apply noise, or alter the speed of the speech clips to artificially expand the data. 

**Other notes**

More training data can sometimes compensate for differences in the machine learning algorithm used, e.g. the SVM outperforms the NN when we have tons and tons of data (in this example).

We can’t really ask questions like, is algorithm A better than algorithm B? It’s so dependent on the training data.

Overfitting is a major problem for NNs and will become a bigger problem especially as computers get more powerful and we have the ability to train larger networks. It’s an extremely active area of NN research.

**Weight initialization	**

Previously, we initialized our parameters using independent normal random variables with mean 0 and standard deviation 1.

This worked well but there are better techniques. 

The problem with our approach is that, when the weights are sampled from the standard normal distribution, the input z leading up to a given neuron has a fairly high chance of being either extremely negative (e.g. less than -5) or extremely positive (say, >5). See the explanation in the book for why this is – it rests on the fact that the variance of a large number of standard normal variables is very large. These very negative or very positive inputs will lead to activations very close to 0 or very close to 1. As we saw earlier (chapter 2), these neurons will have a very hard time learning – they’ll be stuck on 0 or 1.

The thing to do is, instead of initializing weights and biases from a normal distribution with mean 0 and variance 1, we sample instead from a normal distribution with mean 0 and a variance 1/sqrt(number of neurons sending input). 

This has the effect of making extreme inputs to a neuron very unlikely.

**Choosing good hyper-parameter values**

So far we’ve been choosing what looks like arbitrary values for our hyper-parameters, like the learning rate eta and the regularization parameter lambda.

Other hyper-parameters are: the number of layers, the number of neurons in each layer, the step size, the choice of cost function, the choice of the output layer, the encoding of the output, the choice of activation function, the number of epochs to train for, the size of mini-batches, and the initialization strategy. 

Knowing what to set these parameters to is hard, and if the network is doing poorly, we wouldn’t really know which ones are causing the poor performance and which direction to change them in. 

When setting up a neural network, the first important challenge is to achieve any non-trivial learning – we want the network to do better than chance. This can be surprisingly difficult. The priorities at this stage is starting with the smallest network possible, on a very simplified data set, and monitoring how you’re doing often in order to get into the right ballpark of hyper-parameter space.

We continue gradually, tuning hyper-parameters in order to improve performance. We judge performance by running the model on the validation set.

However, this makes it sound easy – we can easily end up with a network that learns nothing even though we spent ages trying to change hyper-parameters. It’s so important to get quick feedback from the system at this stage, so you can try out tons and tons of options. This will help improve your chances that you find a network that can pick up on a signal. From there, you can often get rapid improvements by tweaking hyperparameter values.

We can try different values for hyper-parameters and then plot curves for network error by epoch for the different values.

In practice, although we choose hyperparameters like the mini-batch size and network architecture using the validation set, the learning rate can actually be chosen by looking at error during training. “Its primary purpose is really to control the step size in gradient descent, and monitoring the training cost is the best way to detect if the step size is too big.”
	
We can use early stopping (= stop when error starts going up on the validation set) to determine the optimal number of epochs.  Actually, since the validation error will be a bit jumpy, more precisely we terminate training if the accuracy doesn't improve for a long time. (e.g. the no-improvement-in-ten or no-improvement-in-fifty epoch rules). 

Choosing mini-batch size: “Choosing the best mini-batch size is a compromise. 

Too small, and you don't get to take full advantage of the benefits of good matrix libraries optimized for fast hardware. 

Too large and you're simply not updating your weights often enough. 

What you need is to choose a compromise value which maximizes the speed of learning.”

**Learning rate schedules**

It turns out to often be advantageous to alter the learning rate during training instead of constantly keeping eta the same. Early during training, the weights are badly wrong, so it makes sense to have a high learning rate. Later, we can reduce our learning rate as we are closer to better weights and want to fine tune them.

How can we set the schedule? We could keep it high until the validation classification accuracy gets worse (suggests that instead of approaching a minimum, the high step size bounces around the cost function) and then decrease it (by e.g. a factor of 10).

**Variations on stochastic gradient descent**

Hessian methods for traversing the cost function can converge to a minimum in fewer steps than SGD. But it is difficult to do in practice, since the Hessian matrix (full of a bunch of second derivatives) is so massive. 

Momentum-based gradient descent also includes information not just on the gradients, but on how fast they are changing (these are the second derivatives). It has concepts of position, velocity, momentum and friction applied to something like a physical ball in the cost function valley.

The idea is that if the gradient is in (roughly) the same direction through several rounds of learning, we speed up movement in that direction. So, this can work faster than normal SGD. But that danger is that we can overshoot. However, we can also get out of local minima. If there is more friction, we can’t build up velocity as much.

The momentum technique is widely used, often speeds up learning, and doesn’t require much modification to the SGD code.

**Other models of artificial neuron**

Tanh neurons are a common choice for artificial neurons, replacing the sigmoid function by the hyperbolic function. It looks a lot like a sigmoid function, but its output ranges from -1 to 1.

Unlike in sigmoid neurons, it can avoid the situation where all the changes to the weights to a  given neuron have to be strictly positive or strictly negative. It avoid a systematic bias for the weight updates to be either all positive or all negative. 

Another common type of artificial neuron is the rectified linear unit. It’s linear after a certain threshold.

We do not yet have a deep understanding of when certain neuron types are preferable to others. Sigmoid and tanh neurons slow down or stop learning once they saturate (when output is near a 0 or 1), but ReLu neurons won’t ever experience learning slowdown. 

