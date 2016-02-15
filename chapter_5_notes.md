#Chapter 5 – Why are deep neural networks hard to train?

http://neuralnetworksanddeeplearning.com/chap5.html

##Concepts covered

Gradient descent methods on deep networks -> different layers learn at v. different rates

Unstable gradients

Vanishing gradient problem

Exploding gradient problem


##Some quick notes	

This chapter is about the difficulties that arise when training deep neural networks, or neural networks with many hidden layers. 

It turns out to be possible to compute any function with a circuit that is just two layers deep, if you are allowed to use many-input AND and many-input NAND gates.

However, although this is possible theoretically doesn’t mean it’s a good idea. It’s a better design principle to break down problems into sub-problems and then integrate these solutions. Also, there are mathematical proofs which demonstrate that for some functions, being restricted to shallow circuits means that you’ll require exponentially more circuit elements. So, deep circuits are often better from the design perspective and are also more size efficient.

Networks with multiple layers can use the different layers to build up layers of abstraction, e.g. for image data, the first layers might recognize edges, the next shapes, and then objects. This approach has been shown to have a large advantage in complex pattern recognition problems. 

But in practice, if we use SGD by backpropogation on deeper networks that what we’ve been working with for digit classification, we’ll find that they do not perform much (if at all) better than the shallow networks. This is initially surprising. Deeper networks should be able to learn more complex classification functions and therefore do a better job at classifying digits.

The problem turns out to arise from the fact that the different layers in the network learn at very different rates. There are fundamental reasons for why the later layers in a network could be learning very well while the earlier ones get stuck during training and learn almost nothing at all. This is a big issue, since the later layers can only make use of the information that earlier layers pass on to them. If the early layers aren’t picking up on any useful signal in the dataset, the later layers are handicapped. 

This learning slowdown issue rears its head when we use gradient-based learning techniques.

The opposite problem can also occur – the early layers may be learning well, but the later layers can become stuck. There is in fact an intrinsic instability associated with learning by gradient descent in deep neural networks, which tends to make either the early or later layers get stuck (=do little useful learning, or fail to efficiently optimize weights and biases) during training. 

By delving into these difficulties, we can gain insight into how we can train deep neural networks effectively (next chapter). 

##The vanishing gradient problem

Extra layers in a neural network should help in principle, but if our learning algorithm isn’t finding the right weights and biases, they can be useless. However, they shouldn’t hurt - in the worst case, extra layers can simply do nothing. So what’s going wrong with our learning algorithm?

Recall that the gradient dC/db for each neuron affects how quickly the bias of the neuron will change during training. It also affects how quickly the weights leading up to the neuron will change (see backprop chapter). 

When we initialize the weights and biases and start training, we immediately see that the later layers are learning faster than the first ones. The later gradients are larger. As training progresses, the earlier layers still learn slower than later layers. In one example in the book, the first layer was learning 100 times slower than the last layer. This phenomenon is known as the vanishing gradient problem. It makes sense that we were having trouble training these deep networks. 

But wait, is it really such an issue that these gradients are small? Doesn’t that mean we’re near some extremum and are actually doing quite well with our training? It can’t possibly be the case that the first layer doesn’t need to do much learning because the parameters are randomized initially - we’re starting from the position of the first layer throwing away most of the information about the input image. There is clearly a lot of improvement to be made, but we’re not making it. We are stuck.

##What is causing the vanishing gradient problem?

Basically, to find the gradient of neurons in the earlier layers, we end up having to multiply together the derivatives of the activation functions of later neurons. These activation function derivatives will be <=1/4, if we’re using the sigmoid activation function. Multiplying all of these values together pretty quickly gives you a value that approaches zero. The more layers you have, the larger this effect will be.

However, since we’re multiplying by the weights as well, we might wonder that if the weights are large enough, if this could cancel out these multiplications of the many, small activation function derivatives. Indeed they can, but then the gradient will actually grow exponentially as we move backward through the layers, and we are left with an exploding gradient problem.

The problem is a bit more complex because large values for the weights lead to activations very close to 0 or 1, which again suffers from the problem of a small slope. Making w large makes the input, z, large as well. So, the weights have to be large enough to be in the activation window that still has a large slope. This can happen, but is rare, and by default a big issue for sigmoid networks is the vanishing gradient problem.
The unstable gradient problem

The problem isn’t so much that the gradients in the early layers vanish or explode – it’s that they’re unstable. This comes about because the gradient in the early layers is the product of terms from later layers. The large number of terms in the product leads to an unstable gradient. 

##Other obstacles to deep learning

Unstable gradients are just one obstacle to training deep learning, though they are an important one. Much of current research aims to identify and solve the difficulties of training deep networks.

It has been found that the use of sigmoid activation functions tends to cause the activations in the final hidden layer to saturate near 0 early in training. This substantially slows down learning. Using other activation functions can reduce this saturation problem. 

Specific strategies for weight initialization and the momentum schedule in momentum-based gradient descent has also been found to have a large effect on training deep networks. The choice of network architecture and hyperparameters are also important factors.

