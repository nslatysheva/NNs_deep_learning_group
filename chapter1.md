#Key ideas from Meeting 1 on Neural Networks (21 January 2016)
Cambridge University Data Science Society
_Notes by Natasha Latysheva & Raquel H. Ribeiro_

1. Rather than talking about thresholds of neuron, we can talk about biases. The value of the bias controls how prone a neuron is to firing (where firing = producing an output of 1 instead of 0):

2. 
High negative bias —> neuron not very likely to fire 
High positive bias —> neuron very prone to firing

3. Perceptrons, an early type of artificial neuron, make training neural network very difficult essentially because they produce binary output, which can lead to erratic behaviour since the output does not change gradually as a smooth function of changes in network parameters. Sigmoid neurons make use of a continuous activation function (logistic/sigmoid), rather than a step function, and this feature ends up having some nice properties that make training neural networks easier (i.e. output changes smoothly with input, easily differentiable, etc.)

4. What does “learning” mean in the context of neural networks? It looks like “learning” is a bit of a buzz word, which also applies to other supervised machine learning algorithms, rather than just neural networks. In neural networks, learning algorithms are designed to optimise the weights and biases, and as these gradually approach values that lead to good performance from the network, the network is said to gradually “learn”. In other words, we "learn the optimal parameters" for the task. Supervised machine learning algorithms also “learn”, e.g. when a straight line is fitted through data by minimizing mean squared error (MSE)

5. In stochastic gradient descent, we don’t feed in all the training data into the network simultaneously. We feed mini-batches of data (e.g. 10 or 100 data points) at once, update the parameters, then feed in more mini-batches. One round of feeding in all of the training data and optimising the weights and bias vectors is called an epoch of training. 

6. Just like with any machine learning algorithm, as we continue training (that is, as we keep feeding data into the network and optimising the weights and biases according to this data), the training error will decrease, but at some point the error in the validation set will start increasing due to overfitting. 