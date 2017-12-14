## Neural Networks and Deep Learning

### Chapter 1

**Perceptrons**

Perceptrons take several binary inputs and produce a single binary output.
Add weights and output 1 or 0 depending on the threshold value.
Perceptrons can be used to compute elementary logical functions underlying computation, such as AND, OR, and NAND.
It turns out that we can devise learning algorithms which can automatically tune the weights and biases of a network of artificial neurons.

**Sigmoid Neurons**

What we'd like is for this small change in weight to cause only a small corresponding change in the output from the network. This is difficult with perceptrons-- in comes the sigmoid neuron.
It can take in values between 0 and 1, and it will output the sigmoid function(w * x + b). They can also output values between 0 and 1.

Compare the smoothness of the sigmoid function to the step function of the perceptron.

Now, the change in output is a _linear function_ of the changes of the weights and biases (sum of partial derivatives). Using the sigmoid function simplifies the partial derivatives (exponent properties when differentiated). Linearity allows us to choose small changes to get a small output.

In fact, this sigmoid function could be any general _activation function_.

#### The architecture of neural networks
Multilayer perceptrons (MLPs): Input layer (input neurons), hidden layers, and output layer. Even if sigmoid neurons instead of perceptrons.

Up to now, we've been discussing neural networks where the output from one layer is used as input to the next layer. Such networks are called feedforward neural networks. This means there are no loops in the network - information is always fed forward, never fed back.

```
However, there are other models of artificial neural networks in which feedback loops are possible. These models are called recurrent neural networks. The idea in these models is to have neurons which fire for some limited duration of time, before becoming quiescent. That firing can stimulate other neurons, which may fire a little while later, also for a limited duration. That causes still more neurons to fire, and so over time we get a cascade of neurons firing. Loops don't cause problems in such a model, since a neuron's output only affects its input at some later time, not instantaneously.
```

#### Learning with gradient descent
What we'd like is an algorithm which lets us find weights and biases so that the output from the network approximates y(x) for all training inputs x. To quantify how well we're achieving this goal we define a cost function.

We'll call _C_ the quadratic cost function; it's also sometimes known as the mean squared error or just MSE. Our aim is to minimize the cost _C(w, b)_, using gradient descent.

To make gradient descent work correctly, we need to choose the learning rate to be small enough that the equation is a good approximation. Gradient descent can be viewed as a way of taking small steps in the direction which does the most to immediately decrease the cost function.

An idea called stochastic gradient descent can be used to speed up learning. The idea is to estimate the gradient ∇C by computing ∇Cx for a small sample of randomly chosen training inputs. By averaging over this small sample it turns out that we can quickly get a good estimate. It works by randomly picking out a small number m of randomly chosen training inputs.

And so on, until we've exhausted the training inputs, which is said to complete an epoch of training. At that point we start over with a new training epoch.

We can _scale_ the cost function and mini-batch updates to the weights and biases: scale the overall cost function by a factor of 1/n.

Extreme version of gradient descent uses a mini-batch size of 1. This is called online, on-line, or incremental learning.

#### Implementing simple network to classify handwritten digits

This invokes something called the backpropagation algorithm, which is a fast way of computing the gradient of the cost function.

To compare our performance, let's try using one of the best known algorithms, the support vector machine or SVM. If you're not familiar with SVMs, not to worry, we're not going to need to understand the details of how SVMs work. Instead, we'll use a Python library called scikit-learn, which provides a simple Python interface to a fast C-based library for SVMs known as LIBSVM.

#### Toward Deep Learning

The end result is a network which breaks down a very complicated question - does this image show a face or not - into very simple questions answerable at the level of single pixels. It does this through a series of many layers, with early layers answering very simple and specific questions about the input image, and later layers building up a hierarchy of ever more complex and abstract concepts. Networks with this kind of many-layer structure - two or more hidden layers - are called deep neural networks.


### Chapter 2: How the backpropagation algorithm works

This expression gives us a much more global way of thinking about how the activations in one layer relate to activations in the previous layer: we just apply the weight matrix to the activations, then add the bias vector, and finally apply the σ function.
When using this equation, we compute the intermediate quantity z along the way-- there is a benefit to naming and storing it: _weighted input_ to the neurons in layer l.

For backpropagation to work we need to make two main assumptions about the form of the cost function. The first is that the cost function can be written as an average over cost functions for individual training examples. The second is that it can be written as a function of the outputs from the neural network.

Thus is a function of the output activations. Of course, this cost function also depends on the desired output y, and you may wonder why we're not regarding the cost also as a function of y. Remember, though, that the input training example x is fixed, and so the output y is also a fixed parameter. In particular, it's not something we can modify by changing the weights and biases in any way, i.e., it's not something which the neural network learns. And so it makes sense to regard C as a function of the output activations aL alone, with y merely a parameter that helps define that function.

Elementwise multiplication is sometimes called the Hadamard product (or Schur product). Good matrix libraries usually provide fast implementations of the Hadamard product, and that comes in handy when implementing backpropagation.

**The backpropagation algorithm**
1. Input
2. Feedforward
3. Output error
4. Backpropagate the error
5. Output

### Chapter 3: Improving the way neural networks learn

To improve our vanilla implementation of backpropogation:
* better choice of cost function (cross entropy cost function)
* four "regularization" methods:
  * L1 regularization
  * L2 regularization
  * dropout
  * artificial expansion of the training data
* better method for initializing weights in the network
* set of heuristics for choosing good hyper parameters


#### The cross-entropy cost function

Learning is slow when the network is really wrong - this behavior is strange when contrasted to human learning. We often learn fastest when we're badly wrong about something. But we've just seen that our artificial neuron has a lot of difficulty learning when it's badly wrong - far more difficulty than when it's just a little wrong.

It turns out that we can solve the problem by replacing the quadratic cost with a different cost function, known as the cross-entropy.

The cross-entropy is positive, and tends toward zero as the neuron gets better at computing the desired output, y, for all training inputs, x. These are both properties we'd intuitively expect for a cost function.

Additionally, the larger the error, the faster the neuron will learn (the rate at which the weight is learned).

"By now, we've discussed the cross-entropy at great length. Why go to so much effort when it gives only a small improvement to our MNIST results? Later in the chapter we'll see other techniques - notably, regularization - which give much bigger improvements. So why so much focus on cross-entropy? Part of the reason is that the cross-entropy is a widely-used cost function, and so is worth understanding well. But the more important reason is that _neuron saturation_ is an important problem in neural nets, a problem we'll return to repeatedly throughout the book. And so I've discussed the cross-entropy at length because it's a good laboratory to begin understanding neuron saturation and how it may be addressed."

There is a standard way of interpreting the cross-entropy that comes from the field of information theory. Roughly speaking, the idea is that the cross-entropy is a measure of surprise. Then the cross-entropy measures how "surprised" we are, on average, when we learn the true value for y. We get low surprise if the output is what we expect, and high surprise if the output is unexpected.

#### Softmax
The idea of softmax is to define a new type of output layer for our neural networks. It will help us address the learning slowdown problem.
The activations are always guaranteed to sum up to 1. So if the last activation increases, the other output activations must decrease by the same amount, to ensure the sum over all activations remains 1.

We see that the output from the softmax layer is a set of positive numbers which sum up to 1. In other words, the output from the softmax layer can be thought of as a probability distribution.

The fact that a softmax layer outputs a probability distribution is rather pleasing. In many problems it's convenient to be able to interpret the output activation aLj as the network's estimate of the probability that the correct output is j. So, for instance, in the MNIST classification problem, we can interpret aLj as the network's estimated probability that the correct digit classification is j.

By contrast, if the output layer was a sigmoid layer, then we certainly couldn't assume that the activations formed a probability distribution.

But we haven't yet seen how a softmax layer lets us address the learning slowdown problem. To understand that, let's define the log-likelihood cost function.

**In fact, it's useful to think of a softmax output layer with log-likelihood cost as being quite similar to a sigmoid output layer with cross-entropy cost.**

As a more general point of principle, softmax plus log-likelihood is worth using whenever you want to interpret the output activations as probabilities. That's not always a concern, but can be useful with classification problems (like MNIST) involving disjoint classes.

#### Overfitting and regularization

Overfitting is a major problem in neural networks. This is especially true in modern networks, which often have very large numbers of weights and biases. To train effectively, we need a way of detecting when overfitting is going on, so we don't overtrain. And we'd like to have techniques for reducing the effects of overfitting.

Once the classification accuracy on the validation_data has saturated, we stop training. This strategy is called _early stopping_. Of course, in practice we won't immediately know when the accuracy has saturated.

Why use the validation_data to prevent overfitting, rather than the test_data? In fact, this is part of a more general strategy, which is to use the validation_data to evaluate different trial choices of hyper-parameters such as the number of epochs to train for, the learning rate, the best network architecture, and so on.

Think of the validation data as a type of training data that helps us learn good hyper-parameters. This approach to finding good hyper-parameters is sometimes known as the _hold out method_, since the validation_data is kept apart or "held out" from the training_data.

In general, one of the best ways of reducing overfitting is to increase the size of the training data. With enough training data it is difficult for even a very large network to overfit.

**Techniques for regularization**
1. L2 regularization

Fortunately, there are other techniques which can reduce overfitting, even when we have a fixed network and fixed training data. These are known as regularization techniques. In this section I describe one of the most commonly used regularization techniques, a technique sometimes known as weight decay or L2 regularization. The idea of L2 regularization is to add an extra term to the cost function, a term called the regularization term.

Intuitively, the effect of regularization is to make it so the network prefers to learn small weights, all other things being equal. Large weights will only be allowed if they considerably improve the first part of the cost function. Put another way, regularization can be viewed as a way of compromising between finding small weights and minimizing the original cost function.

Heuristically, _if the cost function is unregularized, then the length of the weight vector is likely to grow, all other things being equal_. Over time this can lead to the weight vector being very large indeed. This can cause the weight vector to get stuck pointing in more or less the same direction, since changes due to gradient descent only make tiny changes to the direction, when the length is long. I believe this phenomenon is making it hard for our learning algorithm to properly explore the weight space, and consequently harder to find good minima of the cost function.

Let's see what this point of view means for neural networks. Suppose our network mostly has small weights, as will tend to happen in a regularized network. The smallness of the weights means that the behaviour of the network won't change too much if we change a few random inputs here and there. That makes it difficult for a regularized network to learn the effects of local noise in the data. Think of it as a way of making it so single pieces of evidence don't matter too much to the output of the network. Instead, a regularized network learns to respond to types of evidence which are seen often across the training set. By contrast, a network with large weights may change its behaviour quite a bit in response to small changes in the input. And so an unregularized network can use large weights to learn a complex model that carries a lot of information about the noise in the training data. In a nutshell, regularized networks are constrained to build relatively simple models based on patterns seen often in the training data, and are resistant to learning peculiarities of the noise in the training data. The hope is that this will force our networks to do real learning about the phenomenon at hand, and to generalize better from what they learn.

...There are three morals to draw from these stories. First, it can be quite a subtle business deciding which of two explanations is truly "simpler". Second, even if we can make such a judgment, simplicity is a guide that must be used with great caution! Third, the true test of a model is not simplicity, but rather how well it does in predicting new phenomena, in new regimes of behaviour.

In fact, our networks already generalize better than one might a priori expect. A network with 100 hidden neurons has nearly 80,000 parameters. We have only 50,000 images in our training data. It's like trying to fit an 80,000th degree polynomial to 50,000 data points. By all rights, our network should overfit terribly. And yet, as we saw earlier, such a network actually does a pretty good job generalizing. Why is that the case? It's not well understood. It has been conjectured In Gradient-Based Learning Applied to Document Recognition, by Yann LeCun, Léon Bottou, Yoshua Bengio, and Patrick Haffner (1998). that "the dynamics of gradient descent learning in multilayer nets has a `self-regularization` effect". This is exceptionally fortunate, but it's also somewhat disquieting that we don't understand why it's the case. In the meantime, we will adopt the pragmatic approach and use regularization whenever we can. Our neural networks will be the better for it.

We don't usually include the bias in the regularization term because of good reasons.

2. L1 regularization


3. Dropout


4. Artificially expanding the training data


#### Weight initialization


#### Choosing a neural network's hyper-parameters



#### Other techniques

Variations on stochastic gradient descent


#### Other models of artificial neuron
**tanh neuron**

**rectified linear unit**
