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
