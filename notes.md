## Notes

### November 9, 2017

Naive bayes can be wrong in so far as its assumptions are wrong.
So thats how we can pick better phi values which are doing for the likelihood equation.

So we can square each of the two conditionally dependent words, or make one 1.0…

No absence phis anymore.
Base odds plus all of the absence features.
Now when a word is presence. We’ll put in the presence ratio, and we’ll kill the absence feature. They’ve been folded into phi0.

Phi is odds.
Let’s just assume everything is absent. Then, when we do see a word, we’ll cancel it out.

Phi0 is the original odds that an email is spam.
phi’ is the feature probability ratio. The ratio that a spam email would have the word over the probability that a non-spam email would have the word.

All the words, of which there are m of them.
From the odds we can derive the probability equations.

ODDS RATIO in logistic regression.

Maximizing the likelihood. Now with less stuff.

Maximizing log likelihood. The rule of log is that it turns products into sums.
A lot of the multiplying has become adding.

Gradient descent. Incremental change in the phi value could even fluctuate a lot. Theres no definition of small anymore. Then your step could actually be huge. So thats a problem. Because gradient descent wouldn’t be making the small steps that it needs to. So we want to get rid of the products. So we will reparameterize the equations!!!
We take the log of the phis to get thetas.

Exponentiate and log are the opposites. Adding a lot of thetas together and just exponentiate.
Much better, since its just addition. So its still accurate!

Logistic function. Sigmoid function.
The reparamizerixed likelihood function!!!!
This error function is called the cross entropy error. Minimizing the cross entropy error is equivalent to maximizing the likelihood.
Let’s find the best thetas!!!
Maximizing this is actually minimizing the error of it.
-log probability!!! Is the error function. This is related to entropy!!!

Probability of the dataset_ probablility that the model picks the spammiest email from the dataset.

Why this sigmoid model??? There is a logic to this thing. It’s related to the RIGHT MODEL under a certain assumption. This also suggests why this model could be better than naive bayes. This model is still LINEAR. Just cause I’m free to choose the thetas…

Is this a generalization of naive babes without assumptions? yes.. but…
The naive bayes is a GENERATIVE MODEL. It allows me to answer any kind of probability query.
These thetas are chosen FOR THE SINGLE TASK OF GIVEN THESE WORDS THAT THE EMAIL IS SPAM?
One thing that the naive bayes model is..
THIS IS A DISCRIMINATIVE CLASSIFIER (conditional classifier, with thetas deciding on spam email) while naive bayes is a GENERATIVE MODEL.

The discriminative one where if they go to a website in incognito mode.. we’re assuming that they didn’t.
For medical diagnosis… other tests that you didn’t run are not equivalent to running tests and getting negative.
GENERATIVE MODELS are in the medical domain, while discriminative models are used in a lot of other domains.

### November 14, 2017

 Neural networks are networks of logistic regression models feeding into each other.
Boolean variables take binary variables and output binary variables. (Like And and Or).
Any Boolean functions can be written in terms of just a few.
Not, and, or, nand.

64 bits in, 32 bits out -> And and floating point division.

So, emails can be written as a vector (boolean circuits), and then outputs 1 or 0.

A logistic regression model could learn how to be a boolean circuit.
The model has a linear decision boundary.

Pair features that words are spam. One word might not mean anything, but both of them together, do.
Add new features which are the combo, then run the logistic regression with the new additional features.
Thats O(N^2) more features. Most will be totally useless.
Every feature that we add adds capacity to the model, but that’s bad since it’ll be too specific.
So with limited data, it’s bad. The weights will be too specific, it’ll memorize.

The goodness of a network/circut is how often it misclassifies (error).
But making one change at a time is inefficient for 2 reasons. Using gradient descent would be good.
1. All of our choices are discreet. (But in a logistic regression model, we can change the theta a little bit). The wiring choices are discreet too.
2. Derivatives of zero.

How to change? Logical gates can be approximates by logistic regression models!
Thetas are implicitly explaining how to wire things.

We want to do 2 things with a neural network:
1. inference. Forward propagation.
2. How to set the the theta values. Back propagation.

Z values are pre-activations.

Multiply by matrix, run the activation function, multiply by matrix, run the activation functions.
Tensor flow: you give it the architecture, and you give it the data and it find them.
It’s pretty low level. You have to tell it that you give it the matrix of M by N versions.

There’s an easier, higher concept version called Keras.

### November 15, 2017

Machine learning individual has intuition for the type of model.
Each layer goes down in powers of 2 or 10.
Image problems use convolution, dense connections.
Activation function, the sigmoid function is not the most popular anymore.

Butterfly effect. But rectified linear unit function… is more proportional error.
The amount of signal you’re getting per layer is going down 1/4th each time, if you’re in the most max change part.

Don’t want to use sigmoid functions for deep networks.
Sigmoid was the standard for decades because there was only 1 hidden layer.
Sigmoid is more similar to digital logic gate.

Sigmoid gives a value of zero to 1.
Alternative Hyperbolic Function (from -1 to 1) when we get to long short term memory.

Any Boolean function can be represented by a single layer of many inputs. Because it’s like a lot of Ands and an OR

50-100 convolutional neural layers for distinguishing tons of types of things.
Just between cats and dogs could be 1 hidden layer with like 512 input variables.
If you have 0 hidden layers its just a logistic regression. Cause just input and output.

Convolutional layers are not as expensive as densely connected a layer.
On our machines.. 4-5 layers with cpu.

For BACK PROPAGATION.
Partial derivatives. We need to know the error with respect to theta.
Gradient descent.

We can’t change z3 directly, but we can change the b’s and the thetas (or a’s?).


### November 16, 2017

Convolutional Neural networks for images.
x_tensor: (60,000, 28, 28). Images are 28 by 28pixels.
y_tensor: (60,000)

An image is 0-255 values as intensities. We want to flatten the image to be one long value.
Our layers are 1 dimensional, so we'll use a Flatten layer.

The z values are the same: the weighted sums of the value of each node. linear sum.
If there are 2 output classes, we had binary cross entropy.

logistic regression because there are no hidden layers. categorical cross entropy.
optimizer is stochastic gradient descent. one hot / to_categorical.

We're testing categorizing images as digits. Linear regression alone works with a pretty surprisingly good accuracy. Kernel_initializer = "zeros" would set all the thetas to zero. Our concern would be if all the derivatives would be zero.

 Learning the spatial representation of the pixels to guess a "5". It's recovering some information about the shapes of numbers.

We want the network to learn the concept of horizontal lines regardless of its location.
But it's hard to learn that if the network doesn't understand that various horizontal lines.. like 100 sunrises, and you ask on the next day what will happen, I don't know it's a totally different day.
But want it to make that connection. To extract a general commonality that is useful.

Filter dimension: receptive field of a filter.
We intuit: Number of filters and filter dimensions!

Learning horizontal and vertical lines.
At the end of the day, a dense layer can do what a convolutional layer does...
Sparse connectivity: the output values of the output image are only connected to a small number of values in the input image. It's much more efficient!

Anything we can do to prep the image (crop, resize, rotate).
Last note: new network recently which is the capsule network.
Currently we are looking for patterns in a 2 dimensional image. But if our image is far away, or rotated.
We haven't taken advantage of the fact that 2d images are from 3d worlds. But currently we live in flatland. There's a problem with pose.

Change 0-255 into canonical range of 0 to 1. This really improved the classifications. The error derivatives.

Next up will be: Recurrent neural network that read continuous data. Like text or sound.


### November 28, 2017

Now, we will learn about recurrent neural networks. recurrent tasks like series of words or letters.

**Markov Models**

A generalization of Markov chains and hidden Markov models.
256 letters (ascii) codes. Words in the vocabulary.
The graphical.

Two problems.

1. This doesn’t represent the way sentences are really formed.
2. High dimensionally of the vocabulary.  So the model will not think that some words will never follow other words.

**Hidden Markov Models**

There is a hidden state. Such as what is in your head.
The states are hidden from us. There might be too many to consider.
But we can just assume that there are 128. states in your brain.

Emission probability.
The initial state will only emit one word. It is a random choice based on the emission probability.

Graphical models. Such as trees are used for medical diagnosis.

How do we learn this if an essential part is unknown to us (the hidden states)?

**Recurrent Neural Networks**

Just like with neural networks, we can begin with random Initial state probabilities, random transition probabilities, and random Emission probabilities (as long as the rows sum to 1).

Prediction, Smoothing. There are 2 things that could tell us information, is the state before and the state after.
How to figure out smoothing? Expectation Maximization.

So we it up. We can calculate the state probabilities.

We want to pick the model that makes the past the most likely out of unlikely events.
Will this reach equilibrium? It will reach some non-improving state.

You monitor the error rate, error metric (loss function) to the number of epochs.
The loss converges but maybe not the models. Unsure.

Look at the implementation of HMM in repository of nodes.

What are the problems of the HMM?
1. If the state space is small... but if we were to increase it, then the transition matrix quadruples in size. And if we have a lot of states and not enough data we won't learn. (Note).

Trying to aggregate the brain states in a way which is predictive of the words. Comes up with categories, though might not be easy to tell. Taxonomies of documents.

This model is _generative_. Which is nice. We can use category 1 to generate a document.
Where as for image recognition, its _discriminative_.

Denoise autoencoder.  

* (Note). So we want to be more efficient in the way we learn.
What if we have "large" and "animal" categories. How about a "large animals" categories. I want to use prior knowledge about largeness and animalness to learn a new category.

We want to move away from discreet state matrices and move to neural networks.
The # of state units is 128.

The states is a combination of the pure state. We will have transition weights.
This kind of network we train with brack propogation and cross entropy error.

The only difference is that the transitions are the same between each layer. So are the emission probabilities.
To generate text, you could randomly initialize the state, randomly sample a word, and calculate, then sample.

This is effectively a hidden markov model with a hidden state space. This just allows mixtures of the pure states.
It's the same dimensionality for the transition matrix. So the transition is a function, in mixing the states.
It s a weight matrix now, multiply the vector.

The most common activation function is Relo. But softmax has meaning. Its the logistic function, probabilities that sum out to 1. The state will be chosen from that probability distribution.

Different ways of training models.
1. Closed form solution.
2. Gradient descent.
3. Expectation maximization.
4. genetic algorithms. kills off or share parts of code.
  - but hard to determine how to share code that doesnt destroy it completely.

### November 29, 2017

Recap and correction from yesterday--
We were trying to use sequence models to:
* generate text
* predict next token in the sequence
* smoothing

If we want to generate text for HMM its simple: we pick the initial state, then we use emission probability to emit a word, then use transitional probability to generate a new state. Generates word by word.

Predicting the next token is a bit different. If you see "to be", and you want to guess next word, you don't know the hidden states, but you want a probability distribution on the 3rd state. Because that determines what word comes out. To figure out that, you need the 2nd hidden state, and work backwards.

There's some algebra you can do to calculate it. How inference works.

For RNNS, We have an initial state.
Problem with other model: even if its 99% accurate... as soon as you get a wrong word, you're probably going to get every other word wrong. Overall your accuracy is low. There's no correction, no way to get back on track.

These are not inferences, they are functions! The word is deterministically produced. Because the 3rd state is determined. While the word is a probability distribution, its always the same one. So you might get different words sometimes, but not what you want. It'll always try to write the same sentence.

Generation: will struggle to learn the model at all, and no way to predict right word.

So the right picture:
It should have as the inputs, the REAL words. Out on top should be the guesses of the words. So it has a way to pull it back. And each is a function of the priming words, instead of the state. You can also predict the next token properly.

Could also turn into generative mode, if you feed the words back in.

---

When you have RNN, the input below has a high dimensionality. For example, 50K words and 128 units. The weight matrix is huge.

If you have elephant 1000 times, and if you only see gazelle 10 times, you dont know how to set the weights correctly for the transformation. We'd like to capture elephant and gazelle.

Latent dirichlet allocation.
We want to find a word embedding.
A word2vec.

We want to transform our words from being 50K dim vectors, to 256 dim vector. So instead of being a one hot encoding of individual words.
Euclidian norm. So from sparse (everything is 0 except for 1 spot) representation to dense representation (distributed).

We want to encode similar words in similar ways. So we've compressed it.

The problem with the 50K encoding. Every word is the same "distance" to the same one. One alternative is to use numeric codes to represent the words. If we assign those words randomly then.. just because the numbers 3 and 4 are close together, if 3 is house and 4th is gazelle, they have nothing to do with each other. So while its more efficient, it still doesn't mean that things close together are conceptually similar. This is still bad since the emission is a LINEAR function. So things that are close, they'll stay close. So in 50K all are different.

So let's figure out dense representation. We'll need more than one dimension. Like an elephant and a car. Once we figure out word embedding, we'll use that as the inputs.

**Representation Learning**

We want a better representation of words, where similar words have similar vectors of values.
One thing that is common in RL is to use _unlabeled data_. The task we'll be performing is an unsupervised task.

`large elephant trods slowly`

So let's look at the two words next to elephant. We could look at a large window, and give more weight to the closer words. We want to train a network to use `large` and `trods` to figure out `elephant`. We really care about changing the embedding weights (and the weights connecting to elephant). Same way with back propagation..

That's why google wants to digitize all books. They want to make a model over all books ever written. There's a ton of unlabeled data (as opposed to labeled cat photos).

This is semi-supervised learning. Maybe. If we want to predict the next word in the sequence, there's a correct word. Like asking "Is the capital of USA Washington DC?".

Let's see some variants of RNNs.
"Is Ned cool?" -> Then answers the question.

**Long short term memory network**

Let's say I have a really long question, and then yes/no.
With gradient descent, we tweak all the weights simultaneously. But here we're going to tweak the weights a little bit. like in the last state and the yes/no node. There are a lot of weights even just there, that matrix. But even if we tweak these..

Even deep feedforward networks. Like telephone, the longer the chain, the worse the implementation. Fundamentally this is why its hard to train deep neural networks. For RNNS, the problem is even worse. Because in deep ones, the weight matrices are different, BUT in RNNS its the same weight matrix, so the problem compounds.

Even if we change the weight matrix in the beginning, we can't predict, its so scrambled. So the network is biased to use the later information. The problem is with the training, it will disregard a source of truth further away.

For LSTM, we'll still have inputs in each time, but instead of a layer of nodes that is the function of the state, we're going to have a more sophisticated cell, which we still feedforward. These cells are components, aggregates, they have a structure. We have the prior state, which comes into the cell, combined with the new input (concatenated), comes up with a new state. Normally if we fed this forward, that'd be a normal RNN. But we're also going to calculate a 256 dimension ~~sigmoid~~ tanh. New state is a delta of the prior state.

There's a path by which a change in the beginning could have a fast/clear route to send signal all the way down. It's not corrupted by intermediate information. It's typical to use Relu (0 - +). But it's also a bit problematic since the state can only grow and grow. Weird. So we also use tanh. Which is like a sigmoid, from -1 to 1. Like a volume nob, an operation.

Different kinds of networks.
Prediction at every time step, another prediction after a couple steps, but what about translation?

The output is variable length if I want for the end to figure it out...or if I output it at each step, that's also not good.

So we'll get a state at the end. Which we'll feed into an RNN, which will generate french words. This is **Encoder/Decoder**. Sequence to sequence. Neural machine translation. State of the art. Harder for simultaneous interpretation. Encoder/Decoder is not a good fit for simultaneous.

Question/Answer problems with memory and attention. Like outputting notes from each chapter instead of one "book" state at the end. And then getting a question state and feeding in the notes, to get a final answer.


### November 30, 2017: Autoencoders and GANs

Representation Learning.
The space is scrambled up, but we'd like a linear regression model with Cat photos on one side and not on the other.
The euclidean distance. You can do 30 x 30 image with a 900 dimensional vector. Images that contain cats are not going to fall...
The original representation doesn't help us. So we want to transform the data points... find new representations maybe different number of dimensions. But we can change the meaning of the space.

The promise of neural networks is to discover new representations. (Today is like a parallel to word2vec yesterday, but with images).
In a sense all of deep learning is to learn about representations. Why do we prefer one representation over another?
We prefer a representation if it sets us up to do a good job. Training signal of yes cat or no cat. This is going to drive the re-featurization process.

If we're looking for images of cats online... they are probably mostly unlabeled. We'd like to be able to use the photos which are unlabeled.
How can use use the (large) unlabeled and (small) labeled data to make a representation. This is semi-supervised learning with autoencoders. This is an important historical technique because it was the first way we trained deep networks.

The idea is we have an input. We don't have a training signal. We might want to map it to a representation. And the output that we want is the input again. In a sense, we're encoding something so that later we can decode it and get ourself back again. This is like MP3 compression. So autoencoders are going to learn to encode and decode.

How do we know we've done a good job. We'll take the L2 distance between the input and output. This is the generalization of the distance metric in the real world. But its not necessarily semantically meaningful. It's not the ideal loss metric but its one. (Examples: audio, throw away high frequency that dogs can hear but humans cant, or image distortions, its hard to know what humans find easy or hard for comp model).

Unsupervised pre-training of a neural network. Instead of starting the weights randomly, we start with the weights from the autoencoder.

**Several types of Autoencoders**

* **Undercomplete**. Make the representation smaller (like word2vec embedding). Advantages include: representation is smaller, so you have a smaller inputs, so thats a significant reduction in reduction of weights. Good for memory usage and compute time, also helps not overfitting and learning values. Another advantage is it is a representation that plays nice with linear functions. E, D are linear transformations. It must have found a representation where applying a linear function restores semantic values. It couldn't have stored the data in a way that a linear function can't understand. Thats what we want in the end, linear separability of the classes. Therefore, similar images are likely to be closer together. Greedy layer unsupervised pre-training.
  * Question. Well, D2 is not just the inverse of E2. But that's too simple (there's a relu). Talk about PCA. We want to store a 2 dim dataset with 1 dim. The line is like the decoding function. The 'l' value is the encoding (how far away the point is from the line). If E and D are linear functions, its like PCA for each step. To make things more sophisticated, we often use a non-linear activation function. In this way we can use non-linear relationships (with a relu function or a deep autoencoder with more hidden layers). Manifold is the curvy plane/line that is analogous to the PCA line. The autoencoder, when projecting onto the manifold, will have a low reconstruction error. We'll still use a length to figure out where on the manifold you are.
  * This would be like 10 lines of keras code, super easy. There is a dataset called Mnist. It's the digits. We want to keep 10% of the data with labels, and ignore the remaining 90%. Train an autoencoder with 90% and train a neural network with the remaining 10%.

  -> Tangent: Transfer learning. Train network to find cats. Then change the last classification layer to find cats. Actually does a pretty good job. Groups like google and microsoft publish the weights.

* What if I have an enocoding that is just as big? Cool, I'll just apply the identity function. What if instead of reducing the dimensionality, I use a **regularization penalty**? For example, for using non-zero values. It's almost in a sense.. **This is a sparse autoencoder**. You can consider it a variable length encoding. With this way, you don't force everyone to have the same dimensionality. So you can say, you can use however many words you need to describe this image.

* **Denoising autoencoder**. Put in a corrupted image (with gaussian noise). There's a matrix for convolution, for sharpening images.

* **Variational autoencoder**. This is a competitor to generative adversarial networks.

**Generative Adversarial Networks**

Another way to find representations and a lot of other cool stuff. We've gotta have another way of comparing the output to the input. The L2 loss function is not semantically meaningful.

Forget about internal representation. Let's talk image synthesis.
Generator is trying to generate fake images and the Discriminator is trying to detect fake images.
We will train the discriminator with the output of the generator. You can train these together.
If your text was "old man crossing the street"... to just use L2 to test generator would be problematic, since a thousand different ways to represent this text in a photo. The L2 loss doesn't respect the idea that two images could be totally different but be about the same thing. Ears are visually salient to us, but L2 loss tend to blur out the ears.

In a sense the discriminator is learning a loss function for the generator. Where are we going to get a better distance metric from? What do we think are semantically meaningful, to inject our knowledge? This has flipped it around. Actually, I'll just pose a simple adversarial function between these two things. Theres no loss function at all. Theres only cross entropy, and you'll figure out the loss function yourself. The generator will try its best with the loss function you learn.

GANs favor answers which are really sharp. As opposed to L2 loss. I have to give a sharp answer. It's more all or nothing. It's good for super resolution. You can also use GANs as a feature extractor. The advantage is not that the features are used to reconstruct a blurry image, but that there are more semantic meanings to the representation than an autoencoder. It has a sharper distinction, different criteria for representation. On the other hand, it could lose a lot of information if its mistakes that the generator makes...
