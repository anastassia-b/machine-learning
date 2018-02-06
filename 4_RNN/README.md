## RNN Lecture Notes

Date: Monday, February 5, 2018

### RNNs

Last time, one character at a time generation.
Different matrices for transition and emission. At each step, a probability distribution is generated, from which we choose. The transition matrix is always the same.

```
State0 + Emission0 -> Transition0 -> (S1 -> E1) -> T1
```

The difficulty is in transmitting information from prior states for a long time.
If the beginning state was "today was sunny", it's hard to transmit through the length of the RNN.

A related problem to this is the concept of vanishing gradients.
If you have a multistep problem and each one of the steps could be optimized better.
The most direct way to optimize the output is to improve step3.
Since if you improve step1, it could be corrupted down the line.
So if you're learning with gradient descent, the beginning will be bad. The network will always be biased toward the last couple time steps, its hard for it to hear about prior time steps. The usefulness of prior information decreases. Like, if you only transmit 80% of information per step, then 10 steps is .8^10, which is like no information at the end, so why would it optimize for that.

A potential NLP task is to read a whole news article. So you're hoping that as you read the paper word by word, you're building up the state which captures up the interesting parts of the news story. Since you don't know what question will be asked, need to store info.
So it's a problem if we use the state to try to learn the entire contents of a book with all the contents. The solution to this problem is to use a technique called attention.


### LSTMs

So let's split the task up into 2 parts.
We'll take the prior state and a prior output and we'll produce an output. This is the delta.
The idea is that its easier to maintain the state from state 0 to state 1.

Q: Why not start the transition matrix to the identity matrix?
A: Well 95% to the 50th power is still nothing.

We want that tweaking s0 will have a direct linear effect on s3. Directly corresponding. Information has a highway, a path, that avoids going through any non-linear transformation.

Previously, backpropagation... when you have many layers, changes on the first set of weights might not have the same effect on the last layer.

SO before the transition matrix was all the same, but here the deltas between states are all different. The weights in the LSTMs come from the delta matrix. Once the delta state has been calculated, there will be an output. This is also a weight matrix, what do I read from this cell? We will use this information to calculate emission probabilities.

The top line is the "memory" of the cells.
There is also a `gated-recurrent-unit` which is similar. Any problem you can do with RNNs, you can do with LSTMs.


### Detail

Concatenate the previous rnn state with the emissions.
Forget gate, write gate, update gate.

Why characters instead of words?
* Fewer characters (128) than words (50K)
* Huge emission matrix. 128 * 50K is huge. then logits before emission.
* Also we need a lot more data. The problem will be overfitting.
* Do we have enough memory for an emission matrix this huge?
* Things will be really slow overall.

We want to reduce the dimensionality of words.. but all the way down to the character level.
But now we have to teach the state how to spell english words and keep track of what word its on, and where its on in the word.

What would be a way to reduce the number of words? So maybe we can replace.. project down in the space of words into simple english. Like poodle and poodles and bulldog all project to dog. So this could take 50K to 1K words.

Let's train a neural network to do this projection! This is called `word2vec`. There is also `Glove`. This is a dimensionality reduction from 50K to 1K "neurons".

Q: Can we use clustering to do this?
A: Well, since its not one hot encoded, it's like 100 affinities.

We're going to use `unsupervised learning`.
Similar words appear around... like we can replace cat with kitten and the words around it stay pretty similar.

`The large _kitten_ jumps`.

y (correct): kitten
x1: large
x2: jumps

1. embed `large` into a 50K * 1K matrix. embed `jumps` as well. E (embedding matrix)
2. use matrix to predict middle word, 1K * 50K to predict the word. We don't care about predicting the middle word in the long run, we care about the embedding.

Reduce words in such a way that you can use a reduction to predict the middle word they surround.

Unsupervised is nice because theres a bunch more data. It's unsupervised, we have a billion webpages. Plus, we only have to do this once! And we can use this matrix.

Learning something in one context and using it in another is called `transfer learning`.
