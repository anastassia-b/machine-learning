# following tutorial at: https://www.tensorflow.org/get_started/mnist/beginners

# Softmax Regression
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#images are "x" and labels are "y"
#softmax is natural here because list of valus 0 to 1 that add up to 1.

#softmax regression
#1. add up the evidence of our input being in certain classes (weighted sum of pixel intensities)
#2. convert that evidence into probabilities

# read data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# create variables for model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_probability = tf.nn.softmax(tf.matmul(x, W) + b)

# when we add up the evidence, its -infinity to infinity.
# we exponentiate in order to get everything on the positive side.
# after that, we worry about normalization. thats the denominator of the softmax.
# z is like log odds, before exponentiation.

# define loss and optimizer
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    -tf.reduce_sum(
        y_true * tf.log(y_probability),
        reduction_indices=[1]
    ))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


# training model
for i in range(1000):
    print(f"batch: {i}")
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys})


# evaluating model

correct_prediction = tf.equal(tf.argmax(y_probability,1), tf.argmax(y_true,1))
# correct_prediction is a vector of 100 boolean values

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))
