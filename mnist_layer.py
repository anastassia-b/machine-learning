# following tutorial at: https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.random_uniform([784, 128], minval=-1, maxval=1)/784)
W2 = tf.Variable(tf.random_uniform([128, 10], minval=-1, maxval=1)/128)

b1 = tf.Variable(tf.zeros([128]))
b2 = tf.Variable(tf.zeros([10]))

hidden_z = (tf.matmul(x, W1) + b1)
hidden_a = tf.nn.relu(hidden_z)

y_logits = tf.matmul(hidden_a, W2) + b2

y_probability = tf.nn.softmax(y_logits)


# define loss and optimizer: this time softmax_cross_entropy_with_logits!
y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=y_logits
    ))

learning_rate_value = 0.1
learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y_probability,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training model
for i in range(1, 1000):
    print(f"batch: {i}", end=" ")
    batch_xs, batch_ys = mnist.train.next_batch(100)
    if i % 100 == 0:
        learning_rate_value = learning_rate_value * 0.9
    # print(np.max(batch_xs))

    #batch_xs, is a numpy array, different from tensor flow tensors (which are nodes in a computational graph)
    #whereas numpy is just values.
    sess.run(train_step, feed_dict={x: batch_xs, y_true: batch_ys, learning_rate: learning_rate_value})
    accuracy_value = sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels})
    print(f"accuracy: {accuracy_value}")


# evaluating model
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_true: mnist.test.labels}))

#we could experiment with the number of units in the hidden layer or more hidden layers
#we could try increasing the batch size as we go. having as small as 1 could work for the very beginning.
#but as we get later in the game, we want a more careful update.
