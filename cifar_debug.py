from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
from keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# print (x_train.shape)
# print (x_train[0])
# print (y_train.shape)
# print (y_train[0])

x_train = (x_train - 128) / 128
x_test = (x_test - 128) / 128

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

x = tf.placeholder(tf.float32, [None, 32, 32, 3])


def weight_variable(shape):
    #changed stddev from 0.1 to 1.0
  initial = tf.truncated_normal(shape, stddev=1.0)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

W_conv1 = (weight_variable([3, 3, 3, 16])) / np.sqrt(27)
b_conv1 = bias_variable([16])

h_conv1 = tf.nn.conv2d(x, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1
h_conv1 = tf.nn.relu(h_conv1)

h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# let's flatten it.
flattened_features = tf.reshape(h_pool1, [-1, (16*16*16)])

W = (weight_variable([(16*16*16), 10])) / np.sqrt(16*16*16)
b = bias_variable([10])

logits = tf.matmul(flattened_features, W) + b
y_probability = tf.nn.softmax(logits)

y_true = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true, logits=logits
    ))

learning_rate_value = 0.01
learning_rate = tf.placeholder(tf.float32)
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

correct_prediction = tf.equal(tf.argmax(y_probability,1), tf.argmax(y_true,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#500 batches of 100 will equal the images we have


for i in range(1, 500):
    print(f"batch: {i}", end=" ")
    # batch_xs, batch_ys = mnist.train.next_batch(100)
    batch_xs = x_train[(i*100-100):(i*100), :, :, :]
    batch_ys = y_train[(i*100-100):(i*100), :]


    #batch_xs, is a numpy array, different from tensor flow tensors (which are nodes in a computational graph)
    #whereas numpy is just values.
    _, train_accuracy = sess.run([train_step, accuracy], feed_dict={x: batch_xs, y_true: batch_ys, learning_rate: learning_rate_value})
    # accuracy_value = sess.run(accuracy, feed_dict={x: x_test, y_true: y_test})
    print(f"train_accuracy: {train_accuracy}")
    # print(f"accuracy: {accuracy_value}")

print(sess.run(accuracy, feed_dict={x: x_test, y_true: y_test}))

#logits is the last layer
#when the z value is really large, sigmoid doesn't see how it's going to change.
#z is the sum of x * w. We want both of them to be small.
#x mean should be zero and range -1 to 1.
#currently x is 3 byte values from 0 to 256. let's knock it down.
#weights should be divided by the number of inputs to the unit.
#two kinds of weights: conv and dense.

#0. normalize inputs
#1, initilaize correctly
#2. learning rate.
